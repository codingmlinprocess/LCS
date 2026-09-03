import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoProcessor

from simlingo_training.utils.logging_project import setup_logging, sync_wandb

from simlingo_training.config import TrainConfig
from simlingo_training.callbacks.visualise import VisualiseCallback

from simlingo_training.dataloader.dataset_driving import Data_Driving
from simlingo_training.dataloader.dataset_dreamer import Data_Dreamer
from simlingo_training.dataloader.dataset_eval_qa_comm import Data_Eval
from simlingo_training.dataloader.dataset_eval_dreamer import Eval_Dreamer


@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading when in debug mode
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
    
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False
    )
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    
    # Create log directory based on experiment name
    log_dir = f"logs/{cfg.wandb_name}_{cfg.name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Add CSV logger for metrics (saves to CSV files)
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="csv_logs",
        version="",
    )
    # Override the experiment's save method to handle hyperparameters safely
    # This prevents YAML serialization errors with complex objects
    original_save = csv_logger.experiment.save
    def safe_save():
        try:
            # Try to save with hyperparameters
            original_save()
        except (AttributeError, TypeError, ValueError) as e:
            # If hyperparameters can't be serialized, save without them
            print(f"Warning: Could not save hyperparameters to CSV logger: {e}")
            print("  Continuing without hyperparameters (metrics will still be logged)")
            # Clear hyperparameters to avoid future errors
            if hasattr(csv_logger.experiment, 'hparams'):
                csv_logger.experiment.hparams = {}
            try:
                original_save()
            except:
                pass
    csv_logger.experiment.save = safe_save
    
    # Also override finalize to handle errors during cleanup
    original_finalize = csv_logger.finalize
    def safe_finalize(status):
        try:
            original_finalize(status)
        except (AttributeError, TypeError, ValueError):
            # Ignore errors during finalize
            pass
    csv_logger.finalize = safe_finalize
    
    loggers.append(csv_logger)
    print(f"CSV logger initialized: {log_dir}/csv_logs/")
    
    # Optionally keep WandbLogger if enable_wandb is True (but disabled by default)
    if cfg.enable_wandb and not cfg.debug:
        try:
            wandblogger = WandbLogger(
                project=cfg.wandb_project,
                id=cfg.wandb_name,
                name=cfg.wandb_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                resume=resume_wandb,
            )
            wandblogger.watch(model)
            loggers.append(wandblogger)
            print("Wandb logger initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize WandbLogger: {e}")
            print("Continuing with local loggers only")
    
    print(f"\nLogging to: {log_dir}")

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
    elif strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)


    # Ensure checkpoint directory exists
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Normal mode: save checkpoint every N epochs
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        monitor=None,
        dirpath=checkpoint_dir,
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=cfg.val_every_n_epochs,
    )
    print(f"✓ Checkpoint normal mode: saving every {cfg.val_every_n_epochs} epochs to {os.path.abspath(checkpoint_dir)}")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
    ]

    enable_visualization = cfg.enable_wandb and not cfg.debug
    if enable_visualization:
        # Check if any logger supports log_image
        has_image_logger = any(hasattr(logger, 'log_image') for logger in loggers)
        if has_image_logger:
            callbacks.append(VisualiseCallback(interval=1000, val_interval=1000))
            print("Visualization callback enabled (for debugging training progress)")
        else:
            print("Visualization disabled: no logger supports image logging (requires WandbLogger)")
    else:
        print("Visualization disabled: enable_wandb=False or debug mode")
    # Only add LearningRateMonitor if there's a logger and not in debug mode
    if not cfg.debug and len(loggers) > 0: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            # val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)
    
    if cfg.enable_wandb and not cfg.debug:
        try:
            wandb.finish()
        except:
            pass
    
if __name__ == "__main__":
    main()