# Latent-Centroid Steering (LCS)

Latent-Centroid Steering: Single-Pass Classifier-Free Guidance for
Command-Aligned Autonomous Driving

## 1. Install environment

```bash
# Install Anaconda if needed, then:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f environment_simplified.yaml
conda activate simlingo
```

Set paths (edit for your machine):

```bash
export CARLA_ROOT=/path/to/carla0915
export WORK_DIR=/path/to/LCS
export PYTHONPATH="${WORK_DIR}:${CARLA_ROOT}/PythonAPI/carla:${WORK_DIR}/Bench2Drive/scenario_runner:${WORK_DIR}/Bench2Drive/leaderboard:${PYTHONPATH}"
```

## 2. Download checkpoint

```bash
# Example: CFG-0.3 checkpoint used for LCS eval
huggingface-cli download phoenixxx66/lcs_dropout03 \
  --local-dir /path/to/checkpoints/lcs_dropout03
```

Weights used by eval:

```text
.../checkpoints/epoch=015.ckpt/pytorch_model.pt
```

LCS also needs the mean-shift deltas file (included in repo):

```text
data/mean_shift_deltas_cfg03.pt
```

## 3. Run training

Edit data paths in `simlingo_training/config/experiment/train_lcs.yaml` and `WORK_DIR` in `scripts/train_lcs.sh`, then:

```bash
conda activate simlingo
bash scripts/train_lcs.sh
# or override GPUs / experiment:
# bash scripts/train_lcs.sh experiment=train_lcs gpus=8
```

## 4. Run inference

Edit paths in `scripts/run_eval_cluster.sh` (`ROUTE_DIR`, `CHECKPOINT`, `CARLA_ROOT`, `PROJECT_ROOT`), then:

```bash
conda activate simlingo
bash scripts/run_eval_cluster.sh \
  /path/to/Bench2Drive/data/bench2drive_split \
  1 \
  /path/to/checkpoints/epoch=015.ckpt \
  0
```

Args: `ROUTE_DIR`, `SEED`, `CHECKPOINT`, `GPU_RANK`.

Agent: `team_code/agent_lcs.py` (config: `team_code/config_lcs.py`).
