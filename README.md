# Latent-Centroid Steering (LCS)

**[IROS 2026]** This is the official implementation of the [paper](https://arxiv.org/pdf/2608.00237):

**Latent-Centroid Steering: Single-Pass Classifier-Free Guidance for Command-Aligned Autonomous Driving**

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

## 3. Dataset preparation

LCS training only requires the **driving data** from the [SimLingo dataset](https://huggingface.co/datasets/RenzKa/simlingo) (files prefixed with `data_*`). VQA, commentary, and dreamer labels are not needed.

### Download driving data only

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download only the driving data files (data_*.tar.gz)
huggingface-cli download RenzKa/simlingo \
  --repo-type dataset \
  --include "data_*.tar.gz" \
  --local-dir simlingo_dataset
```

### Extract to the database directory

```bash
mkdir -p database/simlingo

for file in simlingo_dataset/data_*.tar.gz; do
    echo "Extracting $file to database/simlingo/..."
    tar -xzf "$file" -C database/simlingo/
done
```

For more details on the full dataset (language labels, dreamer data, data generation), see the [SimLingo README](https://github.com/RenzKa/simlingo#dataset-download).

## 4. Run training

Edit data paths in `simlingo_training/config/experiment/train_lcs.yaml` and `WORK_DIR` in `scripts/train_lcs.sh`, then:

```bash
conda activate simlingo
bash scripts/train_lcs.sh
# or override GPUs / experiment:
# bash scripts/train_lcs.sh experiment=train_lcs gpus=8
```

## 5. Run inference

Edit paths in `scripts/run_eval_cluster.sh` (`ROUTE_DIR`, `CHECKPOINT`, `CARLA_ROOT`, `PROJECT_ROOT`), then:

```bash
conda activate simlingo
bash scripts/run_eval_cluster.sh \
  /path/to/Bench2Drive/data/bench2drive_split \
  1 \
  /path/to/checkpoints/epoch=015.ckpt \
  0
```

Args: `ROUTE_DIR`, `CHECKPOINT`, `GPU_RANK`.

Agent: `team_code/agent_lcs.py` (config: `team_code/config_lcs.py`).

## Citation

If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.

```bibtex
@article{hu2026latent,
  title={Latent-Centroid Steering: Single-Pass Classifier-Free Guidance for Command-Aligned Autonomous Driving},
  author={Hu, Meibo and Wang, Jiamian and Wang, Pichao and Tao, Zhiqiang},
  journal={arXiv preprint arXiv:2608.00237},
  year={2026}
}
```
