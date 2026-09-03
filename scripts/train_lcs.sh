#!/bin/bash
# Train simlingo_training (full model with language)

# TODO: modify WORK_DIR for your machine (repo root)
export WORK_DIR=/mnt/localssd/workspace/lcs
cd ${WORK_DIR}


# conda activate simlingo

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH}"
[ -n "${CARLA_ROOT}" ] && export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${PYTHONPATH}"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

TRAIN_ARGS="${@:-experiment=train_lcs}"


GPU_IDS="0,1,2,3,4,5,6,7"
if echo "${TRAIN_ARGS}" | grep -q -E 'gpus=[0-9,]+'; then
    GPU_PARAM=$(echo "${TRAIN_ARGS}" | grep -o -E 'gpus=[0-9,]+' | head -n1 | cut -d'=' -f2)
    if echo "${GPU_PARAM}" | grep -q ','; then
        GPU_IDS="${GPU_PARAM}"
        NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
    else
        GPU_COUNT=${GPU_PARAM}
        if [ -n "${GPU_COUNT}" ] && [ "${GPU_COUNT}" -gt 0 ] 2>/dev/null; then
            GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
            NUM_DEVICES=${GPU_COUNT}
        fi
    fi
    if [ -n "${NUM_DEVICES}" ]; then
        TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
    fi
fi

GPU_LIST_ENV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
if [ -n "${GPU_LIST_ENV}" ]; then
    GPU_IDS="${GPU_LIST_ENV}"
    NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
fi

if [ -z "${GPU_IDS}" ]; then
    GPU_IDS="0"
    NUM_DEVICES=1
    TRAIN_ARGS=$(echo "${TRAIN_ARGS}" | sed -E "s/gpus=[0-9,]+/gpus=${NUM_DEVICES}/")
fi

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
if [ -z "${NUM_DEVICES}" ]; then
    NUM_DEVICES=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
fi

if ! echo "${TRAIN_ARGS}" | grep -q -E '(^|\s)gpus='; then
    TRAIN_ARGS="${TRAIN_ARGS} gpus=${NUM_DEVICES}"
fi

export WANDB_API_KEY=${WANDB_API_KEY:-}
export MASTER_PORT=29500


echo "=========================================="
echo "Starting local training (full)"
echo "=========================================="
echo "Work dir: ${WORK_DIR}"
echo "Train args: ${TRAIN_ARGS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Number of devices: ${NUM_DEVICES}"
echo "=========================================="

cd simlingo_training
python train.py ${TRAIN_ARGS}

