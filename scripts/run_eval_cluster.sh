#!/bin/bash
# Local Bench2Drive eval (one-by-one) over full bench2drive_split

ROUTE_DIR="${1:-/mnt/localssd/workspace/lcs/Bench2Drive/data/bench2drive_split}"
SEED="${2:-1}"
CHECKPOINT="${3:-/mnt/localssd/workspace/lcs/checkpoints/lcs_dropout03/epoch=015.ckpt/pytorch_model.pt}" ##TODO: modify this path for your machine
GPU_RANK="${4:-0}"
MAX_RETRIES="${5:-1}"
CARLA_GPU_RANK="${6:-${GPU_RANK}}"
echo "GPU: model -> GPU ${GPU_RANK}, CARLA -> GPU ${CARLA_GPU_RANK}"

# conda env
CONDA_ENV="${CONDA_ENV:-simlingo}"
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "WARN: conda.sh not found under \$HOME, using current python ($(which python))"
fi
if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}" && echo "[ENV] conda activate ${CONDA_ENV} ($(which python))" \
        || echo "WARN: failed to activate conda env ${CONDA_ENV}, using current python"
fi

# CARLA / Bench2Drive paths
# TODO: modify these paths for your machine
export CARLA_ROOT="${CARLA_ROOT:-/mnt/localssd/workspace/carla0915}"
export WORK_DIR=/mnt/localssd/workspace/lcs/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PROJECT_ROOT=/mnt/localssd/workspace/lcs
INFERENCE_ROOT=/mnt/localssd/workspace/lcs

BASE_PORT=31000
BASE_TM_PORT=41000
PORT_INCREMENT=5

TRAFFIC_MANAGER_SEED=${SEED}
TEAM_AGENT=${INFERENCE_ROOT}/team_code/agent_lcs.py
if [ -d "${CHECKPOINT}" ]; then
    TEAM_CONFIG="${CHECKPOINT}/pytorch_model.pt"
else
    TEAM_CONFIG=${CHECKPOINT}
fi
RESUME=True

OUT_ROOT=${INFERENCE_ROOT}/eval_results/run_eval_cluster
AGENT_NAME="agent"
BENCHMARK="bench2drive"

export PYTHONPATH=${WORK_DIR}/scenario_runner:${PYTHONPATH}
export PYTHONPATH=${WORK_DIR}/leaderboard:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}
export PYTHONPATH=${INFERENCE_ROOT}:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}

if [ ! -f "${TEAM_CONFIG}" ]; then
    echo "ERROR: Model file not found: ${TEAM_CONFIG}"
    exit 1
fi

if [ -d "${PROJECT_ROOT}/leaderboard" ] && [ ! -L "${PROJECT_ROOT}/leaderboard" ]; then
    mv "${PROJECT_ROOT}/leaderboard" "${PROJECT_ROOT}/leaderboard_backup"
    LEADERBOARD_BACKUP=true
else
    LEADERBOARD_BACKUP=false
fi

cleanup_carla() {
    local CARLA_PORT="${1:-}"
    [ -z "${CARLA_PORT}" ] && return 0
    
    local PIDS=$(ps aux | grep -E "[C]arlaUE4.*-carla-rpc-port=${CARLA_PORT}" | awk '{print $2}')
    if [ -n "${PIDS}" ]; then
        for PID in ${PIDS}; do
            kill -TERM "${PID}" 2>/dev/null || true
        done
        sleep 2
        for PID in ${PIDS}; do
            kill -0 "${PID}" 2>/dev/null && kill -9 "${PID}" 2>/dev/null || true
        done
    fi
    
    if command -v lsof >/dev/null 2>&1; then
        local LSOF_PIDS=$(lsof -ti:${CARLA_PORT} 2>/dev/null || true)
        if [ -n "${LSOF_PIDS}" ]; then
            for PID in ${LSOF_PIDS}; do
                ps -p "${PID}" -o comm= 2>/dev/null | grep -q -i carla && kill -TERM "${PID}" 2>/dev/null || true
            done
            sleep 1
            for PID in ${LSOF_PIDS}; do
                if kill -0 "${PID}" 2>/dev/null; then
                    ps -p "${PID}" -o comm= 2>/dev/null | grep -q -i carla && kill -9 "${PID}" 2>/dev/null || true
                fi
            done
        fi
    fi
    sleep 1
}

cd ${PROJECT_ROOT}
export PYTHONUNBUFFERED=1

BASE_DIR="${OUT_ROOT}/${AGENT_NAME}/${BENCHMARK}/1"
mkdir -p "${BASE_DIR}/res" "${BASE_DIR}/viz" "${BASE_DIR}/logs"

SCRIPT_LOG="${BASE_DIR}/script_run.log"
echo "===========================================" | tee "${SCRIPT_LOG}"
echo "Script started at $(date)" | tee -a "${SCRIPT_LOG}"
echo "===========================================" | tee -a "${SCRIPT_LOG}"


FAILED_ROUTES_FILE="${BASE_DIR}/failed_routes.txt"
RETRY_ROUTES_FILE="${BASE_DIR}/retry_routes.txt"

if [ -f "${RETRY_ROUTES_FILE}" ] && [ -s "${RETRY_ROUTES_FILE}" ]; then
    ROUTE_FILES=()
    while IFS= read -r line || [ -n "$line" ]; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        ROUTE_ID=$(echo "$line" | awk '{print $1}')
        FOUND_FILE=$(ls "${ROUTE_DIR}"/*${ROUTE_ID}*.xml 2>/dev/null | head -1)
        [ -n "$FOUND_FILE" ] && [ -f "$FOUND_FILE" ] && ROUTE_FILES+=("$FOUND_FILE")
    done < "${RETRY_ROUTES_FILE}"
else
    ROUTE_FILES=($(ls "${ROUTE_DIR}"/*.xml 2>/dev/null | sort))
fi

if [ ${#ROUTE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No route files found in ${ROUTE_DIR}"
    [ "$LEADERBOARD_BACKUP" = true ] && mv "${PROJECT_ROOT}/leaderboard_backup" "${PROJECT_ROOT}/leaderboard"
    exit 1
fi

echo "=========================================="
echo "Processing ${#ROUTE_FILES[@]} routes from ${ROUTE_DIR}"
echo "Using GPU: ${GPU_RANK}"
echo "Port range: ${BASE_PORT}-$((BASE_PORT + ${#ROUTE_FILES[@]} * PORT_INCREMENT))"
echo "TM Port range: ${BASE_TM_PORT}-$((BASE_TM_PORT + ${#ROUTE_FILES[@]} * PORT_INCREMENT))"
echo "=========================================="

TOTAL_ROUTES=${#ROUTE_FILES[@]}
CURRENT_ROUTE=0
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

> "${FAILED_ROUTES_FILE}"

for ROUTE_FILE in "${ROUTE_FILES[@]}"; do
    CURRENT_ROUTE=$((CURRENT_ROUTE + 1))
    ROUTE_BASENAME=$(basename "${ROUTE_FILE}" .xml)
    ROUTE_ID=$(echo "${ROUTE_BASENAME}" | sed 's/.*_//' || echo "${ROUTE_BASENAME}")
    [ ${#ROUTE_ID} -lt 3 ] && ROUTE_ID=$(printf "%03d" "${ROUTE_ID}" 2>/dev/null || echo "${ROUTE_ID}")
    
    PORT_OFFSET=$((PORT_INCREMENT * (CURRENT_ROUTE - 1)))
    PORT=$((BASE_PORT + PORT_OFFSET))
    TM_PORT=$((BASE_TM_PORT + PORT_OFFSET))
    
    VIZ_PATH="${BASE_DIR}/viz/${ROUTE_ID}"
    RESULT_FILE="${BASE_DIR}/res/${ROUTE_ID}_res.json"
    LOG_FILE="${BASE_DIR}/logs/${ROUTE_ID}_out.log"
    ERR_FILE="${BASE_DIR}/logs/${ROUTE_ID}_err.log"
    
    mkdir -p "${VIZ_PATH}" "$(dirname ${RESULT_FILE})" "$(dirname ${LOG_FILE})"
    export SAVE_PATH="${VIZ_PATH}"
    
    if [ -f "${RESULT_FILE}" ]; then
        CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    # Skip if route has been processed
    if len(progress) >= 2:
        records = checkpoint.get('records', [])
        if records:
            status = records[0].get('status', '')
            # Skip both Completed and Failed routes
            if 'Completed' in status or 'Failed' in status:
                sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null)
        if [ $? -eq 0 ]; then
            SKIP_COUNT=$((SKIP_COUNT + 1))
            echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: SKIPPED (already processed)" | tee -a "${SCRIPT_LOG}"
            continue
        fi
    fi
    
    ROUTE_RETRY_COUNT=0
    ROUTE_SUCCESS=false
    
    while [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ] && [ "$ROUTE_SUCCESS" = false ]; do
        [ $ROUTE_RETRY_COUNT -gt 0 ] && cleanup_carla "${PORT}" && rm -rf "${VIZ_PATH}" && rm -f "${RESULT_FILE}"
        cleanup_carla "${PORT}"
        mkdir -p "${VIZ_PATH}"
        
        START_TIME=$(date +%s)
        CUDA_VISIBLE_DEVICES=${GPU_RANK} python -u ${WORK_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py \
            --routes="${ROUTE_FILE}" \
            --repetitions=1 \
            --track=SENSORS \
            --checkpoint="${RESULT_FILE}" \
            --agent="${TEAM_AGENT}" \
            --agent-config="${TEAM_CONFIG}" \
            --debug=0 \
            --resume=${RESUME} \
            --port="${PORT}" \
            --traffic-manager-port="${TM_PORT}" \
            --traffic-manager-seed="${TRAFFIC_MANAGER_SEED}" \
            --gpu-rank=${GPU_RANK} \
            --carla-gpu-rank=${CARLA_GPU_RANK} \
            --timeout=600 \
            > "${LOG_FILE}" 2> "${ERR_FILE}"
        
        EVAL_EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        cleanup_carla "${PORT}"
        
        if [ $EVAL_EXIT_CODE -eq 0 ] && [ -f "${RESULT_FILE}" ]; then
            CHECK_RESULT=$(python3 -c "
import json
import sys
try:
    with open('${RESULT_FILE}', 'r') as f:
        data = json.load(f)
    checkpoint = data.get('_checkpoint', {})
    progress = checkpoint.get('progress', [])
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    records = checkpoint.get('records', [])
    for r in cp.get('records',[]):
        if r.get('status','') in ('Failed','Failed - Agent couldn\'t be set up','Failed - Simulation crashed','Failed - Agent crashed'): sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null)
            
            if [ $? -eq 0 ]; then
                echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: OK (${DURATION}s)" | tee -a "${SCRIPT_LOG}"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                ROUTE_SUCCESS=true
            else
                ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
            fi
        else
            echo "[${CURRENT_ROUTE}/${TOTAL_ROUTES}] Route ${ROUTE_ID}: FAILED (exit ${EVAL_EXIT_CODE})" | tee -a "${SCRIPT_LOG}"
            ROUTE_RETRY_COUNT=$((ROUTE_RETRY_COUNT + 1))
        fi
        
        [ "$ROUTE_SUCCESS" = false ] && [ $ROUTE_RETRY_COUNT -le $MAX_RETRIES ] && sleep 5
    done
    
    if [ "$ROUTE_SUCCESS" = false ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "${ROUTE_ID} ${ROUTE_FILE}" >> "${FAILED_ROUTES_FILE}"
    fi
    
    sleep 20
done
echo "Summary: ${SUCCESS_COUNT} success, ${SKIP_COUNT} skipped, ${FAIL_COUNT} failed" | tee -a "${SCRIPT_LOG}"
[ ${FAIL_COUNT} -gt 0 ] && [ -f "${FAILED_ROUTES_FILE}" ] && [ -s "${FAILED_ROUTES_FILE}" ] && \
    cp "${FAILED_ROUTES_FILE}" "${RETRY_ROUTES_FILE}"
[ "$LEADERBOARD_BACKUP" = true ] && mv "${PROJECT_ROOT}/leaderboard_backup" "${PROJECT_ROOT}/leaderboard"
echo "===========================================" | tee -a "${SCRIPT_LOG}"
echo "Script finished at $(date)" | tee -a "${SCRIPT_LOG}"
echo "===========================================" | tee -a "${SCRIPT_LOG}"
[ ${FAIL_COUNT} -gt 0 ] && exit 1 || exit 0
