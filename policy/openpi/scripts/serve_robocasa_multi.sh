#!/bin/bash
# Launch multiple pi0 policy servers, one per GPU, on consecutive ports.
# Uses XLA_PYTHON_CLIENT_PREALLOCATE=false to avoid pre-allocating all GPU memory
# (important when sharing the node with other processes like training jobs).

BASE_PORT=${1:-8000}
NUM_GPUS=${2:-4}

POLICY_CONFIG="pi0_robocasa_pretrain_human300"
POLICY_DIR="/scratch/sx11/sx0401/workspace/code/robocasa/checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
_JOBFS_ROBOCASA=$(ls -d /jobfs/*/robocasa/robocasa_env 2>/dev/null | head -1)
PYTHON="${_JOBFS_ROBOCASA:+${_JOBFS_ROBOCASA}/bin/python}"
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "[ERROR] Could not find robocasa_env python under /jobfs/*/robocasa/robocasa_env"
    exit 1
fi
_JOBFS_BASE=$(dirname "$(dirname "${_JOBFS_ROBOCASA}")")
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-${_JOBFS_BASE}/datasets}"
export ROBOCASA_ASSETS_ROOT="${ROBOCASA_ASSETS_ROOT:-${_JOBFS_BASE}/robocasa_assets/assets}"
export MUJOCO_GL=egl

cd "${OPENPI_DIR}"

echo "=== Launching ${NUM_GPUS} pi0 policy servers (ports ${BASE_PORT}–$((BASE_PORT + NUM_GPUS - 1))) ==="

SERVER_PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="serve_gpu${i}_port${PORT}.log"
    echo "  GPU ${i}  port ${PORT}  log ${LOG}"

    CUDA_VISIBLE_DEVICES=${i} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "${PYTHON}" scripts/serve_policy.py \
            --port "${PORT}" \
            policy:checkpoint \
            --policy.config "${POLICY_CONFIG}" \
            --policy.dir "${POLICY_DIR}" \
        > "${LOG}" 2>&1 &

    SERVER_PIDS[$i]=$!
    # Brief pause so JAX initialises before the next server starts competing for CUDA context
    sleep 5
done

echo ""
echo "Server PIDs: ${SERVER_PIDS[*]}"
echo "Waiting for servers to finish loading (≈30 s)..."
sleep 30

# Quick liveness check
ALL_OK=1
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    if kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
        echo "  [OK] GPU ${i} port ${PORT} PID ${SERVER_PIDS[$i]}"
    else
        echo "  [FAIL] GPU ${i} port ${PORT} – process not running; check serve_gpu${i}_port${PORT}.log"
        ALL_OK=0
    fi
done

if [[ ${ALL_OK} -eq 0 ]]; then
    echo "One or more servers failed to start."
    exit 1
fi

echo ""
echo "All servers running. To stop them:"
echo "  kill ${SERVER_PIDS[*]}"
