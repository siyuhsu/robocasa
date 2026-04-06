#!/bin/bash
# Multi-GPU parallel evaluation for robocasa pi0.
# Splits all tasks across NUM_GPUS eval workers; each worker talks to its own
# policy server (launched by serve_robocasa_multi.sh beforehand, or inline here).
#
# Usage:
#   bash scripts/eval_robocasa_multi.sh [NUM_GPUS] [BASE_PORT] [NUM_TRIALS] [LOGDIR] [ZERO_BASE_MOTION]
#
# Defaults:  NUM_GPUS=4  BASE_PORT=8000  NUM_TRIALS=50  LOGDIR=auto  ZERO_BASE_MOTION=false

set -euo pipefail

NUM_GPUS=${1:-4}
BASE_PORT=${2:-8000}
NUM_TRIALS=${3:-50}
LOGDIR=${4:-"expdata/pi0_robocasa_pretrain_human300/multitask_learning/75000_multigpu"}
ZERO_BASE_MOTION=${5:-false}

POLICY_CONFIG="pi0_robocasa_pretrain_human300"
POLICY_DIR="/scratch/sx11/sx0401/workspace/code/robocasa/checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Auto-detect robocasa_env from JOBFS (works across nodes with different job IDs)
_JOBFS_ROBOCASA=$(ls -d /jobfs/*/robocasa/robocasa_env 2>/dev/null | head -1)
PYTHON="${_JOBFS_ROBOCASA:+${_JOBFS_ROBOCASA}/bin/python}"
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "[ERROR] Could not find robocasa_env python. Set PYTHON env var or extract robocasa_env.tar to JOBFS."
    exit 1
fi
PYTHONPATH_EXTRA="/scratch/sx11/sx0401/workspace/code/robocasa:/scratch/sx11/sx0401/workspace/code/robocasa/third_party/robosuite:/scratch/sx11/sx0401/workspace/code/robocasa/policy/openpi/src:/scratch/sx11/sx0401/workspace/code/robocasa/policy/openpi/packages/openpi-client/src"

_JOBFS_BASE=$(dirname "$(dirname "${_JOBFS_ROBOCASA}")")
export DATASET_BASE_PATH="${DATASET_BASE_PATH:-${_JOBFS_BASE}/datasets}"
export ROBOCASA_ASSETS_ROOT="${ROBOCASA_ASSETS_ROOT:-${_JOBFS_BASE}/robocasa_assets/assets}"

cd "${OPENPI_DIR}"
mkdir -p "${LOGDIR}"

# ── 1. Collect full env list ───────────────────────────────────────────────────
TASK_SETS="atomic_seen_no_nav composite_seen_no_nav composite_unseen_no_nav"

ALL_ENVS=()
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    ALL_ENVS+=("${line}")
done < <(
    PYTHONPATH="${PYTHONPATH_EXTRA}" \
    "${PYTHON}" - <<'PYEOF'
import sys, os
sys.path.insert(0, '/scratch/sx11/sx0401/workspace/code/robocasa')
sys.path.insert(0, '/scratch/sx11/sx0401/workspace/code/robocasa/third_party/robosuite')
import warnings; warnings.filterwarnings('ignore')
# Suppress print()-based warnings (e.g. mimicgen) during import
_real_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
sys.stdout.close()
sys.stdout = _real_stdout
for ts in ['atomic_seen_no_nav', 'composite_seen_no_nav', 'composite_unseen_no_nav']:
    for e in TASK_SET_REGISTRY[ts]:
        print(e)
PYEOF
)

TOTAL=${#ALL_ENVS[@]}
echo "=== Multi-GPU Robocasa Evaluation ==="
echo "  Total tasks      : ${TOTAL}"
echo "  GPUs             : ${NUM_GPUS}"
echo "  Base port        : ${BASE_PORT}"
echo "  Num trials       : ${NUM_TRIALS}"
echo "  Log dir          : ${LOGDIR}"
echo "  Zero base motion : ${ZERO_BASE_MOTION}"
echo ""

# ── 2. Launch policy servers (one per GPU) ────────────────────────────────────
echo "--- Launching ${NUM_GPUS} policy servers ---"
SERVER_PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    SLOG="${LOGDIR}/server_gpu${i}_port${PORT}.log"
    echo "  GPU ${i}  port ${PORT}"

    CUDA_VISIBLE_DEVICES=${i} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    PYTHONPATH="${PYTHONPATH_EXTRA}" \
        "${PYTHON}" scripts/serve_policy.py \
            --port "${PORT}" \
            policy:checkpoint \
            --policy.config "${POLICY_CONFIG}" \
            --policy.dir "${POLICY_DIR}" \
        > "${SLOG}" 2>&1 &

    SERVER_PIDS[$i]=$!
    sleep 5   # stagger CUDA context creation
done

echo "Waiting 60 s for servers to load checkpoints..."
sleep 60

# Verify servers are alive
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
        echo "[ERROR] Server on GPU ${i} port ${PORT} crashed. Check ${LOGDIR}/server_gpu${i}_port${PORT}.log"
        cat "${LOGDIR}/server_gpu${i}_port${PORT}.log" | tail -20
        exit 1
    fi
    echo "  [OK] GPU ${i} port ${PORT} PID ${SERVER_PIDS[$i]}"
done
echo ""

# ── 3. Split tasks and launch eval workers ────────────────────────────────────
echo "--- Dispatching tasks to ${NUM_GPUS} eval workers ---"

EVAL_PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    # Assign tasks: worker i gets indices i, i+NUM_GPUS, i+2*NUM_GPUS, ...
    WORKER_ENVS=()
    for ((j = i; j < TOTAL; j += NUM_GPUS)); do
        WORKER_ENVS+=("${ALL_ENVS[$j]}")
    done

    PORT=$((BASE_PORT + i))
    ELOG="${LOGDIR}/eval_gpu${i}_port${PORT}.log"
    echo "  Worker ${i} → GPU ${i} port ${PORT}: ${#WORKER_ENVS[@]} tasks"
    printf "    %s\n" "${WORKER_ENVS[@]}"

    # Build --args.env-names arguments (tyro uses dashes)
    ENV_ARGS=()
    for env in "${WORKER_ENVS[@]}"; do
        ENV_ARGS+=("${env}")
    done

    ZERO_FLAG=()
    [[ "${ZERO_BASE_MOTION}" == "true" ]] && ZERO_FLAG=(--args.zero-base-motion)

    CUDA_VISIBLE_DEVICES=${i} \
    MUJOCO_GL=egl \
    PYTHONPATH="${PYTHONPATH_EXTRA}" \
        "${PYTHON}" examples/robocasa/main.py \
            --args.port "${PORT}" \
            --args.host "0.0.0.0" \
            --args.split pretrain \
            --args.num_trials "${NUM_TRIALS}" \
            --args.log_dir "${LOGDIR}" \
            --args.env-names "${ENV_ARGS[@]}" \
            "${ZERO_FLAG[@]}" \
        > "${ELOG}" 2>&1 &

    EVAL_PIDS[$i]=$!
    sleep 5
done

echo ""
echo "--- All eval workers launched. Waiting for completion ---"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    wait "${EVAL_PIDS[$i]}" || echo "[WARN] Worker ${i} exited with non-zero status"
    echo "  Worker ${i} finished"
done

# ── 4. Shutdown servers ───────────────────────────────────────────────────────
echo ""
echo "--- Shutting down policy servers ---"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    kill "${SERVER_PIDS[$i]}" 2>/dev/null && echo "  Killed server PID ${SERVER_PIDS[$i]}"
done

echo ""
echo "=== Evaluation complete. Results in ${LOGDIR} ==="
