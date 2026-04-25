#!/bin/bash
# Drive the LIBERO motion-only CoT labeling pipeline.
#
# Usage:
#   bash scripts/libero_labeling/run_libero_labeling.sh <step> [<suite>] [<num_episodes>]
#
# Steps:
#   instructions  — generate libero_instruction_subtask_mapping.json (40 entries, one VLM call each)
#   label         — run label_libero_episodes.py for the given suite (or "all")
#
# Suite values: libero_goal | libero_object | libero_spatial | libero_10 | all
#
# Prereqs:
#   • vLLM Qwen3-VL-4B server running at $API_URL (default: localhost:8101)
#   • python env with PIL, numpy, pyarrow, scikit-learn, tqdm, imageio[ffmpeg], requests
#
# Examples:
#   # 1) build the per-task subtask plan mapping (one-time, ~40 VLM calls)
#   bash scripts/libero_labeling/run_libero_labeling.sh instructions
#
#   # 2) 5-episode dry-run on libero_goal (skips VLM, validates schema)
#   bash scripts/libero_labeling/run_libero_labeling.sh label libero_goal 5 --dry_run
#
#   # 3) full 4-suite labeling (1693 episodes; ~3-5h with VLM)
#   bash scripts/libero_labeling/run_libero_labeling.sh label all 0
set -euo pipefail

STEP=${1:-label}
SUITE=${2:-all}
NUM_EPISODES=${3:-0}
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

# Paths (override via env)
ENV_PYTHON=${ENV_PYTHON:-/ssd/sxu/workspace/code/starVLA/.env/starvla_upstream_eval/bin/python}
DATA_ROOT=${DATA_ROOT:-/ssd/sxu/workspace/code/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA}
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/sxu/workspace/code/starVLA/playground/Datasets/LIBERO_COT}
API_URL=${API_URL:-http://localhost:8101/v1/chat/completions}
MODEL_NAME=${MODEL_NAME:-/ssd/sxu/workspace/code/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING="$SCRIPT_DIR/libero_instruction_subtask_mapping.json"

echo "=== LIBERO CoT labeling: step=$STEP, suite=$SUITE, num_episodes=$NUM_EPISODES ==="
echo "    Python:    $ENV_PYTHON"
echo "    Data:      $DATA_ROOT"
echo "    Output:    $OUTPUT_ROOT"
echo "    VLM API:   $API_URL"
echo "    Mapping:   $MAPPING"
echo ""

case "$STEP" in
    instructions)
        $ENV_PYTHON "$SCRIPT_DIR/generate_libero_instruction_subtasks.py" \
            --data_root "$DATA_ROOT" \
            --api_url "$API_URL" \
            --model_name "$MODEL_NAME" \
            --output "$MAPPING" \
            $EXTRA_ARGS
        ;;
    label)
        if [ ! -f "$MAPPING" ]; then
            echo "[error] $MAPPING not found. Run 'instructions' step first." >&2
            exit 1
        fi
        $ENV_PYTHON "$SCRIPT_DIR/label_libero_episodes.py" \
            --data_root "$DATA_ROOT" \
            --output_root "$OUTPUT_ROOT" \
            --suite "$SUITE" \
            --num_episodes "$NUM_EPISODES" \
            --instruction_mapping "$MAPPING" \
            --api_url "$API_URL" \
            --model_name "$MODEL_NAME" \
            $EXTRA_ARGS
        ;;
    *)
        echo "Unknown step: $STEP. Use 'instructions' or 'label'." >&2
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
