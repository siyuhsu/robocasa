#!/usr/bin/env bash
# Start Qwen3-VL-4B via vLLM on GPU 6 for RoboCasa subtask annotation
#
# Usage:
#   bash scripts/robocasa_labeling/start_vlm.sh
#
# Prerequisites:
#   conda activate siyu_robocasa
#   pip install vllm  (if not already installed)

set -euo pipefail

MODEL_PATH="/data/app/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
PORT=8100
GPU_ID=6

echo "Starting vLLM server for Qwen3-VL-4B on GPU ${GPU_ID}, port ${PORT}"
echo "Model: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port ${PORT} \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 100}' \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
