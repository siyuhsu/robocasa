#!/bin/bash
# Qwen3.5-9B vLLM for CoT labeling. Usage: G=<gpu> bash vlm_launch.sh
export PATH=/ssd/sxu/miniconda3/envs/vllm/bin:$PATH   # ninja etc. for engine init
CUDA_VISIBLE_DEVICES=${G:-7} python -m vllm.entrypoints.openai.api_server \
  --model /ssd/sxu/workspace/code/starVLA/playground/Pretrained_models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --port 8110 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 100}' --trust-remote-code --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --enforce-eager
