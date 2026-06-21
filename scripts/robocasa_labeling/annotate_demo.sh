#!/bin/bash
# End-to-end CoT + grounding annotation of ONE RoboCasa human demo.
#   Stage-1 (instruction→subtask) + Stage-2 (segment+VLM+LNDS) → state-replay
#   grounding (gripper_2d + bbox) → unified review video.
# Usage:  annotate_demo.sh <TASK> [EP]
#   env overrides: DATA, COT, VLM, GPU
set -e
TASK=${1:?usage: annotate_demo.sh <TASK> [EP]}; EP=${2:-0}
DATA=${DATA:-/ssd/sxu/workspace/sizhe/datasets/robocasa_v01_libero256_lerobot}
COT=${COT:-/tmp/annotate_demo/robocasa}
VLM=${VLM:-http://localhost:8110/v1/chat/completions}
SD=$(cd "$(dirname "$0")" && pwd)
SV=/ssd/sxu/miniconda3/envs/starVLA/bin/python
RC=/ssd/sxu/miniconda3/envs/robocasa0.2/bin/python
export PYTHONPATH=$SD/..:$SD/../libero_labeling:$SD/../vla_arena_labeling:$SD

echo "[1/3] label (Stage-1 + Stage-2)  $TASK ep$EP"
$SV $SD/run_robocasa_cot_sample.py --tasks "$TASK" --ep "$EP" --data-root "$DATA" --output "$COT" --api_url "$VLM"
echo "[2/3] grounding (sim state-replay: gripper_2d + bbox)"
CUDA_VISIBLE_DEVICES=${GPU:-0} MUJOCO_GL=egl $RC $SD/grounding_full.py --cot-root "$COT" --tasks "$TASK" --episodes "$EP"
echo "[3/3] review video"
$SV $SD/unified_cot_viz.py --layout task --cot-root "$COT" --data-root "$DATA" --out "$COT/viz" --units "$TASK" --episodes "$EP"
printf 'DONE -> %s/viz/%s_ep%04d_cot.mp4\n' "$COT" "$TASK" "$EP"
