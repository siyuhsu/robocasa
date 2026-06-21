#!/bin/bash
# End-to-end CoT + grounding annotation of ONE LIBERO human demo.
#   Stage-1 + Stage-2 (label) → STATE-REPLAY grounding (the fixed method) → review video.
# Usage:  annotate_demo.sh <SUITE> [EP]      e.g. annotate_demo.sh libero_spatial 0
#   env overrides: DATA, COT, HDF5_DIR, MAP, VLM, GPU
# Needs the source *_no_noops HDF5 (states+model_file) for grounding — provide via HDF5_DIR.
set -e
SUITE=${1:?usage: annotate_demo.sh <SUITE> [EP]}; EP=${2:-0}
DATA=${DATA:-/ssd/sxu/workspace/code/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA}
COT=${COT:-/tmp/annotate_demo/libero}
HDF5_DIR=${HDF5_DIR:-/ssd/sxu/workspace/Datasets/LIBERO-datasets/${SUITE}_no_noops}
SD=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SD:$SD/../robocasa_labeling:$SD/../vla_arena_labeling${PYTHONPATH:+:$PYTHONPATH}
MAP=${MAP:-$SD/libero_instruction_subtask_mapping.json}
VLM=${VLM:-http://localhost:8110/v1/chat/completions}
SV=/ssd/sxu/miniconda3/envs/starVLA/bin/python
LE=/ssd/sxu/miniconda3/envs/libero_eval/bin/python

echo "[1/3] Stage-1 (inline map) + Stage-2 label  $SUITE ep$EP"
$SV $SD/../robocasa_labeling/make_single_map.py --suite "$SUITE" --episodes "$EP" \
    --data_root "$DATA" --out_map "$COT/_map.json" --video_key observation.images.image --api_url "$VLM"
$SV $SD/label_libero_episodes.py --suite "$SUITE" --start_ep "$EP" --num_episodes 1 \
    --data_root "$DATA" --output_root "$COT" --instruction_mapping "$COT/_map.json" --api_url "$VLM"
echo "[2/3] grounding (sim state-replay: gripper_2d + bbox)"
CUDA_VISIBLE_DEVICES=${GPU:-0} MUJOCO_GL=egl $LE $SD/extract_libero_grounding_sr.py \
    --suite "$SUITE" --episodes "$EP" --cot_root "$COT" --data_root "$DATA" --hdf5_dir "$HDF5_DIR"
echo "[3/3] review video"
$SV $SD/../robocasa_labeling/unified_cot_viz.py --layout suite --cot-root "$COT" \
    --data-root "$DATA" --out "$COT/viz" --units "$SUITE" --episodes "$EP"
printf 'DONE -> %s/viz/%s_ep%04d_cot.mp4\n' "$COT" "$SUITE" "$EP"
