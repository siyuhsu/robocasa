#!/bin/bash
# End-to-end CoT + grounding annotation of ONE VLA-Arena human demo.
#   Stage-1 + Stage-2 (label) → STATE-REPLAY grounding → review video.
# Usage:  annotate_demo.sh <SUITE> [EP]      e.g. annotate_demo.sh vla_arena_l0 0
#   env overrides: DATA, COT, HDF5_DIR, VLM, GPU
# Grounding needs the source HDF5 (states+model_file) of the VLA-Arena demos — the
# RLDS builder's source. Provide it via HDF5_DIR. Without it, grounding is skipped
# (label + review video still produced). Set HDF5_DIR= to skip grounding explicitly.
set -e
SUITE=${1:?usage: annotate_demo.sh <SUITE> [EP]}; EP=${2:-0}
DATA=${DATA:-/ssd/sxu/workspace/sizhe/datasets/vla_arena_libero_cot}
COT=${COT:-/tmp/annotate_demo/vla_arena}
HDF5_DIR=${HDF5_DIR-/ssd/sxu/workspace/sizhe/VLA-Arena/datasets/${SUITE}}   # set HDF5_DIR= to skip grounding
SD=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SD/../libero_labeling:$SD/../robocasa_labeling:$SD${PYTHONPATH:+:$PYTHONPATH}
VLM=${VLM:-http://localhost:8110/v1/chat/completions}
SV=/ssd/sxu/miniconda3/envs/starVLA/bin/python
LE=/ssd/sxu/miniconda3/envs/libero_eval/bin/python

echo "[1/3] Stage-1 (inline map) + Stage-2 label  $SUITE ep$EP"
$SV $SD/../robocasa_labeling/make_single_map.py --suite "$SUITE" --episodes "$EP" \
    --data_root "$DATA" --out_map "$COT/_map.json" --video_key observation.images.image --api_url "$VLM"
$SV $SD/label_vla_arena_episodes.py --suite "$SUITE" --episodes "$EP" \
    --data_root "$DATA" --output_root "$COT" --instruction_mapping "$COT/_map.json" --api_url "$VLM"
if [ -n "$HDF5_DIR" ] && ls "$HDF5_DIR"/*.hdf5 >/dev/null 2>&1; then
  echo "[2/3] grounding (sim state-replay: gripper_2d + bbox)"
  CUDA_VISIBLE_DEVICES=${GPU:-0} MUJOCO_GL=egl $LE $SD/../libero_labeling/extract_libero_grounding_sr.py \
      --suite "$SUITE" --episodes "$EP" --cot_root "$COT" --data_root "$DATA" --hdf5_dir "$HDF5_DIR"
else
  echo "[2/3] grounding SKIPPED (no source HDF5 at HDF5_DIR=$HDF5_DIR) — label-only"
fi
echo "[3/3] review video"
$SV $SD/../robocasa_labeling/unified_cot_viz.py --layout suite --cot-root "$COT" \
    --data-root "$DATA" --out "$COT/viz" --units "$SUITE" --episodes "$EP"
printf 'DONE -> %s/viz/%s_ep%04d_cot.mp4\n' "$COT" "$SUITE" "$EP"
