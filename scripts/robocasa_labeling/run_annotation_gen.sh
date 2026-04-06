export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl
NUM_SHARDS=8
SHARD_ID=7

python scripts/robocasa_labeling/create_dataset.py \
  --dataset /tmp/sx0401/workspace/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --subtasks tmp/robocasa_subtasks/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/subtasks.json \
  --bbox_dataset tmp/robocasa_grounding/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --output_dataset tmp/robocasa_movement/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --visualize
  2>&1 | tee -a "tmp/log/grounding_gen_${SHARD_ID}.log"


    # --dataset /home/nvidia/siyu/workspace/robocasa/datasets/v1.0/pretrain/atomic/CloseStandMixerHead/20250820/lerobot  \

# bash scripts/run_singularity.sh
# source scripts/env.sh
# bash scripts/robocasa_labeling/run_grounding_gen.sh
