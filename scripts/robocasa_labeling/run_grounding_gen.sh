export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl
NUM_SHARDS=8
SHARD_ID=7

python scripts/robocasa_labeling/extract_annotations.py \
    --datasets_root /tmp/sx0401/workspace/datasets/v1.0/ \
    --output  tmp/robocasa_grounding \
    --num_shards $NUM_SHARDS \
    --shard_id $SHARD_ID \
    --visualize \
    2>&1 | tee -a "tmp/log/grounding_gen_${SHARD_ID}.log"


    # --dataset /home/nvidia/siyu/workspace/robocasa/datasets/v1.0/pretrain/atomic/CloseStandMixerHead/20250820/lerobot  \

# bash scripts/run_singularity.sh
# source scripts/env.sh
# bash scripts/robocasa_labeling/run_grounding_gen.sh
