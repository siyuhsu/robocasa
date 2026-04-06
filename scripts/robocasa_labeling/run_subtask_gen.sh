# bash scripts/run_singularity.sh
# source scripts/env.sh
# bash scripts/robocasa_labeling/run_subtask_gen.sh

export PBS_JOBID="163175029.gadi-pbs"

python scripts/robocasa_labeling/generate_subtasks.py \
    --datasets_root /tmp/sx0401/workspace/datasets/v1.0/ \
    --output  tmp/robocasa_subtasks \
    --api_url "http://localhost:8001/v1/chat/completions" \
    --model_name "/jobfs/${PBS_JOBID}/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" \
    --num_shards 6 \
    --shard_id 0 \
    --visualize \
    2>&1 | tee -a "tmp/subtask_gen_0.log"