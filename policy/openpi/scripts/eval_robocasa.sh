export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

LOGDIR="expdata/pi0_robocasa_pretrain_human300/multitask_learning/75000_2"
LOGFILE="$LOGDIR/pi0_robocasa_pretrain_human300.log"

python examples/robocasa/main.py \
    --args.port 8000 \
    --args.task_set atomic_seen_no_nav composite_seen_no_nav composite_unseen_no_nav \
    --args.split pretrain \
    --args.num_trials 50 \
    --args.log_dir "$LOGDIR" \
    2>&1 | tee -a "$LOGFILE"
