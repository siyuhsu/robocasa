# RoboCasa CoT Labeling

Full-scale CoT annotation for RoboCasa (24 atomic tasks, ~8493 episodes). RoboCasa
is **per-task** (`<dataset>/<Task>/...`) and uses the LIBERO state-8 layout
(`observation.state` = eef_pos[0:3] + axis-angle[3:6] + gripper[6:8]) at 256², so
it mixes directly with LIBERO/VLA-Arena.

See `../COT_ANNOTATION.md` for the architecture; this is the run book.

## 0. Data prep (HDF5 → LeRobot)
```bash
# robocasa0.2 env. Cosmos-9 (224, for Cosmos-Policy parity) or LIBERO-8 (256, for mixing).
python convert_robocasa_cosmos_to_lerobot.py --res 256          # or render_robocasa_224.py + build_lerobot_from_npz.py
```
Recommended for annotation/mixing: **`robocasa_v01_libero256_lerobot`** (state-8, 256²).

## 1. Label (Stage-1 map + Stage-2, parallel)
`label_robocasa_full.py` builds the Stage-1 map once per unique instruction, then
labels every episode in parallel, round-robin across VLM endpoints.
```bash
# start N vLLM servers (Qwen3.5-VL, enable_thinking=False) on free GPUs, then:
python label_robocasa_full.py \
  --data-root /…/robocasa_v01_libero256_lerobot \
  --output    /…/robocasa_cot_lib256_full \
  --api_urls  http://localhost:8110/v1/chat/completions,…,8117 \
  --num_workers 24
```
Indices are libero256 `[0,1,2]/[3,4,5]/[6,7]`, `video_key=robot0_agentview_left_image`.
For a 1-ep-per-task smoke test use `run_robocasa_cot_sample.py`.

## 2. Grounding (gripper_2d + bbox, sim state-replay)
`grounding_full.py` (robocasa0.2 env, raw mujoco) writes grounding INTO the cots.
Run **foreground**, one shard per GPU (detached EGL hangs):
```bash
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl python grounding_full.py \
    --cot-root /…/robocasa_cot_lib256_full --num-shards 8 --shard-id $g \
    >/tmp/ground_$g.log 2>&1 </dev/null &
done
```
Maps lerobot ep → human-sorted + mg300-sorted demo, asserts no-op frame alignment,
state-replays `states[t]`, projects eef + segments objects. `robocasa_grounding.py`
is the single-task (ep0) variant used for the sample.

## 3. Coverage QC
```bash
python check_subtask_coverage.py --roots /…/robocasa_cot_lib256_full/*
python reapply_robocasa.py        --cot-root /…/robocasa_cot_lib256_full   # fixes flagged only
```
RoboCasa PnP-with-container tasks (open→pick→place→close) are long-horizon; the
re-map lays the full plan proportionally across segments. Re-check until 0 trunc.

## 4. Review videos
```bash
python unified_cot_viz.py --layout task \
  --cot-root /…/robocasa_cot_lib256_full \
  --data-root /…/robocasa_v01_libero256_lerobot \
  --out /…/qc_viz --units PnPCounterToSink CoffeeServeMug …
```

## Script index
| script | role |
|---|---|
| `convert_robocasa_cosmos_to_lerobot.py`, `render_robocasa_224.py`, `build_lerobot_from_npz.py` | HDF5 → LeRobot (Cosmos-9 / LIBERO-8) |
| `generate_instruction_subtasks.py` | Stage-1 decompose helper |
| `run_robocasa_cot_sample.py` | 1-ep-per-task smoke label+ground+viz |
| `label_robocasa_full.py` | **full-scale Stage-1 map + Stage-2 label** |
| `grounding_full.py` | **full-scale gripper_2d + bbox (state-replay), sharded** |
| `robocasa_grounding.py` | single-task grounding (sample) |
| `check_subtask_coverage.py` | coverage QC (shared) |
| `reapply_robocasa.py` / `reapply_multiobj.py` | re-map truncated cots (linear / multi-object) |
| `unified_cot_viz.py` | review videos (shared, all benchmarks) |
| `robocasa_viz_full.py`, `reviz_with_grounding.py` | earlier robocasa-only viz (superseded by unified) |
