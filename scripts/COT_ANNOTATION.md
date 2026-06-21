# Chain-of-Thought (CoT) Annotation Pipeline

This directory holds the CoT-annotation pipeline used to label manipulation
benchmarks (LIBERO, VLA-Arena, RoboCasa) with **hierarchical subtask plans +
per-segment reasoning + visual grounding**, for training/evaluating hierarchical
VLA models (e.g. qwenlap-ctxdemo).

The pipeline is intentionally **benchmark-agnostic**: the heavy lifting
(segmentation, VLM labeling, order correction, coverage QC) lives in shared
components; each benchmark only supplies a thin adapter (data layout, state
indices, video key). Adding a new benchmark is a small, well-defined job — see
[Extending to a new benchmark](#extending-to-a-new-benchmark).

```
scripts/
├── COT_ANNOTATION.md              ← this file (architecture + extension guide)
├── libero_labeling/      README.md + LIBERO scripts
├── vla_arena_labeling/   README.md + VLA-Arena scripts
└── robocasa_labeling/    README.md + RoboCasa scripts (+ shared QC/viz)
```

---

## 1. Output schema

One `cot_annotations.json` per episode under `<COT_ROOT>/<unit>/extras/episode_NNNNNN/`,
where `<unit>` is a *suite* (LIBERO/VLA-Arena) or a *task* (RoboCasa). Top-level:

| field | meaning |
|---|---|
| `instruction` | the task language string |
| `subtask_list` | Stage-1 plan: the instruction decomposed into an ordered subtask list (the "ground-truth" decomposition) |
| `segment_labels_raw` | Stage-2 raw per-segment `{seg_id: [subtask, reasoning]}` (before order correction) |
| `frames` | per-frame records (below) |
| `all_object_names`, `obj_cat`, `distr_cats` | grounding object metadata |

Per-frame (`frames[i]`):

| field | meaning |
|---|---|
| `frame_index`, `segment_id` | frame ↔ segment mapping |
| `subtask` | **merged** subtask (order-corrected, dedup) — the LONG PLAN step active this frame |
| `reason` | **per-segment** reasoning (one per triple-segment, NOT merged) |
| `assistant_plan_level` | rendered LONG PLAN (numbered subtask sequence) |
| `assistant_short_plan` | rendered SHORT PLAN (current subtask + reasoning) |
| `gripper_2d` | `[u,v]` end-effector pixel (grounding) |
| `task_obj_bbox`, `distractor_bboxes` | object boxes (grounding) |

**Two-tier design (important):** `subtask` is *merged* (consecutive duplicates
collapsed → a clean LONG PLAN), while `reason` is kept *per triple-segment*
(finer granularity). Do not collapse reasoning into the merged subtask.

---

## 2. Pipeline stages

### Stage 1 — instruction → subtask plan  (VLM, once per unique instruction)
`generate_*_instruction_subtasks.py` → `*_instruction_subtask_mapping.json`.
Decompose each **unique** instruction once (not per-episode) so every episode of
the same task shares an identical, consistent plan. Uses `decompose_instruction_vlm`
with a few keyframes. **Always verify 0 fallback** (a fallback `["approach",
"interact","complete"]` means the VLM/parse failed → bad labels downstream).

### Stage 2 — segmentation + per-segment labeling + order correction
`label_*_episodes.py` (core: `label_one_episode`):
1. **triple_segment** — HDBSCAN over per-step spatial(Δeef-pos) / orientation
   (Δorient) / gripper(Δgrip) signals → contiguous motion segments.
2. **per-segment VLM** — for each segment, pick the best-matching subtask from
   `subtask_list` + write a one-line reasoning, given segment keyframes.
3. **LNDS order correction** (`correct_subtask_order`) — a weighted
   longest-non-decreasing-subsequence keeps the plan-monotone backbone and
   position-fills the rest, fixing VLM mislabels while respecting execution order.
   `subtask` = corrected+merged; `reason` = raw per-segment.

### Stage 3 — visual grounding  (per-frame gripper_2d + object bbox)
Replay the recorded **sim states** and, per frame: project the world eef through
the camera → `gripper_2d`; render instance/geom **segmentation** → object boxes.

> ⚠️ **State-replay, never action-replay.** Rebuilding the trajectory by stepping
> recorded *actions* open-loop diverges (grasps fail) → the object never moves in
> the replay → **bbox sticks at the initial position** (gripper_2d still ~tracks).
> Always set the sim to the recorded `states[t]` (`set_state_from_flattened` /
> `data.qpos[:]=states[t]; mj_forward`). RoboCasa (`grounding_full.py`) does this
> correctly; the original LIBERO/VLA-Arena grounding used action-replay and has a
> known static-bbox bug (~20–50 % of manipulated episodes).

---

## 3. Quality mechanisms

| concern | mechanism | script |
|---|---|---|
| plan consistency across episodes | Stage-1 map keyed by **unique instruction** | `*_instruction_subtasks` |
| VLM mislabels / order errors | weighted-LNDS order correction | `subtask_order` / `correct_subtask_order` |
| long-horizon 2nd-object collapse | coverage check → re-map full plan | `check_subtask_coverage.py` + `reapply_*` |
| grounding tracking | sim **state**-replay + alignment guard (HDF5 frames == lerobot frames) | `grounding_full.py` |
| visual review | unified per-task review videos | `unified_cot_viz.py` |

**Coverage QC** (`check_subtask_coverage.py`): flags episodes whose merged LONG
PLAN doesn't reach the end of its Stage-1 plan (instruction not fully realized):
- `trunc_multi` — multi-object plan, 2nd object phase dropped → `reapply_multiobj.py`
  (gripper-cycle rounds × plan phases, proportional within round).
- `trunc_single` — single linear task didn't reach final subtask → `reapply_robocasa.py`
  (lay the full plan proportionally across segments; robocasa exec is linear).
Re-run the check after `reapply_*` until counts are 0.

---

## 4. Extending to a new benchmark

The shared core (`label_one_episode`, `triple_segment`, `correct_subtask_order`,
`decompose_instruction_vlm`, `check_subtask_coverage`, `reapply_*`,
`unified_cot_viz`) is reused as-is. You write a **thin adapter**:

1. **Get data into LeRobot** — per-task or per-suite dirs with
   `data/chunk-*/episode_*.parquet` (`observation.state`, `action`) + `videos/.../*.mp4`.
2. **State indices** — set `spatial/orient/gripper` indices into `observation.state`
   for `triple_segment` (e.g. LIBERO-8: `[0,1,2]/[3,4,5]/[6,7]`; Cosmos-9:
   `[2,3,4]/[5,6,7,8]/[0,1]`).
3. **`video_key`** — the lerobot image key (e.g. `observation.images.image`,
   `robot0_agentview_left_image`).
4. **Stage-1 driver** — build the instruction→subtask map once per unique lang.
5. **Stage-2 driver** — copy `label_robocasa_full.py`: iterate `(unit, ep) →
   instruction → map[instruction]`, call `label_one_episode(..., spatial_indices,
   orient_indices, gripper_indices, video_key, model)`, parallel + round-robin VLM.
6. **Grounding (optional)** — need recorded **sim states + model/env**. Copy
   `grounding_full.py`: map lerobot ep → source demo, **state-replay**, project
   eef + segment objects, **assert kept-frame-count == lerobot-frame-count**.
   No states available → fall back to a per-frame 2D detector (lower quality).
7. **QC** — `check_subtask_coverage.py --roots <COT>/*` → `reapply_*` on flagged
   → re-check until 0.
8. **Viz** — `unified_cot_viz.py --layout {suite|task} --units ...`.

### Quality checklist (run before declaring a benchmark done)
- [ ] Stage-1 map: **0 fallback** plans.
- [ ] Stage-2: **0 errors**; per-frame `reason` count == #triple-segments.
- [ ] Coverage: `check_subtask_coverage` → **0 trunc** (after `reapply_*`).
- [ ] Grounding (if done): **bbox tracks moving objects** (state-replay, not
      action-replay); per-ep frame-count alignment guard passes.
- [ ] Spot-check N per-task review videos (`unified_cot_viz.py`).

---

## 5. Known issues / caveats

- **Static-bbox bug** in the original LIBERO/VLA-Arena grounding (action-replay
  divergence). RoboCasa is unaffected (state-replay). Fix = re-ground via
  state-replay (verified) but needs the raw HDF5 `states` for the affected suites
  (not all present locally). See `git log` / project notes.
- **No-op alignment** — lerobot is usually `*_no_noops` (filtered). Grounding must
  align source states 1:1 to lerobot frames (use the `_no_noops` source HDF5, or
  match by frame-count + action[:6]; the gripper action dim is often sign-flipped
  between lerobot and source HDF5).
- **EGL rendering** hangs in detached (`setsid`) processes — run grounding
  **foreground** (per-GPU sharding) and free `mujoco.Renderer` per episode.

## 6. End-to-end single-demo annotation

Each benchmark dir has `annotate_demo.sh <unit> [ep]` for **one-shot** annotation
of a single human demo (new task, small dataset): inline Stage-1 → Stage-2 →
**state-replay grounding** → review video. Shared inline Stage-1 = `make_single_map.py`;
LIBERO/VLA-Arena state-replay grounding = `libero_labeling/extract_libero_grounding_sr.py`
(the fixed method — sets `states[t]`, so the moving-object bbox tracks). Grounding
needs the source HDF5 (`states`+`model_file`); pass it via `HDF5_DIR`.
