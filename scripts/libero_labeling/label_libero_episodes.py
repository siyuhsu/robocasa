"""
End-to-end LIBERO motion-only CoT labeling: per episode, run

  1. parquet read         → states (8-dim) + actions (7-dim) + task_index
  2. triple_segment       → segment id per frame (HDBSCAN spatial+orient+gripper)
  3. video frame sample   → key frames per segment (primary camera only)
  4. VLM call (per seg)   → [subtask, reason] per segment, propagated per frame
  5. cot_annotations.json → coercive JSON in robocasa schema, bbox/gripper-2D = null

Output:
  <out_root>/<suite>/extras/episode_XXXXXX/cot_annotations.json

Each cot_annotations.json contains:
  {sample_key, instruction, num_frames, movement_statistics, frames: [
    {frame_index, segment_id, subtask, reason,
     assistant_plan_level (LONG PLAN), assistant_short_plan, assistant_movement_level,
     assistant_position_level (null), assistant_object_level (null),
     gripper_2d (null), task_obj_bbox (null), distractor_bboxes (null),
     delta_full_state, delta_full_state_norm, movement_text}, ...]}

Usage:
  python scripts/libero_labeling/label_libero_episodes.py \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --suite libero_goal --num_episodes 5 \
      --output_root playground/Datasets/LIBERO_COT \
      --api_url http://localhost:8101/v1/chat/completions \
      --instruction_mapping scripts/libero_labeling/libero_instruction_subtask_mapping.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import (
    HFQwenVLClient,
    LIBERO_SUITES,
    QwenVLLM,
    describe_move,
    discover_libero_suites,
    get_key_frames_per_segment,
    get_segmentation_indices,
    load_dataset_meta,
    load_episode_lengths,
    load_task_index_to_instruction,
    read_episode_states_actions,
    read_video_frames,
    suite_dir as libero_suite_dir,
    triple_segment,
    video_path,
)

# Reuse robocasa generate_subtasks single-segment VLM call
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robocasa_labeling"))
from generate_subtasks import (  # noqa: E402
    build_frame_annotations,
    segment_to_subtask_single,
)


# ───── prompt templates (align with robocasa_labeling/create_dataset.py) ────

INPUT_TEMPLATE = (
    "What action should the robot take to achieve the instruction\n"
    "INSTRUCTION:\n{instruction}\n"
)

LONG_PLAN_TEMPLATE = "LONG PLAN:\n{long_plan}\n"

SHORT_PLAN_TEMPLATE = (
    "SHORT PLAN:\n"
    "Subtask: {subtask}\n"
    "Subtask Reasoning: {reasoning}\n"
)

MOVEMENT_LEVEL_TEMPLATE = (
    "MOVEMENT:\n"
    "Current Movement: {current_movement}\n"
    "Subtask Movement: {subtask_movement}\n"
)


# ───── helpers ──────────────────────────────────────────────────────────────


"""
LIBERO action units: robosuite OSC_POSE controller maps normalized commands
[-1, 1] to physical deltas using:
    output_max_pos = 0.05 m/step    (action[0:3] xyz)
    output_max_rot = 0.5 rad/step   (action[3:6] roll/pitch/yaw)
Empirically verified (action stats over 4 suites): xyz Q99 ≈ 0.94 → ~47mm/step,
rpy Q99 ≈ 0.18-0.35 → ~6-10°/step. Without scaling, summing 100+ frames
produces fictitious "14000 mm" values; with scaling, subtask-level deltas
become realistic (~50-200mm = 5-20cm tabletop range).
"""
LIBERO_POS_ACTION_SCALE = 0.05   # m per unit normalized action
LIBERO_ROT_ACTION_SCALE = 0.5    # rad per unit normalized action


def get_position_change(actions: np.ndarray, idx_from: int, idx_to: int) -> np.ndarray:
    """
    7-D delta vector for describe_move (xyz in meters, rpy in radians, gripper 0..1).

    LIBERO action layout: [norm_dx, norm_dy, norm_dz, norm_droll, norm_dpitch, norm_dyaw, gripper]
      - actions[:,0:3]: normalized xyz OSC commands → multiply by 0.05 m to get meters
      - actions[:,3:6]: normalized rpy OSC commands → multiply by 0.5 rad to get radians
      - actions[:,6]:   gripper command (1=open, 0=close); describe_move treats >0.5 as "open".
    """
    if actions.shape[0] == 0:
        return np.zeros(7, dtype=float)
    idx_to = min(idx_to, actions.shape[0])
    if idx_to <= idx_from:
        delta = np.zeros(6, dtype=float)
    else:
        delta = actions[idx_from:idx_to, :6].sum(axis=0)
    pos_m = delta[:3] * LIBERO_POS_ACTION_SCALE
    rot_rad = delta[3:6] * LIBERO_ROT_ACTION_SCALE
    g = float(actions[min(idx_to, actions.shape[0] - 1), 6])
    return np.concatenate([pos_m, rot_rad, [g]])


def segment_end_indices(segments: list) -> dict:
    """Map segment_id → last-occurrence frame index."""
    end_idx = {}
    for i, sid in enumerate(segments):
        end_idx[sid] = i
    return end_idx


def build_long_plan_text(frame_annotations: list, segments: list) -> str:
    """
    Concatenate distinct subtask labels in order of first appearance into a
    numbered LONG PLAN (motion-only mode — no 2D waypoint coords).
    """
    items = []
    prev = None
    for i, ann in enumerate(frame_annotations):
        st = ann[0] if isinstance(ann, list) and len(ann) >= 1 else "unknown"
        if st == prev:
            continue
        items.append(st)
        prev = st
    return "\n".join(f"{j + 1}. {st}" for j, st in enumerate(items))


def normalize_episode_movements(frame_records: list) -> tuple[list, dict]:
    """Episode-level Q1/Q99 normalization of delta_full_state into [-1,1]."""
    if not frame_records:
        return frame_records, {"mean": [], "std": [], "Q1": [], "Q99": []}
    mv = np.array([r["delta_full_state"] for r in frame_records], dtype=float)
    low = np.percentile(mv, 1, axis=0)
    high = np.percentile(mv, 99, axis=0)
    normed = np.clip(2 * (mv - low) / (high - low + 1e-8) - 1, -1, 1)
    for i, rec in enumerate(frame_records):
        rec["delta_full_state_norm"] = normed[i].tolist()
    return frame_records, {
        "mean": mv.mean(axis=0).tolist(),
        "std": mv.std(axis=0).tolist(),
        "Q1": low.tolist(),
        "Q99": high.tolist(),
    }


# ───── core: label one episode ──────────────────────────────────────────────


def label_one_episode(
    suite_dir: Path,
    suite_name: str,
    ep_idx: int,
    instruction: str,
    subtask_list: list[str],
    fps: int,
    spatial_indices: list[int],
    orient_indices: list[int],
    gripper_indices: list[int],
    model: QwenVLLM | None,
    dry_run: bool = False,
    max_frames_per_segment: int = 5,
) -> dict | None:
    """Run the full pipeline on one episode and return the cot_annotations.json payload."""
    states, actions, _ = read_episode_states_actions(suite_dir, ep_idx)
    n = len(states)
    if n == 0:
        return None

    # Triple segmentation
    spatial_state = states[:, spatial_indices]
    orient_state = states[:, orient_indices]
    gripper_qpos = states[:, gripper_indices]
    overall_segment = triple_segment(spatial_state, orient_state, gripper_qpos, fps=fps)

    # Sample key frames per segment from primary camera
    vp = video_path(suite_dir, ep_idx, "observation.images.image")
    frames = read_video_frames(vp) if vp.exists() else None
    if frames is None or len(frames) < n:
        # Some LIBERO videos may be slightly shorter than parquet — truncate to min
        if frames:
            n = min(n, len(frames))
            states = states[:n]
            actions = actions[:n]
            overall_segment = overall_segment[:n]
        else:
            print(f"[warn] no frames for {suite_name} ep{ep_idx}; skipping VLM, using empty subtasks")
            frames = []

    if frames:
        # Convert RGB ndarrays to PIL
        pil_frames = [Image.fromarray(f, "RGB") for f in frames[:n]]
        key_frames_per_segment, frame_ranges = get_key_frames_per_segment(
            pil_frames, overall_segment, max_frames_per_segment=max_frames_per_segment
        )
    else:
        key_frames_per_segment, frame_ranges = [], []

    segment_count = len(key_frames_per_segment)

    # VLM annotation: one call per segment (reuses robocasa_labeling segment_to_subtask_single)
    if dry_run or model is None or segment_count == 0:
        segment_labels = {i + 1: ["dry_run", "dry_run"] for i in range(segment_count)}
    else:
        segment_labels = {}
        prev_subtask = None
        for seg_num, seg_key_frames in enumerate(key_frames_per_segment, 1):
            frame_range = frame_ranges[seg_num - 1]
            label = segment_to_subtask_single(
                instruction, subtask_list, seg_key_frames,
                seg_num, segment_count, frame_range, n, model,
                prev_subtask=prev_subtask,
                use_dual_view=False,
            )
            if isinstance(label, list):
                segment_labels[seg_num] = label
                if label[0] in subtask_list:
                    prev_subtask = label[0]
            else:
                segment_labels[seg_num] = ["error", str(label)]

    frame_annotations = build_frame_annotations(overall_segment, segment_labels)

    # Build per-frame records
    long_plan = build_long_plan_text(frame_annotations, overall_segment)
    assistant_plan_level = LONG_PLAN_TEMPLATE.format(long_plan=long_plan)
    seg_end = segment_end_indices(overall_segment)

    frame_records = []
    for i in range(n):
        ann = frame_annotations[i] if i < len(frame_annotations) else ["unknown", "unknown"]
        if not isinstance(ann, list) or len(ann) < 2:
            ann = ["unknown", "unknown"]

        next_i = i + 1 if i + 1 < n else i
        delta = get_position_change(actions, i, next_i)
        current_movement = describe_move(delta)

        seg_id = overall_segment[i] if i < len(overall_segment) else None
        seg_end_i = min(seg_end.get(seg_id, i), n - 1)
        delta_subtask = get_position_change(actions, i, seg_end_i)
        subtask_movement = describe_move(delta_subtask)

        user_text = INPUT_TEMPLATE.format(instruction=instruction)
        assistant_short_plan = SHORT_PLAN_TEMPLATE.format(subtask=ann[0], reasoning=ann[1])
        assistant_movement_level = MOVEMENT_LEVEL_TEMPLATE.format(
            current_movement=current_movement,
            subtask_movement=subtask_movement,
        )

        frame_records.append({
            "frame_index": i,
            "segment_id": int(seg_id) if seg_id is not None else None,
            "subtask": ann[0],
            "reason": ann[1],
            "user": user_text,
            "assistant_plan_level": assistant_plan_level,
            "assistant_short_plan": assistant_short_plan,
            "assistant_movement_level": assistant_movement_level,
            # Motion-only: position/object/grounding fields are null (filled in Phase D)
            "assistant_position_level": None,
            "assistant_object_level": None,
            "gripper_2d": None,
            "task_obj_bbox": None,
            "distractor_bboxes": None,
            "delta_full_state": delta.tolist(),
            "delta_full_state_norm": [],
            "movement_text": current_movement,
        })

    frame_records, mv_stats = normalize_episode_movements(frame_records)

    return {
        "sample_key": f"{suite_name}|{ep_idx}",
        "instruction": instruction,
        "suite": suite_name,
        "num_frames": n,
        "segment_count": segment_count,
        "movement_statistics": mv_stats,
        "frames": frame_records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=str)
    p.add_argument("--output_root", required=True, type=str)
    p.add_argument("--suite", default="all", choices=["all", *LIBERO_SUITES])
    p.add_argument("--start_ep", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=0,
                   help="0 = all episodes in suite")
    p.add_argument("--instruction_mapping", required=True,
                   help="Path to libero_instruction_subtask_mapping.json")
    p.add_argument("--vlm_backend", choices=["http", "hf_local"], default="http",
                   help="http=OpenAI-compat (vLLM, default — Qwen3.5-9B); hf_local=in-process transformers")
    p.add_argument("--api_url", default="http://shou_node09:8101/v1/chat/completions",
                   help="vLLM OpenAI-compat endpoint (set to actual host:port if different)")
    p.add_argument("--model_name", default="Qwen3.5-9B",
                   help="vLLM registered name (http) or absolute path (hf_local)")
    p.add_argument("--device", default="cuda:0", help="Used when --vlm_backend hf_local")
    p.add_argument("--max_frames_per_segment", type=int, default=5)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()

    # Load instruction → subtask mapping
    with open(args.instruction_mapping) as f:
        instr_mapping = json.load(f)
    print(f"[info] Loaded {len(instr_mapping)} instruction → subtasks entries")

    # VLM client. Disable Qwen3.5's built-in thinking mode (which would otherwise
    # consume max_tokens on a "Thinking Process:" preamble before the answer).
    if args.dry_run:
        model = None
        print("[info] Dry-run mode — VLM disabled")
    elif args.vlm_backend == "http":
        model = QwenVLLM(
            api_url=args.api_url,
            model_name=args.model_name,
            chat_template_kwargs={"enable_thinking": False},
        )
    else:
        model = HFQwenVLClient(model_path=args.model_name, device=args.device)

    suites = LIBERO_SUITES if args.suite == "all" else [args.suite]
    suite_dirs = [(s, libero_suite_dir(data_root, s)) for s in suites]

    total_done = 0
    for suite_name, sdir in suite_dirs:
        if not (sdir / "meta" / "info.json").exists():
            print(f"[warn] suite {suite_name} not found at {sdir}, skipping")
            continue

        modality_dict, fps = load_dataset_meta(sdir)
        spatial_idx, orient_idx, gripper_idx = get_segmentation_indices(modality_dict)
        eps = load_episode_lengths(sdir)
        total = len(eps)
        end_ep = total if args.num_episodes <= 0 else min(args.start_ep + args.num_episodes, total)
        ep_indices = list(range(args.start_ep, end_ep))

        print(f"\n[suite {suite_name}] {len(ep_indices)} episodes, fps={fps}")

        for ep_idx in tqdm(ep_indices, desc=suite_name):
            _ep_idx, _length, instruction = eps[ep_idx]
            mapping = instr_mapping.get(instruction)
            if not mapping or not mapping.get("subtasks"):
                tqdm.write(f"[skip] no subtask mapping for: '{instruction[:60]}'")
                continue

            out_dir = output_root / suite_name / "extras" / f"episode_{ep_idx:06d}"
            out_path = out_dir / "cot_annotations.json"
            if out_path.exists() and not args.overwrite:
                continue

            try:
                payload = label_one_episode(
                    suite_dir=sdir,
                    suite_name=suite_name,
                    ep_idx=ep_idx,
                    instruction=instruction,
                    subtask_list=mapping["subtasks"],
                    fps=fps,
                    spatial_indices=spatial_idx,
                    orient_indices=orient_idx,
                    gripper_indices=gripper_idx,
                    model=model,
                    dry_run=args.dry_run,
                    max_frames_per_segment=args.max_frames_per_segment,
                )
            except Exception as e:
                tqdm.write(f"[error] {suite_name} ep{ep_idx}: {e}")
                continue
            if payload is None:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            total_done += 1

    print(f"\n[done] Wrote {total_done} cot_annotations.json under {output_root}")


if __name__ == "__main__":
    main()
