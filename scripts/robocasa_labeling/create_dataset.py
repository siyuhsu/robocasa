"""
Create per-episode movement annotations for RoboCasa labeling.

This script merges:
1) subtask labels from generate_subtasks.py output (subtasks.json),
2) bbox / gripper annotations from extract_annotations.py output
   (extras/episode_xxxxxx/annotations.json),
3) action trajectories from the LeRobot dataset.

It only extracts movement labels (delta action + text description), then writes
one output JSON per episode to:
<output_root>/<dataset-relative>/extras/episode_xxxxxx/movement_annotations.json

Usage:
    # Single dataset
    python scripts/robocasa_labeling/create_dataset.py \
        --dataset datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
        --subtasks tmp/robocasa_subtasks/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/subtasks.json

    # Multi-dataset by subset registry (default subsets)
    python scripts/robocasa_labeling/create_dataset.py \
        --datasets_root datasets/v1.0 \
        --subtasks_root tmp/robocasa_subtasks \
        --bbox_root tmp/robocasa_grounding \
        --output_root tmp/robocasa_movement
"""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2

import robocasa.utils.lerobot_utils as LU

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import describe_move


CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]

COT_PANEL_H = 620


input_template = (
    "What action should the robot take to achieve the instruction\n"
    "INSTRUCTION: \n{instruction}\n"
    "CURRENT GRIPPER: {gripper_2d}\n"
)

long_plan_template = "LONG PLAN:\n{long_plan}\n"

position_level_template = (
    "NEXT GRIPPER: {gripper_2d_next}\n"
)

object_level_template = "OBJECT:\n{objects}\n"

short_plan_template = """SHORT PLAN:
Current Positions: {current_positions}
Subtask: {subtask}, move from {subtask_trajectory}
Subtask Reasoning: {reasoning}
"""

movement_level_template = """MOVEMENT:
Current Movement: {current_movement}
Subtask Movement: {subtask_movement}
"""


def parse_episode_index(sample_key: str) -> int:
    """Parse episode index from key like 'TaskName|12'."""
    if "|" not in sample_key:
        raise ValueError(f"Invalid sample key (missing '|'): {sample_key}")
    return int(sample_key.rsplit("|", 1)[1])


def resolve_subset_paths(subsets: List[str]) -> set:
    """Resolve DATASET_SOUP_REGISTRY subset names to canonical lerobot paths."""
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

    allowed = set()
    for name in subsets:
        if name not in DATASET_SOUP_REGISTRY:
            raise ValueError(
                "Unknown subset '%s'. Available: %s"
                % (name, sorted(DATASET_SOUP_REGISTRY.keys()))
            )
        for ds_meta in DATASET_SOUP_REGISTRY[name]:
            p = ds_meta.get("path")
            if p:
                allowed.add(str(Path(p).resolve()))
    return allowed


def discover_lerobot_datasets(datasets_root: Path, allowed_paths=None) -> List[Path]:
    """Discover all lerobot datasets under datasets_root."""
    discovered = []
    seen = set()

    for ep_file in sorted(datasets_root.rglob("meta/episodes.jsonl")):
        lerobot_dir = ep_file.parent.parent
        resolved = str(lerobot_dir.resolve())
        if allowed_paths is not None and resolved not in allowed_paths:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        discovered.append(lerobot_dir)

    return discovered


def load_lerobot_dataset(dataset_path: Path):
    """Load LeRobot dataset for RGB frame access in visualization mode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id="robocasa365", root=str(dataset_path))


def load_episode_frames(ds, ep_idx: int, cameras: List[str], num_workers: int = 8) -> Dict[str, List[np.ndarray]]:
    """Load RGB frames for all requested cameras in one episode."""
    from_idx = int(ds.episode_data_index["from"][ep_idx])
    to_idx = int(ds.episode_data_index["to"][ep_idx])
    n = to_idx - from_idx

    img_keys = ["observation.images.%s" % c for c in cameras]

    def _load_one(local_idx: int):
        sample = ds[from_idx + local_idx]
        imgs = {}
        for cam, key in zip(cameras, img_keys):
            tensor = sample[key]
            imgs[cam] = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return local_idx, imgs

    frames_dict = dict((c, [None] * n) for c in cameras)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load_one, i) for i in range(n)]
        for future in as_completed(futures):
            local_idx, imgs = future.result()
            for cam in cameras:
                frames_dict[cam][local_idx] = imgs[cam]
    return frames_dict


def get_position_change(actions: np.ndarray, idx_from: int, idx_to: int) -> np.ndarray:
    """
    Compute 7-D movement vector used by describe_move.

    Uses HDF5-order action accumulated between frames:
      delta_eef_pos(3) + delta_eef_rot(3) + gripper_close_next(1)

    gripper convention: 1=closed, while describe_move expects >0.5 as "open",
    so we invert gripper_close.
    """
    idx_to = min(idx_to, actions.shape[0] - 1)

    if idx_to <= idx_from:
        delta = np.zeros(6, dtype=float)
    else:
        delta = actions[idx_from:idx_to, :6].sum(axis=0)

    gripper_next = float(actions[min(idx_to, actions.shape[0] - 1), 6])
    gripper_for_desc = 1.0 - gripper_next

    return np.concatenate([delta[:3], delta[3:6], [gripper_for_desc]])


def split_bbox_fields(bbox_ep: dict, camera: str) -> Tuple[List, List, List[str], str, List[str]]:
    """Support both old and new bbox schema from extract_annotations.py."""
    cam_data = bbox_ep.get("cameras", {}).get(camera)
    if cam_data is None:
        raise ValueError("Camera '%s' not found in bbox annotations" % camera)

    gripper_2d = cam_data.get("gripper_2d", [])
    all_object_names = bbox_ep.get("all_object_names", [])

    # New schema: all_object_bboxes
    if "all_object_bboxes" in cam_data:
        all_boxes = cam_data.get("all_object_bboxes", [])
        task_obj_bbox = []
        distractor_bboxes = []
        for frame_boxes in all_boxes:
            if frame_boxes and len(frame_boxes) > 0:
                task_obj_bbox.append(frame_boxes[0])
                distractor_bboxes.append(frame_boxes[1:])
            else:
                task_obj_bbox.append(None)
                distractor_bboxes.append([])
    else:
        # Old schema fallback
        task_obj_bbox = cam_data.get("task_obj_bbox", [])
        distractor_bboxes = cam_data.get("distractor_bboxes", [])

    if bbox_ep.get("obj_cat"):
        obj_cat = bbox_ep.get("obj_cat", "")
    elif all_object_names:
        obj_cat = all_object_names[0]
    else:
        obj_cat = "object"

    if bbox_ep.get("distr_cats"):
        distr_cats = bbox_ep.get("distr_cats", [])
    elif len(all_object_names) > 1:
        distr_cats = all_object_names[1:]
    else:
        distr_cats = []

    return gripper_2d, task_obj_bbox, distractor_bboxes, obj_cat, distr_cats


def _bbox_center(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def _objects_text(frame_boxes: List, object_names: List[str]) -> str:
    lines = []
    for i, bbox in enumerate(frame_boxes):
        if bbox is None or len(bbox) != 4:
            continue
        name = object_names[i] if i < len(object_names) else "obj_%d" % i
        x1, y1, x2, y2 = [int(v) for v in bbox]
        lines.append("%s: [%d,%d], [%d,%d]" % (name, x1, y1, x2, y2))
    return "\n".join(lines)


def _current_positions_text(gripper_pt, frame_boxes: List, object_names: List[str]) -> str:
    parts = []
    if gripper_pt is not None and len(gripper_pt) == 2:
        parts.append("Gripper [%d, %d]" % (int(gripper_pt[0]), int(gripper_pt[1])))
    for i, bbox in enumerate(frame_boxes):
        ctr = _bbox_center(bbox)
        if ctr is None:
            continue
        name = object_names[i] if i < len(object_names) else "obj_%d" % i
        parts.append("%s [%d, %d]" % (name, ctr[0], ctr[1]))
    return ", ".join(parts)


def _segment_end_indices(segments: List) -> Dict:
    end_idx = {}
    for i, sid in enumerate(segments):
        end_idx[sid] = i
    return end_idx


def _subtask_trajectory_text(grippers: List, frame_idx: int, end_idx: int) -> str:
    n = len(grippers)
    if n == 0:
        return "[0, 0], [0, 0], [0, 0]"

    frame_idx = max(0, min(frame_idx, n - 1))
    end_idx = max(frame_idx, min(end_idx, n - 1))
    mid_idx = (frame_idx + end_idx) // 2

    def _safe_pt(idx):
        pt = grippers[idx] if idx < n else None
        if pt is None or len(pt) != 2:
            return [0, 0]
        return [int(pt[0]), int(pt[1])]

    p1 = _safe_pt(frame_idx)
    p2 = _safe_pt(mid_idx)
    p3 = _safe_pt(end_idx)
    return "[%d, %d], [%d, %d], [%d, %d]" % (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])


def _build_long_plan_text(frame_annotations: List, grippers: List, segments: List) -> str:
    items = []
    prev_subtask = None
    for i in range(len(frame_annotations)):
        ann = frame_annotations[i] if i < len(frame_annotations) else ["unknown", "unknown"]
        subtask = ann[0] if isinstance(ann, list) and len(ann) >= 1 else "unknown"
        if subtask == prev_subtask:
            continue
        sid = segments[i] if i < len(segments) else None
        items.append((i, sid, subtask))
        prev_subtask = subtask

    seg_end = _segment_end_indices(segments)
    lines = []
    for idx, (frame_i, sid, subtask) in enumerate(items, 1):
        end_i = seg_end.get(sid, frame_i)
        traj = _subtask_trajectory_text(grippers, frame_i, end_i)
        lines.append("%d. %s, move from %s" % (idx, subtask, traj))
    return "\n".join(lines)


def normalize_episode_movements(frame_records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Normalize one episode's movement vectors into [-1, 1] by 1%-99% range."""
    if not frame_records:
        return frame_records, {"mean": [], "std": [], "Q1": [], "Q99": []}

    all_mv = np.array([r["delta_full_state"] for r in frame_records], dtype=float)
    mean = np.mean(all_mv, axis=0)
    std = np.std(all_mv, axis=0)
    low = np.percentile(all_mv, 1, axis=0)
    high = np.percentile(all_mv, 99, axis=0)

    normed = 2 * (all_mv - low) / (high - low + 1e-8) - 1
    normed = np.clip(normed, -1, 1)

    for i, record in enumerate(frame_records):
        record["delta_full_state_norm"] = normed[i].tolist()

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "Q1": low.tolist(),
        "Q99": high.tolist(),
    }
    return frame_records, stats


def _load_font(size: int = 14):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _color_for_idx(i: int) -> Tuple[int, int, int]:
    rgb = _PALETTE[i % len(_PALETTE)]
    return (rgb[2], rgb[1], rgb[0])


def _draw_traj(frame_bgr: np.ndarray, grippers: List, idx: int, trail_len: int = 20):
    start = max(0, idx - trail_len)
    pts = []
    for j in range(start, idx + 1):
        pt = grippers[j] if j < len(grippers) else None
        if pt is None:
            continue
        pts.append((int(pt[0]), int(pt[1])))
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(frame_bgr, pts[i - 1], pts[i], (0, 0, 255), 1, cv2.LINE_AA)


def _draw_cam_overlay(frame_bgr: np.ndarray, cam_data: dict, object_names: List[str], frame_idx: int, cam_name: str):
    grippers = cam_data.get("gripper_2d", [])
    all_boxes = cam_data.get("all_object_bboxes", [])

    if frame_idx < len(all_boxes):
        frame_boxes = all_boxes[frame_idx]
        if frame_boxes:
            for bi, bbox in enumerate(frame_boxes):
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                color = _color_for_idx(bi)
                label = object_names[bi] if bi < len(object_names) else "obj_%d" % bi
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, label, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    _draw_traj(frame_bgr, grippers, frame_idx, trail_len=24)
    if frame_idx < len(grippers) and grippers[frame_idx] is not None:
        p = grippers[frame_idx]
        cv2.circle(frame_bgr, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

    cv2.putText(frame_bgr, cam_name, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _make_cot_panel(payload: dict, frame_idx: int, width: int) -> np.ndarray:
    panel_h = COT_PANEL_H
    panel = np.ones((panel_h, width, 3), dtype=np.uint8) * 245
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(13)
    font_s = _load_font(11)
    y, lh = 6, 16

    frames = payload.get("frames", [])
    rec = frames[frame_idx] if frame_idx < len(frames) else {}

    user_text = str(rec.get("user", ""))
    long_plan_text = str(rec.get("assistant_plan_level", ""))
    short_plan_text = str(rec.get("assistant_short_plan", ""))
    position_text = str(rec.get("assistant_position_level", ""))
    object_text = str(rec.get("assistant_object_level", ""))
    movement_text = str(rec.get("assistant_movement_level", ""))

    max_chars = max(40, width // 9)

    def _draw_block(title: str, content: str, title_color, max_lines: int = 999):
        nonlocal y
        if y >= panel_h - lh:
            return
        draw.text((8, y), title, fill=title_color, font=font)
        y += lh
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            y += 2
            return
        used = 0
        for raw in lines:
            for ln in textwrap.wrap(raw, width=max_chars):
                if used >= max_lines:
                    return
                if y >= panel_h - lh:
                    return
                draw.text((12, y), ln, fill=(40, 40, 40), font=font_s)
                y += lh - 2
                used += 1
        y += 2

    draw.text((8, y), "Frame %d" % frame_idx, fill=(90, 90, 90), font=font_s)
    y += lh
    _draw_block("USER:", user_text, (0, 0, 170), max_lines=6)
    # Limit long-plan lines so SHORT PLAN movement fields remain visible.
    _draw_block("LONG PLAN:", long_plan_text, (170, 0, 0), max_lines=10)
    _draw_block("SHORT PLAN:", short_plan_text, (0, 120, 0), max_lines=16)
    _draw_block("MOVEMENT:", movement_text, (130, 0, 130), max_lines=8)
    _draw_block("POSITION LEVEL:", position_text, (120, 0, 120), max_lines=4)
    _draw_block("OBJECT:", object_text, (130, 70, 0), max_lines=8)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _to_even_hw(h: int, w: int) -> Tuple[int, int]:
    """yuv420p requires even height / width."""
    return h - (h % 2), w - (w % 2)


def render_episode_video(frames_by_cam: Dict[str, List[np.ndarray]], bbox_ep: dict, payload: dict, out_path: Path, fps: int = 20):
    """Render 3-view stitched video with bbox, gripper trajectory and COT text panel."""
    active = [c for c in CAMERA_NAMES if c in frames_by_cam and frames_by_cam[c]]
    if len(active) == 0:
        return

    base_h, base_w = frames_by_cam[active[0]][0].shape[:2]
    base_h, base_w = _to_even_hw(base_h, base_w)

    # Normalize all camera frames to a shared size to keep rawvideo stream shape stable.
    norm_frames = {}
    for cam in active:
        cam_frames = []
        for fr in frames_by_cam[cam]:
            if fr.shape[0] != base_h or fr.shape[1] != base_w:
                fr = cv2.resize(fr, (base_w, base_h), interpolation=cv2.INTER_AREA)
            else:
                fr = fr[:base_h, :base_w]
            cam_frames.append(fr)
        norm_frames[cam] = cam_frames

    n = min(len(frames_by_cam[c]) for c in active)
    out_w = base_w * len(active)
    panel_h = COT_PANEL_H
    out_h = base_h + panel_h
    out_h, out_w = _to_even_hw(out_h, out_w)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", "%dx%d" % (out_w, out_h),
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    object_names = bbox_ep.get("all_object_names", [])
    cam_ann = bbox_ep.get("cameras", {})

    for i in range(n):
        stitched = []
        for cam in active:
            frame_bgr = cv2.cvtColor(norm_frames[cam][i], cv2.COLOR_RGB2BGR)
            _draw_cam_overlay(frame_bgr, cam_ann.get(cam, {}), object_names, i, cam)
            cv2.putText(frame_bgr, "Frame %d/%d" % (i + 1, n), (6, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            stitched.append(frame_bgr)

        top = cv2.hconcat(stitched)
        panel = _make_cot_panel(payload, i, out_w)
        if panel.shape[0] != panel_h or panel.shape[1] != out_w:
            panel = cv2.resize(panel, (out_w, panel_h), interpolation=cv2.INTER_AREA)
        composed = np.vstack([top, panel])
        if composed.shape[0] != out_h or composed.shape[1] != out_w:
            composed = composed[:out_h, :out_w]
        proc.stdin.write(composed.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError("ffmpeg failed (rc=%d): %s" % (proc.returncode, stderr[-500:]))


def build_episode_output(
    sample_key: str,
    subtask_ep: dict,
    bbox_ep: dict,
    actions: np.ndarray,
    camera: str,
) -> dict:
    """Build the final per-episode annotation JSON payload."""
    gripper_2d, task_obj_bbox, distractor_bboxes, obj_cat, distr_cats = split_bbox_fields(bbox_ep, camera)
    all_object_names = bbox_ep.get("all_object_names", [])

    frame_annotations = subtask_ep.get("frame_annotations", [])
    segments = subtask_ep.get("segments", [])

    n = min(len(gripper_2d), len(task_obj_bbox), len(distractor_bboxes), len(frame_annotations), actions.shape[0])
    if n <= 0:
        raise ValueError("Empty episode after alignment")

    long_plan = _build_long_plan_text(frame_annotations, gripper_2d, segments)
    assistant_plan_level = long_plan_template.format(long_plan=long_plan)
    seg_end = _segment_end_indices(segments)

    frame_records = []
    for i in range(n):
        ann = frame_annotations[i] if i < len(frame_annotations) else ["unknown", "unknown"]
        if not isinstance(ann, list) or len(ann) != 2:
            ann = ["unknown", str(ann)]

        next_i = i + 1 if i + 1 < n else i
        delta = get_position_change(actions, i, next_i)
        current_movement = describe_move(delta)

        seg_id = segments[i] if i < len(segments) else None
        seg_end_i = seg_end.get(seg_id, i)
        seg_end_i = min(seg_end_i, n - 1)
        delta_to_subtask_end = get_position_change(actions, i, seg_end_i)
        subtask_movement = describe_move(delta_to_subtask_end)

        frame_boxes = []
        frame_boxes.append(task_obj_bbox[i])
        frame_boxes.extend(distractor_bboxes[i])

        current_positions = _current_positions_text(gripper_2d[i], frame_boxes, all_object_names)
        subtask_traj = _subtask_trajectory_text(gripper_2d, i, seg_end_i)

        user_text = input_template.format(
            instruction=subtask_ep.get("instruction", bbox_ep.get("instruction", "")),
            gripper_2d=gripper_2d[i],
        )
        assistant_position_level = position_level_template.format(
            gripper_2d_next=gripper_2d[next_i],
        )
        objects_text = _objects_text(frame_boxes, all_object_names)
        assistant_object_level = object_level_template.format(objects=objects_text) if objects_text else ""
        assistant_short_plan = short_plan_template.format(
            current_positions=current_positions,
            subtask=ann[0],
            subtask_trajectory=subtask_traj,
            reasoning=ann[1],
        )
        assistant_movement_level = movement_level_template.format(
            current_movement=current_movement,
            subtask_movement=subtask_movement,
        )

        frame_records.append(
            {
                "current_image_path": "episode_%06d/frame_%d.jpg" % (parse_episode_index(sample_key), i),
                "user": user_text,
                "assistant_plan_level": assistant_plan_level,
                "assistant_short_plan": assistant_short_plan,
                "assistant_position_level": assistant_position_level,
                "assistant_object_level": assistant_object_level,
                "assistant_movement_level": assistant_movement_level,
                "frame_index": i,
                "segment_id": seg_id,
                "subtask": ann[0],
                "reason": ann[1],
                "gripper_2d": gripper_2d[i],
                "task_obj_bbox": task_obj_bbox[i],
                "distractor_bboxes": distractor_bboxes[i],
                "delta_full_state": delta.tolist(),
                "delta_full_state_norm": [],
                "movement_text": current_movement,
            }
        )

    frame_records, movement_stats = normalize_episode_movements(frame_records)

    return {
        "sample_key": sample_key,
        "instruction": subtask_ep.get("instruction", bbox_ep.get("instruction", "")),
        "camera": camera,
        "obj_cat": obj_cat,
        "distr_cats": distr_cats,
        "num_frames": n,
        "movement_statistics": movement_stats,
        "frames": frame_records,
    }


def create_dataset(
    dataset_path: Path,
    bbox_dataset_path: Path,
    output_dataset_path: Path,
    subtask_data: dict,
    camera: str,
    output_name: str,
    overwrite: bool,
    visualize: bool,
    viz_fps: int,
    num_workers: int,
) -> dict:
    """Create and save per-episode movement annotations."""
    stats = {
        "total_subtask_entries": len(subtask_data),
        "saved": 0,
        "missing_bbox": 0,
        "missing_actions": 0,
        "camera_missing": 0,
        "invalid_episode": 0,
        "viz_saved": 0,
        "viz_failed": 0,
    }

    ds = None
    viz_done_for_dataset = False
    if visualize:
        ds = load_lerobot_dataset(dataset_path)

    for sample_key, subtask_ep in tqdm(subtask_data.items(), desc="Building movement annotations"):
        try:
            ep_idx = parse_episode_index(sample_key)
        except Exception:
            stats["invalid_episode"] += 1
            continue

        episode_dir = output_dataset_path / "extras" / f"episode_{ep_idx:06d}"
        bbox_episode_dir = bbox_dataset_path / "extras" / f"episode_{ep_idx:06d}"
        bbox_path = bbox_episode_dir / "annotations.json"
        out_path = episode_dir / output_name

        if out_path.exists() and not overwrite:
            continue

        if not bbox_path.exists():
            stats["missing_bbox"] += 1
            continue

        try:
            actions = np.array(LU.get_episode_actions(dataset_path, ep_idx), dtype=float)
        except Exception:
            stats["missing_actions"] += 1
            continue

        if actions.ndim != 2 or actions.shape[1] < 7:
            stats["missing_actions"] += 1
            continue

        with open(bbox_path) as f:
            bbox_ep = json.load(f)

        try:
            payload = build_episode_output(sample_key, subtask_ep, bbox_ep, actions, camera)
        except ValueError as e:
            if "Camera" in str(e):
                stats["camera_missing"] += 1
            else:
                stats["invalid_episode"] += 1
            continue
        except Exception:
            stats["invalid_episode"] += 1
            continue

        episode_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        stats["saved"] += 1

        if visualize and not viz_done_for_dataset:
            try:
                frames_by_cam = load_episode_frames(ds, ep_idx, CAMERA_NAMES, num_workers=num_workers)
                viz_name = "%s_viz.mp4" % Path(output_name).stem
                viz_path = episode_dir / viz_name
                render_episode_video(frames_by_cam, bbox_ep, payload, viz_path, fps=viz_fps)
                stats["viz_saved"] += 1
                viz_done_for_dataset = True
            except Exception as e:
                stats["viz_failed"] += 1
                tqdm.write("  [viz-error] ep %d: %s" % (ep_idx, e))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create RoboCasa movement-only dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional: single lerobot dataset dir. If set, overrides --datasets_root/--subsets.",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="datasets/v1.0",
        help="Root dir containing pretrain/ and target/",
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        default="",
        help="Optional: subtask JSON path for single-dataset mode",
    )
    parser.add_argument(
        "--subtasks_root",
        type=str,
        default="tmp/robocasa_subtasks",
        help="Root dir of subtask outputs in multi-dataset mode (keeps dataset-relative structure)",
    )
    parser.add_argument(
        "--bbox_dataset",
        type=str,
        default="",
        help="Optional: bbox dataset dir for single-dataset mode (contains extras/episode_xxxxxx/annotations.json)",
    )
    parser.add_argument(
        "--bbox_root",
        type=str,
        default="tmp/robocasa_grounding",
        help="Root dir of bbox outputs in multi-dataset mode (keeps dataset-relative structure)",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="",
        help="Optional: output dataset dir for single-dataset mode",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="tmp/robocasa_movement",
        help="Root dir of movement outputs in multi-dataset mode (keeps dataset-relative structure)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["target_no_nav", "pretrain_human300_no_navigation"],
        help="DATASET_SOUP_REGISTRY subset names to process. Pass '' to disable filtering.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for parallel runs (default: 1, disabled)",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards-1]",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_agentview_left",
        help="Camera used for bbox/gripper annotations",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="movement_annotations.json",
        help="Output filename under each extras/episode_xxxxxx/ directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-episode output files",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render per-episode 3-view video with bbox, gripper trajectory and COT text",
    )
    parser.add_argument(
        "--viz_fps",
        type=int,
        default=20,
        help="FPS for rendered visualization videos",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Thread workers for loading episode RGB frames in visualization mode",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards")

    datasets_root = Path(args.datasets_root)
    subtasks_root = Path(args.subtasks_root)
    bbox_root = Path(args.bbox_root)
    output_root = Path(args.output_root)

    if args.dataset:
        dataset_paths = [Path(args.dataset)]
        print("[info] Single-dataset mode: %s" % dataset_paths[0])
    else:
        allowed_paths = None
        if args.subsets and args.subsets != [""]:
            allowed_paths = resolve_subset_paths(args.subsets)
            print("[info] Filtering to subsets: %s (%d dataset paths)" % (args.subsets, len(allowed_paths)))
        dataset_paths = discover_lerobot_datasets(datasets_root, allowed_paths=allowed_paths)
        print("[info] Found %d datasets to process" % len(dataset_paths))

    if args.num_shards > 1:
        if args.dataset:
            print(
                "[info] Shard mode enabled: %d/%d (single-dataset mode: dataset-level sharding is ignored)"
                % (args.shard_id, args.num_shards)
            )
        else:
            dataset_paths = [p for i, p in enumerate(dataset_paths) if i % args.num_shards == args.shard_id]
            print(
                "[info] Shard mode enabled: %d/%d, assigned %d datasets"
                % (args.shard_id, args.num_shards, len(dataset_paths))
            )

    if not dataset_paths:
        print("[warn] No datasets found to process.")
        return

    total_stats = {
        "datasets": len(dataset_paths),
        "saved": 0,
        "missing_subtasks": 0,
        "missing_bbox": 0,
        "missing_actions": 0,
        "camera_missing": 0,
        "invalid_episode": 0,
        "viz_saved": 0,
        "viz_failed": 0,
    }

    for dataset_i, dataset_path in enumerate(dataset_paths, 1):
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print("[warn] Dataset path not found: %s" % dataset_path)
            total_stats["missing_subtasks"] += 1
            continue

        if args.dataset:
            if args.subtasks:
                subtask_path = Path(args.subtasks)
            else:
                try:
                    rel = dataset_path.relative_to(datasets_root)
                    subtask_path = subtasks_root / rel / "subtasks.json"
                except ValueError:
                    subtask_path = subtasks_root / dataset_path.name / "subtasks.json"

            if args.bbox_dataset:
                bbox_dataset_path = Path(args.bbox_dataset)
            else:
                try:
                    rel = dataset_path.relative_to(datasets_root)
                    bbox_dataset_path = bbox_root / rel
                except ValueError:
                    bbox_dataset_path = dataset_path

            if args.output_dataset:
                output_dataset_path = Path(args.output_dataset)
            else:
                try:
                    rel = dataset_path.relative_to(datasets_root)
                    output_dataset_path = output_root / rel
                except ValueError:
                    output_dataset_path = output_root / dataset_path.name
        else:
            rel = dataset_path.relative_to(datasets_root)
            subtask_path = subtasks_root / rel / "subtasks.json"
            bbox_dataset_path = bbox_root / rel
            output_dataset_path = output_root / rel

        print("\n[dataset %d/%d] %s" % (dataset_i, len(dataset_paths), dataset_path))
        print("[info] Loading subtask results: %s" % subtask_path)
        print("[info] Loading bbox annotations from: %s/extras/episode_xxxxxx/annotations.json" % bbox_dataset_path)
        print("[info] Saving movement outputs to: %s/extras/episode_xxxxxx/%s" % (output_dataset_path, args.output_name))

        if not subtask_path.exists():
            print("[warn] Missing subtask file, skipping dataset")
            total_stats["missing_subtasks"] += 1
            continue

        with open(subtask_path) as f:
            subtask_data = json.load(f)
        print("[info] %d subtask entries" % len(subtask_data))

        stats = create_dataset(
            dataset_path=dataset_path,
            bbox_dataset_path=bbox_dataset_path,
            output_dataset_path=output_dataset_path,
            subtask_data=subtask_data,
            camera=args.camera,
            output_name=args.output_name,
            overwrite=args.overwrite,
            visualize=args.visualize,
            viz_fps=args.viz_fps,
            num_workers=args.num_workers,
        )

        print("[done] Movement annotation summary:")
        for k, v in stats.items():
            print("  %s: %s" % (k, v))

        total_stats["saved"] += stats.get("saved", 0)
        total_stats["missing_bbox"] += stats.get("missing_bbox", 0)
        total_stats["missing_actions"] += stats.get("missing_actions", 0)
        total_stats["camera_missing"] += stats.get("camera_missing", 0)
        total_stats["invalid_episode"] += stats.get("invalid_episode", 0)
        total_stats["viz_saved"] += stats.get("viz_saved", 0)
        total_stats["viz_failed"] += stats.get("viz_failed", 0)

    print("\n[done] Global movement annotation summary:")
    for k, v in total_stats.items():
        print("  %s: %s" % (k, v))

    print("\nOutput path pattern:")
    print("  <output_root>/<dataset-relative>/extras/episode_000000/%s" % args.output_name)
    if args.visualize:
        print("  <output_root>/<dataset-relative>/extras/episode_000000/%s_viz.mp4" % Path(args.output_name).stem)


if __name__ == "__main__":
    main()
