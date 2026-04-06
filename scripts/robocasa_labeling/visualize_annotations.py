"""
Visualize extracted annotations on video frames from the LeRobot MP4 files.

Overlays gripper dots, object bounding boxes, distractor bboxes, and
optional subtask segment labels onto video frames to verify extraction
quality before running the full pipeline.

Usage:
    # Basic — overlay extracted annotations
    python scripts/robocasa_labeling/visualize_annotations.py \
        --dataset   datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
        --extracted scripts/robocasa_labeling/annotations/PickPlaceCounterToCabinet/extracted_episodes.json \
        --output    tmp/annotation_viz \
        --max_episodes 3

    # With subtask overlay
    python scripts/robocasa_labeling/visualize_annotations.py \
        --dataset   ... --extracted ... \
        --subtasks  scripts/robocasa_labeling/annotations/subtask_results/subtasks_PickPlaceCounterToCabinet.json \
        --output    tmp/annotation_viz \
        --max_episodes 3
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ─── colours ──────────────────────────────────────────────────────────────────

COLOR_GRIPPER = (0, 255, 0)       # green  (BGR)
COLOR_TASK_OBJ = (0, 0, 255)      # red
COLOR_DISTRACTOR = (255, 165, 0)  # orange
SEGMENT_COLORS = [
    (255,   0,   0), (0, 255,   0), (0,   0, 255),
    (255, 255,   0), (0, 255, 255), (255, 0, 255),
    (128, 128,   0), (0, 128, 128), (128, 0, 128),
    (255, 128,   0),
]


# ─── drawing helpers ──────────────────────────────────────────────────────────

def draw_gripper(frame, pt, color=COLOR_GRIPPER, radius=4):
    if pt is not None:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius + 1, (255, 255, 255), 1)


def draw_bbox(frame, bbox, label, color, thickness=2):
    if bbox is None:
        return
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(frame, label, (x1 + 1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_segment_bar(frame, seg_label, total_segs, frame_idx, total_frames):
    """Draw a small coloured bar at the top with segment info."""
    color = SEGMENT_COLORS[seg_label % len(SEGMENT_COLORS)]
    h, w = frame.shape[:2]
    bar_h = 14
    cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
    # progress bar
    prog_w = int(w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (0, 0), (prog_w, bar_h), color, -1)
    text = f"seg {seg_label}/{total_segs}  f{frame_idx}"
    cv2.putText(frame, text, (4, bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


# ─── video helpers ────────────────────────────────────────────────────────────

def get_video_path(dataset_path: Path, ep_idx: int, camera: str) -> Path:
    pattern = f"videos/*/observation.images.{camera}/episode_{ep_idx:06d}.mp4"
    hits = list(dataset_path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Video not found: {pattern}")
    return hits[0]


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Return list of BGR frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─── main visualisation ──────────────────────────────────────────────────────

def visualize_episode(dataset_path, ep_data, ep_key,
                      subtask_entry, camera, output_dir):
    """Render an annotated MP4 for one episode."""
    task_name, ep_idx_str = ep_key.split("|")
    ep_idx = int(ep_idx_str)

    video_path = get_video_path(dataset_path, ep_idx, camera)
    raw_frames = load_video_frames(video_path)

    cam = ep_data["cameras"].get(camera)
    if cam is None:
        print(f"  [warn] camera {camera} missing for {ep_key}")
        return

    gripper_2d = cam["gripper_2d"]
    task_obj_bbox = cam["task_obj_bbox"]
    distractor_bboxes = cam["distractor_bboxes"]
    obj_cat = ep_data.get("obj_cat", "obj")
    distr_cats = ep_data.get("distr_cats", [])

    n_frames = min(len(raw_frames), len(gripper_2d))

    # Subtask segments
    segments = None
    total_segs = 0
    if subtask_entry is not None:
        segments = subtask_entry.get("segments")
        total_segs = subtask_entry.get("segment_count", 0)

    # Build annotated frames
    annotated = []
    for t in range(n_frames):
        frame = raw_frames[t].copy()

        # Segment bar
        if segments is not None and t < len(segments):
            seg_label = int(segments[t] / 100) if segments[t] > 10 else int(segments[t])
            draw_segment_bar(frame, seg_label, total_segs, t, n_frames)

        # Gripper
        draw_gripper(frame, gripper_2d[t])

        # Task object bbox
        bbox = task_obj_bbox[t] if t < len(task_obj_bbox) else None
        draw_bbox(frame, bbox, obj_cat, COLOR_TASK_OBJ)

        # Distractor bboxes
        if t < len(distractor_bboxes) and distractor_bboxes[t]:
            for i, dbbox in enumerate(distractor_bboxes[t]):
                name = distr_cats[i] if i < len(distr_cats) else f"distr_{i}"
                draw_bbox(frame, dbbox, name, COLOR_DISTRACTOR)

        annotated.append(frame)

    # Write video
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_name}_ep{ep_idx:04d}_{camera}.mp4"
    h, w = annotated[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30, (w, h))
    for f in annotated:
        writer.write(f)
    writer.release()
    print(f"  -> {out_path}  ({n_frames} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize RoboCasa annotations")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--extracted", type=str, required=True,
                        help="extracted_episodes.json")
    parser.add_argument("--subtasks", type=str, default=None,
                        help="subtask results JSON (optional)")
    parser.add_argument("--output", type=str, default="tmp/annotation_viz")
    parser.add_argument("--camera", type=str, default="robot0_agentview_left")
    parser.add_argument("--max_episodes", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print(f"Loading extracted annotations: {args.extracted}")
    with open(args.extracted) as f:
        extracted = json.load(f)
    print(f"  {len(extracted)} episodes")

    subtask_data = {}
    if args.subtasks:
        print(f"Loading subtask results: {args.subtasks}")
        with open(args.subtasks) as f:
            subtask_data = json.load(f)
        print(f"  {len(subtask_data)} episodes")

    keys = list(extracted.keys())[:args.max_episodes]
    print(f"\nVisualizing {len(keys)} episodes with camera={args.camera}")
    for key in tqdm(keys, desc="Rendering"):
        try:
            visualize_episode(
                dataset_path,
                extracted[key],
                key,
                subtask_data.get(key),
                args.camera,
                args.output,
            )
        except Exception as e:
            print(f"  [error] {key}: {e}")

    print(f"\nDone. Videos saved to: {args.output}")


if __name__ == "__main__":
    main()
