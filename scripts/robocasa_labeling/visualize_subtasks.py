"""
Visualize per-frame subtask annotations on RoboCasa episode videos.

Reads the subtask result JSON produced by generate_subtasks.py, loads the
original MP4 video, and renders an annotated video with:
- A colour-coded segment bar at the top
- The current subtask name overlaid on each frame
- A text panel below the video with instruction, subtask and reason

Usage:
    python scripts/robocasa_labeling/visualize_subtasks.py \
        --dataset  datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
        --subtasks datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/subtasks/subtasks_PickPlaceCounterToCabinet.json \
        --output   tmp/subtask_viz \
        --max_episodes 3
"""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ─── colours ──────────────────────────────────────────────────────────────────

SEGMENT_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def subtask_color(subtask_name: str) -> tuple:
    """Deterministic colour for a subtask string (RGB)."""
    idx = hash(subtask_name) % len(SEGMENT_PALETTE)
    return SEGMENT_PALETTE[idx]


def rgb_to_bgr(c):
    return (c[2], c[1], c[0])


# ─── font ─────────────────────────────────────────────────────────────────────

def load_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ─── drawing helpers ──────────────────────────────────────────────────────────

def draw_segment_bar(frame, subtask, seg_idx, total_segs, frame_idx, total_frames):
    """Coloured progress bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 20
    color_rgb = subtask_color(subtask)
    color_bgr = rgb_to_bgr(color_rgb)

    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)
    prog_w = int(w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (0, 0), (prog_w, bar_h), color_bgr, -1)

    label = f"Seg {seg_idx}/{total_segs}  Frame {frame_idx}/{total_frames}"
    cv2.putText(frame, label, (6, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


def draw_subtask_label(frame, subtask):
    """Semi-transparent banner with subtask name at the bottom of the image."""
    h, w = frame.shape[:2]
    banner_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    color_bgr = rgb_to_bgr(subtask_color(subtask))
    cv2.putText(frame, subtask, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_bgr, 1, cv2.LINE_AA)


def create_text_panel(instruction, subtask, reason, frame_idx, img_width):
    """PIL-rendered text panel below the video frame."""
    panel_h = 160
    panel = np.ones((panel_h, img_width, 3), dtype=np.uint8) * 245

    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(13)
    font_small = load_font(11)

    y = 6
    line_h = 16

    draw.text((8, y), f"Frame {frame_idx}", fill=(100, 100, 100), font=font_small)
    y += line_h

    draw.text((8, y), "INSTRUCTION:", fill=(0, 0, 180), font=font)
    y += line_h
    for line in textwrap.wrap(instruction, width=80):
        draw.text((12, y), line, fill=(30, 30, 30), font=font_small)
        y += line_h - 2
    y += 4

    color_rgb = subtask_color(subtask)
    draw.text((8, y), "SUBTASK:", fill=color_rgb, font=font)
    y += line_h
    for line in textwrap.wrap(subtask, width=80):
        draw.text((12, y), line, fill=(30, 30, 30), font=font_small)
        y += line_h - 2
    y += 4

    draw.text((8, y), "REASON:", fill=(128, 0, 128), font=font)
    y += line_h
    for line in textwrap.wrap(reason, width=80):
        draw.text((12, y), line, fill=(60, 60, 60), font=font_small)
        y += line_h - 2

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ─── video I/O ────────────────────────────────────────────────────────────────

def get_video_path(dataset_path: Path, ep_idx: int, camera: str) -> Path:
    pattern = f"videos/*/observation.images.{camera}/episode_{ep_idx:06d}.mp4"
    hits = list(dataset_path.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Video not found: {pattern}")
    return hits[0]


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ─── per-episode visualisation ────────────────────────────────────────────────

def visualize_episode(dataset_path: Path, ep_key: str, entry: dict,
                      camera: str, output_dir: Path, fps: int = 20):
    task_name, ep_idx_str = ep_key.split("|")
    ep_idx = int(ep_idx_str)

    video_path = get_video_path(dataset_path, ep_idx, camera)
    raw_frames = load_video_frames(video_path)
    if not raw_frames:
        print(f"  [warn] no frames for {ep_key}")
        return False

    instruction = entry.get("instruction", "")
    segments = entry.get("segments", [])
    segment_count = entry.get("segment_count", 0)
    frame_annotations = entry.get("frame_annotations", [])

    n_frames = len(raw_frames)
    h, w = raw_frames[0].shape[:2]

    # Build segment-index mapping (overall_segment value → sequential 1-based idx)
    seen_segs: dict[float, int] = {}
    seg_order = 0
    for s in segments:
        if s not in seen_segs:
            seg_order += 1
            seen_segs[s] = seg_order

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_name}_ep{ep_idx:04d}_{camera}.mp4"

    panel_w = w
    text_panel_h = 160
    out_h = h + text_panel_h

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{panel_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for t in range(n_frames):
        frame = raw_frames[t].copy()

        if t < len(frame_annotations):
            subtask, reason = frame_annotations[t]
        else:
            subtask, reason = "unknown", "unknown"

        seg_idx = seen_segs.get(segments[t], 0) if t < len(segments) else 0

        draw_segment_bar(frame, subtask, seg_idx, segment_count, t, n_frames)
        draw_subtask_label(frame, subtask)

        text_panel = create_text_panel(instruction, subtask, reason, t, panel_w)
        combined = np.vstack([frame, text_panel])
        proc.stdin.write(combined.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")

    tqdm.write(f"  -> {out_path}  ({n_frames} frames)")
    return True


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-frame subtask annotations for RoboCasa episodes")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to lerobot dataset dir")
    parser.add_argument("--subtasks", type=str, required=True,
                        help="Subtask result JSON from generate_subtasks.py")
    parser.add_argument("--output", type=str, default="tmp/subtask_viz",
                        help="Output directory for annotated videos")
    parser.add_argument("--camera", type=str, default="robot0_agentview_left")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="Max episodes to render (0 = all)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--episode", type=int, default=None,
                        help="Visualize a single episode by index")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)

    print(f"Loading subtask results: {args.subtasks}")
    with open(args.subtasks) as f:
        subtask_data: dict = json.load(f)
    print(f"  {len(subtask_data)} episodes in result file")

    if args.episode is not None:
        keys = [k for k in subtask_data if k.endswith(f"|{args.episode}")]
        if not keys:
            print(f"[error] Episode {args.episode} not found in subtask results")
            return
    else:
        keys = list(subtask_data.keys())
        if args.max_episodes > 0:
            keys = keys[:args.max_episodes]

    print(f"Rendering {len(keys)} episodes  camera={args.camera}  fps={args.fps}")
    print(f"Output: {output_dir}\n")

    ok, fail = 0, 0
    for key in tqdm(keys, desc="Rendering"):
        try:
            success = visualize_episode(
                dataset_path, key, subtask_data[key],
                args.camera, output_dir, args.fps,
            )
            if success:
                ok += 1
            else:
                fail += 1
        except Exception as e:
            tqdm.write(f"  [error] {key}: {e}")
            fail += 1

    print(f"\nDone. {ok} videos saved to {output_dir}")
    if fail:
        print(f"  Failed: {fail}")


if __name__ == "__main__":
    main()
