"""
Visualize LIBERO motion-only CoT annotations on episode videos.

Reads cot_annotations.json files produced by label_libero_episodes.py and
renders an annotated MP4 with:
  - Top: colour-coded segment progress bar (segment id -> deterministic colour)
  - Centre: original third-person frame from observation.images.image, plus a
            semi-transparent banner showing the current subtask
  - Bottom: COT text panel — INSTRUCTION, LONG PLAN (full per-task plan),
            CURRENT SUBTASK + reason, MOVEMENT (current + subtask totals)

Adapted from robocasa_labeling/visualize_subtasks.py — schema differences:
  • robocasa subtasks.json keys are "TaskName|ep_idx" with values
    {instruction, segments, segment_count, frame_annotations: [[st, reason]]}.
  • LIBERO cot_annotations.json is one file per episode under
    LIBERO_COT/<suite>/extras/episode_XXXXXX/cot_annotations.json with:
      {sample_key, instruction, suite, num_frames, segment_count,
       frames: [{frame_index, segment_id, subtask, reason,
                 assistant_plan_level, assistant_short_plan,
                 assistant_movement_level, ...}]}

Usage:
  python scripts/libero_labeling/visualize_libero_annotations.py \
      --cot_root playground/Datasets/LIBERO_COT \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --output tmp/libero_cot_viz \
      --suite libero_goal --num_episodes 5

  # Pick specific episodes (one suite, comma-separated indices)
  python scripts/libero_labeling/visualize_libero_annotations.py \
      --cot_root ... --data_root ... --output ... \
      --suite libero_object --episodes 0,5,12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import LIBERO_SUITES, read_video_frames, suite_dir, video_path


# ───── colour helpers (copied from robocasa visualize_subtasks) ─────────────

SEGMENT_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def subtask_color(subtask_name: str) -> tuple:
    return SEGMENT_PALETTE[hash(subtask_name) % len(SEGMENT_PALETTE)]


def rgb_to_bgr(c):
    return (c[2], c[1], c[0])


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


# ───── frame overlays ───────────────────────────────────────────────────────


def draw_segment_bar(frame, subtask, seg_idx, total_segs, frame_idx, total_frames):
    h, w = frame.shape[:2]
    bar_h = 20
    color_bgr = rgb_to_bgr(subtask_color(subtask))
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)
    prog_w = int(w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (0, 0), (prog_w, bar_h), color_bgr, -1)
    label = f"Seg {seg_idx}/{total_segs}  Frame {frame_idx}/{total_frames}"
    cv2.putText(frame, label, (6, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


def draw_subtask_banner(frame, subtask):
    h, w = frame.shape[:2]
    banner_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    color_bgr = rgb_to_bgr(subtask_color(subtask))
    cv2.putText(frame, subtask, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_bgr, 1, cv2.LINE_AA)


# ───── COT text panel (richer than robocasa: shows LONG PLAN + MOVEMENT) ────


COT_PANEL_H = 320


def _strip_header(text: str, header: str) -> str:
    """Drop the leading 'HEADER:\n' line from text blocks like 'LONG PLAN:\n...'."""
    if text.startswith(header + ":"):
        text = text[len(header) + 1:].lstrip("\n")
    return text


def make_cot_panel(rec: dict, instruction: str, frame_idx: int, panel_w: int) -> np.ndarray:
    panel = np.ones((COT_PANEL_H, panel_w, 3), dtype=np.uint8) * 248
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(13)
    font_s = load_font(11)
    line_h = 16

    long_plan = _strip_header(rec.get("assistant_plan_level", ""), "LONG PLAN")
    short_plan = _strip_header(rec.get("assistant_short_plan", ""), "SHORT PLAN")
    movement = _strip_header(rec.get("assistant_movement_level", ""), "MOVEMENT")
    subtask = rec.get("subtask", "unknown")
    reason = rec.get("reason", "")
    seg_id = rec.get("segment_id", "?")

    max_chars = max(40, panel_w // 8)
    y = 6

    def block(title: str, content: str, title_color, max_lines: int = 6):
        nonlocal y
        if y >= COT_PANEL_H - line_h:
            return
        draw.text((8, y), title, fill=title_color, font=font)
        y += line_h
        if not content:
            y += 2
            return
        used = 0
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            for ln in textwrap.wrap(raw_line, width=max_chars):
                if used >= max_lines or y >= COT_PANEL_H - line_h:
                    return
                draw.text((16, y), ln, fill=(40, 40, 40), font=font_s)
                y += line_h - 2
                used += 1
        y += 2

    draw.text((8, y), f"Frame {frame_idx}   seg_id={seg_id}",
              fill=(120, 120, 120), font=font_s)
    y += line_h
    block("INSTRUCTION:", instruction, (0, 0, 180), max_lines=2)
    block("LONG PLAN:", long_plan, (170, 0, 0), max_lines=8)
    color_rgb = subtask_color(subtask)
    block(f"SUBTASK: {subtask}", reason, color_rgb, max_lines=4)
    block("MOVEMENT:", movement, (130, 0, 130), max_lines=4)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ───── per-episode rendering ────────────────────────────────────────────────


def render_episode(payload: dict, lerobot_suite_dir: Path, ep_idx: int,
                   output_path: Path, fps: int = 20) -> bool:
    instruction = payload.get("instruction", "")
    frames_data = payload.get("frames", [])
    n_frames = len(frames_data)
    if n_frames == 0:
        return False

    # Load video frames (primary camera). LIBERO MP4s are AV1-encoded — cv2
    # can't decode without hw accel; use the pyav-backed helper instead.
    vp = video_path(lerobot_suite_dir, ep_idx, "observation.images.image")
    if not vp.exists():
        print(f"  [warn] no video at {vp}")
        return False
    rgb_frames = read_video_frames(vp)  # list of HxWx3 uint8 RGB
    if not rgb_frames:
        return False
    # ffmpeg pipe expects bgr24
    raw_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames]

    n = min(len(raw_frames), n_frames)
    h, w = raw_frames[0].shape[:2]

    # Map segment_id -> sequential 1-based index by first-occurrence order
    seg_order: dict = {}
    for rec in frames_data:
        sid = rec.get("segment_id")
        if sid is not None and sid not in seg_order:
            seg_order[sid] = len(seg_order) + 1
    total_segs = max(payload.get("segment_count", len(seg_order)), len(seg_order)) or 1

    # Compose every frame in memory then write via imageio.v3 (uses bundled
    # imageio-ffmpeg). Keeps dimensions even (yuv420p requirement).
    panel_h = COT_PANEL_H
    out_w = w if w % 2 == 0 else w - 1
    out_h_total = h + panel_h
    if out_h_total % 2:
        out_h_total -= 1

    composed_frames = []
    for t in range(n):
        rec = frames_data[t]
        subtask = rec.get("subtask", "unknown")
        seg_idx = seg_order.get(rec.get("segment_id"), 0)

        frame = raw_frames[t].copy()
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        draw_segment_bar(frame, subtask, seg_idx, total_segs, t, n)
        draw_subtask_banner(frame, subtask)

        panel = make_cot_panel(rec, instruction, t, out_w)
        if panel.shape[0] != panel_h:
            panel = cv2.resize(panel, (out_w, panel_h))

        composed = np.vstack([frame[:h, :out_w], panel])
        if composed.shape[:2] != (out_h_total, out_w):
            composed = composed[:out_h_total, :out_w]
        # imageio expects RGB; we built BGR for cv2 drawing — convert back
        composed_frames.append(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(
        str(output_path),
        np.stack(composed_frames, axis=0),
        plugin="pyav",
        codec="libx264",
        fps=fps,
        out_pixel_format="yuv420p",
    )
    return True


# ───── episode discovery ────────────────────────────────────────────────────


def discover_cot_episodes(cot_root: Path, suite: str) -> list[tuple[int, Path]]:
    """Return list of (ep_idx, cot_annotations.json path) for a given suite."""
    base = cot_root / suite / "extras"
    out = []
    if not base.exists():
        return out
    for ep_dir in sorted(base.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        cot_path = ep_dir / "cot_annotations.json"
        if cot_path.exists():
            ep_idx = int(ep_dir.name.split("_")[1])
            out.append((ep_idx, cot_path))
    return out


# ───── main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Visualize LIBERO CoT annotations")
    p.add_argument("--cot_root", required=True,
                   help="Root of cot_annotations (e.g. playground/Datasets/LIBERO_COT)")
    p.add_argument("--data_root", required=True,
                   help="Root of LeRobot LIBERO data (for video MP4s)")
    p.add_argument("--output", required=True, help="Output dir for rendered MP4s")
    p.add_argument("--suite", default="libero_goal", choices=LIBERO_SUITES)
    p.add_argument("--episodes", default="",
                   help="Comma-separated episode indices, e.g. '0,5,12'. "
                        "Overrides --num_episodes when set.")
    p.add_argument("--num_episodes", type=int, default=5,
                   help="Render the first N labeled episodes from --suite")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    cot_root = Path(args.cot_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    sd = suite_dir(data_root, args.suite)

    available = discover_cot_episodes(cot_root, args.suite)
    if not available:
        print(f"[error] no cot_annotations.json under {cot_root / args.suite}")
        return

    # Pick episodes
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        episodes = [(ep, p) for ep, p in available if ep in wanted]
        missing = wanted - {ep for ep, _ in episodes}
        if missing:
            print(f"[warn] requested but not labeled yet: {sorted(missing)}")
    else:
        episodes = available[: max(args.num_episodes, 0)]

    if not episodes:
        print("[error] no episodes selected")
        return

    print(f"Rendering {len(episodes)} episodes from {args.suite}")
    print(f"  cot_root:  {cot_root}")
    print(f"  data_root: {sd}")
    print(f"  output:    {output_dir}\n")

    ok, fail = 0, 0
    for ep_idx, cot_path in tqdm(episodes, desc=args.suite):
        try:
            with open(cot_path) as f:
                payload = json.load(f)
        except Exception as e:
            tqdm.write(f"  [warn] ep{ep_idx}: cannot read {cot_path}: {e}")
            fail += 1
            continue

        out_path = output_dir / f"{args.suite}_ep{ep_idx:04d}.mp4"
        try:
            success = render_episode(payload, sd, ep_idx, out_path, args.fps)
        except Exception as e:
            tqdm.write(f"  [error] ep{ep_idx}: {e}")
            fail += 1
            continue

        if success:
            ok += 1
            tqdm.write(f"  -> {out_path} ({payload.get('num_frames', '?')} frames)")
        else:
            fail += 1

    print(f"\nDone. {ok} rendered, {fail} failed → {output_dir}")


if __name__ == "__main__":
    main()
