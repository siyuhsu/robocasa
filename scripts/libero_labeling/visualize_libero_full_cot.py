"""
Visualize FULL LIBERO CoT annotations (motion + grounding) on episode videos.

Reads cot_annotations.json files that have been backfilled with bbox + gripper_2d
(Phase D.3). Renders an annotated MP4 with:
  - Top: 256x256 agentview video frame, with per-frame overlay:
      * coloured bbox per object (task obj + distractors)
      * red gripper trajectory trail (last 24 frames) + dot at current
      * frame counter
  - Bottom: ~620px CoT text panel showing all the assistant_* fields:
      USER / LONG PLAN / SHORT PLAN / MOVEMENT / POSITION LEVEL / OBJECT

Adapted from robocasa_labeling/create_dataset.py:render_episode_video, with
LIBERO-specific changes:
  * single agentview camera (not 3 stitched)
  * bbox + gripper coords are already in raw-MP4 frame (coord_frame=raw_mp4) —
    no flip transforms needed
  * pyav decode + imageio.v3 write (LIBERO MP4 is AV1; no system ffmpeg)

Usage:
  python scripts/libero_labeling/visualize_libero_full_cot.py \
      --cot_root playground/Datasets/LIBERO_COT \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --output tmp/libero_full_cot_viz \
      --suite libero_goal --episodes 0,1
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import LIBERO_SUITES, read_video_frames, suite_dir, video_path


# ───── colours ──────────────────────────────────────────────────────────────

_PALETTE_RGB = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
]


def color_bgr(idx: int) -> tuple[int, int, int]:
    rgb = _PALETTE_RGB[idx % len(_PALETTE_RGB)]
    return (rgb[2], rgb[1], rgb[0])


def color_rgb(idx: int) -> tuple[int, int, int]:
    return _PALETTE_RGB[idx % len(_PALETTE_RGB)]


def load_font(size: int = 13):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ───── per-frame overlay (bbox + gripper trail) ──────────────────────────────


def draw_traj(frame_bgr: np.ndarray, grippers: list, idx: int, trail_len: int = 24):
    """Draw past trail in red — last `trail_len` frames up to current idx."""
    start = max(0, idx - trail_len)
    pts = []
    for j in range(start, idx + 1):
        pt = grippers[j] if j < len(grippers) else None
        if pt is None or len(pt) < 2:
            continue
        pts.append((int(pt[0]), int(pt[1])))
    for i in range(1, len(pts)):
        cv2.line(frame_bgr, pts[i - 1], pts[i], (0, 0, 255), 1, cv2.LINE_AA)


def draw_future_traj(frame_bgr: np.ndarray, grippers: list, idx: int,
                     horizon: int = 20):
    """Draw future 20-step path in cyan starting from current idx.

    Cyan (BGR (255, 255, 0)) chosen for contrast with the red past-trail.
    """
    end = min(idx + horizon, len(grippers) - 1)
    pts = []
    for j in range(idx, end + 1):
        pt = grippers[j] if j < len(grippers) else None
        if pt is None or len(pt) < 2:
            continue
        pts.append((int(pt[0]), int(pt[1])))
    cyan = (255, 255, 0)
    for i in range(1, len(pts)):
        cv2.line(frame_bgr, pts[i - 1], pts[i], cyan, 1, cv2.LINE_AA)
    # Mark the path endpoint with a small hollow cyan circle so the user can
    # see where the gripper will be in 1 second.
    if len(pts) >= 2:
        ex, ey = pts[-1]
        cv2.circle(frame_bgr, (ex, ey), 3, cyan, 1, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, frame_rec: dict,
                 all_object_names: list[str], gripper_history: list,
                 obj_cat: str | None):
    # Bboxes for all objects (task obj first → red palette[0], distractors → other colours)
    task_bbox = frame_rec.get("task_obj_bbox")
    distractors = frame_rec.get("distractor_bboxes", []) or []
    bboxes_with_idx = []
    if task_bbox is not None:
        bboxes_with_idx.append((0, task_bbox, obj_cat or "task_obj"))
    for di, dbbox in enumerate(distractors):
        name = all_object_names[di + 1] if di + 1 < len(all_object_names) else f"distr_{di}"
        bboxes_with_idx.append((di + 1, dbbox, name))

    for ci, bbox, label in bboxes_with_idx:
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        c = color_bgr(ci)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 4)),
                      (min(x1 + tw + 2, frame_bgr.shape[1] - 1), y1), c, -1)
        cv2.putText(frame_bgr, label, (x1 + 1, max(10, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    # Gripper trajectory: red past trail + cyan future 20-step path + current dot
    cur_idx = frame_rec.get("frame_index", 0)
    draw_traj(frame_bgr, gripper_history, cur_idx, trail_len=24)
    draw_future_traj(frame_bgr, gripper_history, cur_idx, horizon=20)
    cur_pt = gripper_history[cur_idx] if cur_idx < len(gripper_history) else None
    if cur_pt is not None and len(cur_pt) >= 2:
        cv2.circle(frame_bgr, (int(cur_pt[0]), int(cur_pt[1])), 4, (0, 0, 255), -1)
        cv2.circle(frame_bgr, (int(cur_pt[0]), int(cur_pt[1])), 5, (255, 255, 255), 1)


# ───── multi-block CoT panel ────────────────────────────────────────────────

COT_PANEL_H = 620


def make_cot_panel(payload: dict, frame_idx: int, panel_w: int) -> np.ndarray:
    panel = np.ones((COT_PANEL_H, panel_w, 3), dtype=np.uint8) * 248
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(13)
    font_s = load_font(11)
    y, lh = 6, 16
    max_chars = max(40, panel_w // 7)

    frames = payload.get("frames", [])
    rec = frames[frame_idx] if frame_idx < len(frames) else {}

    instr = payload.get("instruction", "")
    long_plan = str(rec.get("assistant_plan_level", "") or "")
    short_plan = str(rec.get("assistant_short_plan", "") or "")
    movement = str(rec.get("assistant_movement_level", "") or "")
    position = str(rec.get("assistant_position_level", "") or "")
    obj_block = str(rec.get("assistant_object_level", "") or "")

    def block(title: str, content: str, title_color, max_lines: int = 999):
        nonlocal y
        if y >= COT_PANEL_H - lh:
            return
        draw.text((8, y), title, fill=title_color, font=font)
        y += lh
        if not content:
            y += 2; return
        used = 0
        for raw in content.splitlines():
            if not raw.strip():
                continue
            for ln in textwrap.wrap(raw, width=max_chars):
                if used >= max_lines or y >= COT_PANEL_H - lh:
                    return
                draw.text((16, y), ln, fill=(40, 40, 40), font=font_s)
                y += lh - 2
                used += 1
        y += 2

    seg_id = rec.get("segment_id", "?")
    subtask = rec.get("subtask", "?")
    draw.text((8, y), f"Frame {frame_idx}   seg={seg_id}   subtask={subtask}",
              fill=(120, 120, 120), font=font_s)
    y += lh
    block("INSTRUCTION:", instr, (0, 0, 180), max_lines=2)
    block("LONG PLAN:", long_plan, (170, 0, 0), max_lines=10)
    block("SHORT PLAN:", short_plan, (0, 120, 0), max_lines=6)
    block("MOVEMENT:", movement, (130, 0, 130), max_lines=6)
    block("POSITION LEVEL:", position, (120, 0, 120), max_lines=8)
    block("OBJECT:", obj_block, (130, 70, 0), max_lines=10)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ───── per-episode rendering ────────────────────────────────────────────────


def render_episode(payload: dict, lerobot_suite_dir: Path, ep_idx: int,
                   output_path: Path, fps: int = 20) -> bool:
    instruction = payload.get("instruction", "")
    frames_data = payload.get("frames", [])
    n_frames = len(frames_data)
    if n_frames == 0:
        return False

    # Verify grounding present
    if "grounding_coord_frame" not in payload:
        print(f"  [warn] ep{ep_idx} missing grounding_coord_frame — viz will lack bbox/gripper")
    elif payload.get("grounding_coord_frame") != "raw_mp4":
        print(f"  [warn] ep{ep_idx} coord_frame={payload.get('grounding_coord_frame')} — bbox may misalign")

    all_object_names = payload.get("all_object_names", [])
    obj_cat = payload.get("obj_cat")

    # Build full gripper history once for trajectory trail
    gripper_history = [rec.get("gripper_2d") for rec in frames_data]

    # Load video frames (agentview)
    vp = video_path(lerobot_suite_dir, ep_idx, "observation.images.image")
    if not vp.exists():
        print(f"  [warn] no video at {vp}")
        return False
    rgb_frames = read_video_frames(vp)
    if not rgb_frames:
        return False
    bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames]

    n = min(len(bgr_frames), n_frames)
    h, w = bgr_frames[0].shape[:2]
    out_w = w if w % 2 == 0 else w - 1
    out_h_total = h + COT_PANEL_H
    if out_h_total % 2:
        out_h_total -= 1

    composed = []
    for t in range(n):
        frame = bgr_frames[t].copy()
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))

        rec = frames_data[t]
        draw_overlay(frame, rec, all_object_names, gripper_history, obj_cat)

        # Top-left frame counter
        cv2.putText(frame, f"f{t}/{n}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)

        panel = make_cot_panel(payload, t, out_w)
        if panel.shape[0] != COT_PANEL_H:
            panel = cv2.resize(panel, (out_w, COT_PANEL_H))

        top = frame[:h, :out_w]
        composed_frame = np.vstack([top, panel])
        if composed_frame.shape[:2] != (out_h_total, out_w):
            composed_frame = composed_frame[:out_h_total, :out_w]
        composed.append(cv2.cvtColor(composed_frame, cv2.COLOR_BGR2RGB))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(
        str(output_path),
        np.stack(composed, axis=0),
        plugin="pyav",
        codec="libx264",
        fps=fps,
        out_pixel_format="yuv420p",
    )
    return True


def discover_cot_episodes(cot_root: Path, suite: str) -> list[tuple[int, Path]]:
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


def main():
    p = argparse.ArgumentParser(description="Visualize FULL LIBERO CoT (motion + grounding)")
    p.add_argument("--cot_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--suite", default="libero_goal", choices=LIBERO_SUITES)
    p.add_argument("--episodes", default="",
                   help="Comma-separated episode indices, e.g. '0,5,12'")
    p.add_argument("--num_episodes", type=int, default=5)
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

    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        episodes = [(ep, p) for ep, p in available if ep in wanted]
        missing = wanted - {ep for ep, _ in episodes}
        if missing:
            print(f"[warn] requested but missing: {sorted(missing)}")
    else:
        episodes = available[: max(args.num_episodes, 0)]

    if not episodes:
        print("[error] no episodes selected")
        return

    print(f"Rendering {len(episodes)} full-CoT episodes from {args.suite}")
    print(f"  cot_root:  {cot_root}")
    print(f"  data_root: {sd}")
    print(f"  output:    {output_dir}\n")

    ok, fail = 0, 0
    for ep_idx, cot_path in tqdm(episodes, desc=args.suite):
        try:
            with open(cot_path) as f:
                payload = json.load(f)
        except Exception as e:
            tqdm.write(f"  [warn] ep{ep_idx}: {e}"); fail += 1; continue
        out_path = output_dir / f"{args.suite}_ep{ep_idx:04d}_full_cot.mp4"
        try:
            success = render_episode(payload, sd, ep_idx, out_path, args.fps)
        except Exception as e:
            tqdm.write(f"  [error] ep{ep_idx}: {e}"); fail += 1; continue
        if success:
            ok += 1
            tqdm.write(f"  -> {out_path} ({payload.get('num_frames', '?')} frames)")
        else:
            fail += 1

    print(f"\nDone. {ok} rendered, {fail} failed → {output_dir}")


if __name__ == "__main__":
    main()
