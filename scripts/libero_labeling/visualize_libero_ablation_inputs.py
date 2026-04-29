"""
Phase E.1.6 ablation input visualizer.

Renders, for each ablation variant, the EXACT input the LowLevel sees during
training:
  * the third-person frame (with or without bbox/gripper overlay, gated by
    `use_visual_grounding`)
  * a text panel containing only the prompt segments enabled by the variant's
    `use_X` flags.

This mirrors `HiCoVLA_LowLevel._prepare_batch` for the corresponding yaml so
viewers can sanity-check what each ablation actually trains on.

Usage:
  python scripts/libero_labeling/visualize_libero_ablation_inputs.py \
      --cot_root playground/Datasets/LIBERO_COT \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --output tmp/ablation_input_viz \
      --suite libero_goal --episodes 0,1 \
      --variants visual_only plan_only long_plan_only subtask_only \
                 steering_long_3d_only steering_short_3d_only steering_3d \
                 steering_2d full baseline

If --variants omitted: all 10 variants are rendered (8 ablations + full + baseline reference).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import LIBERO_SUITES, read_video_frames, suite_dir, video_path  # noqa: E402

# ───── ablation variant definitions ──────────────────────────────────────────
# Mirror exactly the use_X gates in starvla_hicovala_lowlevel_libero_*.yaml +
# HiCoVLA_LowLevel._prepare_batch (defaults: visual+plan+subtask+3D on, 2D off).

ABLATIONS: dict[str, dict] = {
    "visual_only":            dict(visual=True,  long_plan=False, subtask=False, short_3d=False, long_3d=False, short_2d=False, long_2d=False),
    "plan_only":              dict(visual=False, long_plan=True,  subtask=True,  short_3d=False, long_3d=False, short_2d=False, long_2d=False),
    "long_plan_only":         dict(visual=False, long_plan=True,  subtask=False, short_3d=False, long_3d=False, short_2d=False, long_2d=False),
    "subtask_only":           dict(visual=False, long_plan=False, subtask=True,  short_3d=False, long_3d=False, short_2d=False, long_2d=False),
    "steering_long_3d_only":  dict(visual=False, long_plan=False, subtask=False, short_3d=False, long_3d=True,  short_2d=False, long_2d=False),
    "steering_short_3d_only": dict(visual=False, long_plan=False, subtask=False, short_3d=True,  long_3d=False, short_2d=False, long_2d=False),
    "steering_3d":            dict(visual=False, long_plan=False, subtask=False, short_3d=True,  long_3d=True,  short_2d=False, long_2d=False),
    "steering_2d":            dict(visual=False, long_plan=False, subtask=False, short_3d=False, long_3d=False, short_2d=True,  long_2d=True),
    "full":                   dict(visual=True,  long_plan=True,  subtask=True,  short_3d=True,  long_3d=True,  short_2d=False, long_2d=False),
    "baseline":               dict(visual=False, long_plan=False, subtask=False, short_3d=False, long_3d=False, short_2d=False, long_2d=False),
}


# ───── shared helpers (palette, font, line splitting) ───────────────────────

_PALETTE_RGB = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
]


def color_bgr(idx: int):
    r, g, b = _PALETTE_RGB[idx % len(_PALETTE_RGB)]
    return b, g, r


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


def split_movement_block(text: str) -> tuple[str, str]:
    cur = next20 = ""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("Current Movement:"):
            cur = s.split(":", 1)[1].strip()
        elif s.startswith("Next 20 Movement:"):
            next20 = s.split(":", 1)[1].strip()
    return cur, next20


def split_position_block(text: str) -> tuple[str, str]:
    nx = nx20 = ""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("NEXT GRIPPER:"):
            nx = s.split(":", 1)[1].strip()
        elif s.startswith("NEXT 20 GRIPPER PATH:"):
            nx20 = s.split(":", 1)[1].strip()
    return nx, nx20


# ───── per-frame visual overlay (visual_grounding axis) ─────────────────────


def draw_traj(frame_bgr: np.ndarray, grippers: list, idx: int, trail_len: int = 24):
    start = max(0, idx - trail_len)
    pts = []
    for j in range(start, idx + 1):
        pt = grippers[j] if j < len(grippers) else None
        if pt is None or len(pt) < 2:
            continue
        pts.append((int(pt[0]), int(pt[1])))
    for i in range(1, len(pts)):
        cv2.line(frame_bgr, pts[i - 1], pts[i], (0, 0, 255), 1, cv2.LINE_AA)


def draw_future_traj(frame_bgr: np.ndarray, grippers: list, idx: int, horizon: int = 20):
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
    if len(pts) >= 2:
        cv2.circle(frame_bgr, pts[-1], 3, cyan, 1, cv2.LINE_AA)


def draw_overlay(frame_bgr: np.ndarray, frame_rec: dict,
                 all_object_names: list[str], gripper_history: list,
                 obj_cat: str | None):
    bboxes_with_idx = []
    if frame_rec.get("task_obj_bbox") is not None:
        bboxes_with_idx.append((0, frame_rec["task_obj_bbox"], obj_cat or "task_obj"))
    for di, dbbox in enumerate(frame_rec.get("distractor_bboxes") or []):
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
    cur_idx = frame_rec.get("frame_index", 0)
    draw_traj(frame_bgr, gripper_history, cur_idx, trail_len=24)
    draw_future_traj(frame_bgr, gripper_history, cur_idx, horizon=20)
    cur_pt = gripper_history[cur_idx] if cur_idx < len(gripper_history) else None
    if cur_pt is not None and len(cur_pt) >= 2:
        cv2.circle(frame_bgr, (int(cur_pt[0]), int(cur_pt[1])), 4, (0, 0, 255), -1)
        cv2.circle(frame_bgr, (int(cur_pt[0]), int(cur_pt[1])), 5, (255, 255, 255), 1)


# ───── per-variant text panel ────────────────────────────────────────────────

PANEL_H = 480
HEADER_FILL = {
    "INSTRUCTION":               (0, 0, 180),
    "LONG PLAN":                 (170, 0, 0),
    "CURRENT SUBTASK":           (0, 120, 0),
    "3D gripper movement now":   (130, 0, 130),
    "3D gripper movement next 20": (130, 0, 130),
    "2D gripper movement now":   (0, 110, 130),
    "2D gripper movement next 20": (0, 110, 130),
    "VISUAL GROUNDING":          (200, 80, 0),
}


def render_panel(payload: dict, frame_idx: int, panel_w: int,
                 variant_name: str, flags: dict) -> np.ndarray:
    panel = np.ones((PANEL_H, panel_w, 3), dtype=np.uint8) * 248
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(13)
    font_s = load_font(11)
    y, lh = 6, 16
    max_chars = max(40, panel_w // 7)

    frames_data = payload.get("frames", [])
    rec = frames_data[frame_idx] if frame_idx < len(frames_data) else {}

    instr = payload.get("instruction", "")
    long_plan = re.sub(r"^LONG PLAN:\s*\n?", "",
                       str(rec.get("assistant_plan_level", "") or ""))
    short_plan_raw = str(rec.get("assistant_short_plan", "") or "")
    short_plan_match = re.search(r"Subtask:\s*(.+)", short_plan_raw)
    subtask = short_plan_match.group(1).strip() if short_plan_match else ""
    short3, long3 = split_movement_block(rec.get("assistant_movement_level", ""))
    short2, long2 = split_position_block(rec.get("assistant_position_level", ""))

    def block(title: str, content: str, max_lines: int = 999):
        nonlocal y
        if y >= PANEL_H - lh:
            return
        color = HEADER_FILL.get(title, (50, 50, 50))
        draw.text((8, y), title + ":", fill=color, font=font)
        y += lh
        if not content:
            draw.text((16, y), "(none)", fill=(150, 150, 150), font=font_s)
            y += lh
            return
        used = 0
        for raw in content.splitlines():
            if not raw.strip():
                continue
            for ln in textwrap.wrap(raw, width=max_chars):
                if used >= max_lines or y >= PANEL_H - lh:
                    return
                draw.text((16, y), ln, fill=(40, 40, 40), font=font_s)
                y += lh - 2
                used += 1
        y += 2

    # Header banner: variant name + frame counter + active flag set
    header = f"variant={variant_name}    frame={frame_idx}/{len(frames_data)}"
    draw.text((8, y), header, fill=(0, 0, 0), font=font)
    y += lh
    flag_summary = " ".join(k for k, v in flags.items() if v) or "(all OFF — baseline)"
    draw.text((8, y), f"flags ON: {flag_summary}", fill=(80, 80, 80), font=font_s)
    y += lh + 4

    # Always show INSTRUCTION (passed in plain in baseline_mode and full mode)
    block("INSTRUCTION", instr, max_lines=2)

    # Visual grounding marker (for visibility in panel even though the actual
    # overlay is on the image side)
    if flags["visual"]:
        block("VISUAL GROUNDING", "ON — bbox + future gripper path drawn on third-person frame",
              max_lines=2)

    if flags["long_plan"]:
        block("LONG PLAN", long_plan, max_lines=10)
    if flags["subtask"]:
        block("CURRENT SUBTASK", subtask or short_plan_raw, max_lines=4)
    if flags["short_3d"]:
        block("3D gripper movement now", short3, max_lines=2)
    if flags["long_3d"]:
        block("3D gripper movement next 20", long3, max_lines=2)
    if flags["short_2d"]:
        block("2D gripper movement now", short2, max_lines=2)
    if flags["long_2d"]:
        block("2D gripper movement next 20", long2, max_lines=4)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ───── per-episode rendering ────────────────────────────────────────────────


def render_episode(payload: dict, lerobot_suite_dir: Path, ep_idx: int,
                   variant_name: str, flags: dict, output_path: Path,
                   fps: int = 20) -> bool:
    frames_data = payload.get("frames", [])
    n_frames = len(frames_data)
    if n_frames == 0:
        return False

    all_object_names = payload.get("all_object_names", [])
    obj_cat = payload.get("obj_cat")
    gripper_history = [rec.get("gripper_2d") for rec in frames_data]

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
    out_h = h + PANEL_H
    if out_h % 2:
        out_h -= 1

    composed = []
    for t in range(n):
        frame = bgr_frames[t].copy()
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        # Visual grounding overlay only when the variant has it ON
        if flags["visual"]:
            draw_overlay(frame, frames_data[t], all_object_names,
                         gripper_history, obj_cat)
        cv2.putText(frame, f"f{t}/{n}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panel = render_panel(payload, t, out_w, variant_name, flags)
        if panel.shape[0] != PANEL_H:
            panel = cv2.resize(panel, (out_w, PANEL_H))
        top = frame[:h, :out_w]
        composed_frame = np.vstack([top, panel])
        if composed_frame.shape[:2] != (out_h, out_w):
            composed_frame = composed_frame[:out_h, :out_w]
        composed.append(cv2.cvtColor(composed_frame, cv2.COLOR_BGR2RGB))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        str(output_path),
        np.stack(composed, axis=0),
        plugin="pyav",
        codec="libx264",
        fps=fps,
        out_pixel_format="yuv420p",
    )
    return True


# ───── episode discovery + main ──────────────────────────────────────────────


def discover_cot_episodes(cot_root: Path, suite: str) -> list[tuple[int, Path]]:
    base = cot_root / suite / "extras"
    if not base.exists():
        return []
    out = []
    for ep_dir in sorted(base.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        cot_path = ep_dir / "cot_annotations.json"
        if cot_path.exists():
            ep_idx = int(ep_dir.name.split("_")[1])
            out.append((ep_idx, cot_path))
    return out


def main():
    p = argparse.ArgumentParser(
        description="Visualize Phase E.1.6 ablation INPUTS (image overlay + text panel)"
    )
    p.add_argument("--cot_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--suite", default="libero_goal", choices=LIBERO_SUITES)
    p.add_argument("--episodes", default="",
                   help="Comma-separated episode indices, e.g. '0,5'")
    p.add_argument("--num_episodes", type=int, default=2,
                   help="If --episodes empty, render the first N available")
    p.add_argument("--variants", nargs="+", default=list(ABLATIONS.keys()),
                   help=f"Subset of: {sorted(ABLATIONS.keys())}")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    bad = [v for v in args.variants if v not in ABLATIONS]
    if bad:
        sys.exit(f"unknown variants: {bad}; valid={list(ABLATIONS.keys())}")

    cot_root = Path(args.cot_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    sd = suite_dir(data_root, args.suite)

    available = discover_cot_episodes(cot_root, args.suite)
    if not available:
        sys.exit(f"[error] no cot_annotations.json under {cot_root / args.suite}")

    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        episodes = [(ep, p) for ep, p in available if ep in wanted]
    else:
        episodes = available[: max(args.num_episodes, 0)]

    if not episodes:
        sys.exit("[error] no episodes selected")

    total = len(episodes) * len(args.variants)
    print(f"Rendering {total} videos: {len(episodes)} episodes × {len(args.variants)} variants")
    print(f"  cot_root  : {cot_root}")
    print(f"  data_root : {sd}")
    print(f"  output    : {output_dir}\n")

    pbar = tqdm(total=total, ncols=100)
    for ep_idx, cot_path in episodes:
        try:
            payload = json.loads(cot_path.read_text())
        except Exception as e:
            print(f"  [warn] ep{ep_idx} JSON load fail: {e}")
            continue
        for variant in args.variants:
            flags = ABLATIONS[variant]
            out_p = output_dir / variant / f"{args.suite}_episode_{ep_idx:06d}.mp4"
            try:
                ok = render_episode(payload, sd, ep_idx, variant, flags, out_p, fps=args.fps)
                if ok:
                    pbar.set_postfix_str(f"{variant} ep{ep_idx}")
                else:
                    pbar.write(f"  [warn] {variant} ep{ep_idx}: render skipped")
            except Exception as e:
                pbar.write(f"  [error] {variant} ep{ep_idx}: {e}")
            pbar.update(1)
    pbar.close()
    print(f"\nDone. Videos under {output_dir}/<variant>/")


if __name__ == "__main__":
    main()
