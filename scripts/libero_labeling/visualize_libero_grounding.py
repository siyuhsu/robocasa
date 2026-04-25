"""
Visualize LIBERO bbox + gripper-2D grounding annotations on episode videos.

Reads annotations.json files produced by extract_libero_grounding.py and
overlays per-frame bbox + gripper dot on the original LeRobot MP4. Does NOT
load CoT annotations (motion-only) — purely a check on the spatial grounding.

For each episode, renders one MP4 per camera (default agentview only).
Different colors:
  - task objects of interest (obj_of_interest) → red
  - distractors                                → orange
  - gripper dot                                 → green

Adapted from robocasa_labeling/visualize_annotations.py.

Usage:
  python scripts/libero_labeling/visualize_libero_grounding.py \
      --grounding_root playground/Datasets/LIBERO_GROUNDING \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --output tmp/libero_grounding_viz \
      --suite libero_goal --episodes 0,5,10,20,30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import LIBERO_SUITES, read_video_frames, suite_dir, video_path


# ───── colours (BGR for OpenCV; matches robocasa visualize_annotations) ─────

COLOR_GRIPPER = (0, 255, 0)         # green
COLOR_TASK_OBJ = (0, 0, 255)        # red
COLOR_DISTRACTOR = (255, 165, 0)    # orange


def draw_gripper(frame, pt, color=COLOR_GRIPPER, radius=4):
    if pt is None or len(pt) < 2:
        return
    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
    cv2.circle(frame, (int(pt[0]), int(pt[1])), radius + 1, (255, 255, 255), 1)


def draw_bbox(frame, bbox, label, color, thickness=2):
    if bbox is None or len(bbox) != 4:
        return
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (min(x1 + tw + 2, frame.shape[1] - 1), y1), color, -1)
    cv2.putText(frame, label, (x1 + 1, max(10, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame_info(frame, frame_idx, total_frames, instruction):
    h, w = frame.shape[:2]
    bar_h = 16
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)
    prog = int(w * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(frame, (0, 0), (prog, bar_h), (80, 200, 80), -1)
    cv2.putText(frame, f"f{frame_idx}/{total_frames}",
                (4, bar_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    # Bottom: instruction
    band_h = 18
    cv2.rectangle(frame, (0, h - band_h), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, instruction[:60], (4, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)


# ───── per-episode rendering ────────────────────────────────────────────────


def render_episode(grounding: dict, lerobot_suite_dir: Path, ep_idx: int,
                   camera: str, output_path: Path, fps: int = 20) -> bool:
    instruction = grounding.get("instruction", "")
    cam_data = grounding.get("cameras", {}).get(camera)
    if cam_data is None:
        print(f"  [warn] camera {camera} missing from grounding for ep{ep_idx}")
        return False

    gripper_2d = cam_data.get("gripper_2d", [])
    all_object_bboxes = cam_data.get("all_object_bboxes", [])
    all_object_names = grounding.get("all_object_names", [])
    obj_of_interest = grounding.get("distr_cats", [])  # used for distractor naming

    # Determine which objects are task vs distractor.
    # extract_libero_grounding.py orders all_object_names = [task...] + [distractor...]
    # with obj_cat = first task obj. Use len(obj_of_interest)+1 = task count.
    # Simpler: use env.obj_of_interest if stored; otherwise treat first as task obj
    # and the rest as distractors. obj_cat field gives the first task obj name.
    obj_cat = grounding.get("obj_cat")
    distr_cats = grounding.get("distr_cats", [])
    n_task_objs = 1 + 0  # default: at least 1 task obj is at index 0
    # Recover the actual task-obj count by checking env's obj_of_interest list,
    # which was stored as the prefix of all_object_names.
    # The annotations.json structure stores `obj_cat` (first task) and
    # `distr_cats` (everything else *including* additional task objs because
    # we couldn't distinguish at extraction time). For visualisation, treat
    # only obj_cat as task object; rest as distractors.

    # Decide camera key for video MP4
    camera_to_video_key = {
        "agentview": "observation.images.image",
        "robot0_eye_in_hand": "observation.images.wrist_image",
    }
    video_key = camera_to_video_key.get(camera, f"observation.images.{camera}")

    vp = video_path(lerobot_suite_dir, ep_idx, video_key)
    if not vp.exists():
        print(f"  [warn] no video at {vp}")
        return False
    rgb_frames = read_video_frames(vp)
    if not rgb_frames:
        return False
    # LIBERO MP4 is stored in raw MuJoCo orientation, but eval_libero.py applies
    # [::-1, ::-1] (180° flip) before sending to the model — meaning the
    # simulator-native segmentation/bbox coords correspond to the flipped frame.
    # We flip the video to match so the bbox overlays land correctly.
    bgr_frames = [cv2.cvtColor(f[::-1, ::-1], cv2.COLOR_RGB2BGR) for f in rgb_frames]

    n = min(len(bgr_frames), len(gripper_2d), len(all_object_bboxes))
    h, w = bgr_frames[0].shape[:2]
    out_w = w if w % 2 == 0 else w - 1
    out_h = h if h % 2 == 0 else h - 1

    composed = []
    for t in range(n):
        frame = bgr_frames[t].copy()

        # Per-frame bboxes for all known objects
        frame_boxes = all_object_bboxes[t] if t < len(all_object_bboxes) else []
        for i, bbox in enumerate(frame_boxes):
            if bbox is None:
                continue
            name = all_object_names[i] if i < len(all_object_names) else f"obj_{i}"
            color = COLOR_TASK_OBJ if name == obj_cat else COLOR_DISTRACTOR
            draw_bbox(frame, bbox, name, color)

        # Gripper 2D
        draw_gripper(frame, gripper_2d[t] if t < len(gripper_2d) else None)

        # HUD bars
        draw_frame_info(frame, t, n, instruction)

        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = frame[:out_h, :out_w]
        composed.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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


# ───── episode discovery ────────────────────────────────────────────────────


def discover_grounding_episodes(grounding_root: Path, suite: str) -> list[tuple[int, Path]]:
    base = grounding_root / suite / "extras"
    out = []
    if not base.exists():
        return out
    for ep_dir in sorted(base.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        ann_path = ep_dir / "annotations.json"
        if ann_path.exists():
            ep_idx = int(ep_dir.name.split("_")[1])
            out.append((ep_idx, ann_path))
    return out


# ───── main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Visualize LIBERO bbox + gripper-2D grounding")
    p.add_argument("--grounding_root", required=True,
                   help="Root of LIBERO_GROUNDING annotations")
    p.add_argument("--data_root", required=True,
                   help="Root of LEROBOT_LIBERO_DATA (for source videos)")
    p.add_argument("--output", required=True, help="Output dir for rendered MP4s")
    p.add_argument("--suite", default="libero_goal", choices=LIBERO_SUITES)
    p.add_argument("--episodes", default="",
                   help="Comma-separated episode indices, e.g. '0,5,12'")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--camera", default="agentview",
                   choices=["agentview", "robot0_eye_in_hand"])
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    grounding_root = Path(args.grounding_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output).resolve()
    sd = suite_dir(data_root, args.suite)

    available = discover_grounding_episodes(grounding_root, args.suite)
    if not available:
        print(f"[error] no annotations.json under {grounding_root / args.suite}")
        return

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

    print(f"Rendering {len(episodes)} episodes, camera={args.camera}")
    print(f"  grounding_root: {grounding_root}")
    print(f"  data_root:      {sd}")
    print(f"  output:         {output_dir}\n")

    ok, fail = 0, 0
    for ep_idx, ann_path in tqdm(episodes, desc=args.suite):
        try:
            with open(ann_path) as f:
                grounding = json.load(f)
        except Exception as e:
            tqdm.write(f"  [warn] ep{ep_idx}: cannot read {ann_path}: {e}")
            fail += 1; continue

        out_path = output_dir / f"{args.suite}_ep{ep_idx:04d}_{args.camera}.mp4"
        try:
            success = render_episode(grounding, sd, ep_idx, args.camera, out_path, args.fps)
        except Exception as e:
            tqdm.write(f"  [error] ep{ep_idx}: {e}")
            fail += 1; continue

        if success:
            ok += 1
            tqdm.write(f"  -> {out_path} ({grounding.get('num_frames', '?')} frames)")
        else:
            fail += 1

    print(f"\nDone. {ok} rendered, {fail} failed → {output_dir}")


if __name__ == "__main__":
    main()
