#!/usr/bin/env python3
"""Per-task full-COT review viz for robocasa_cot_lib256_full (grounding baked into
the cot). Reads cot frames (subtask/reason/gripper_2d/bbox) + the libero256 video,
overlays gripper_2d cross+trail + obj/distractor bbox + subtask/reason/LONG PLAN
panel. One episode per task."""
import json, glob, argparse, textwrap
from pathlib import Path
import numpy as np, cv2, decord

SCALE = 3


def wrap(t, w): return textwrap.wrap(t, w) or [""]


def viz_one(cot_path, video_path, out_path, fps=20):
    c = json.load(open(cot_path)); frames = c["frames"]
    vr = decord.VideoReader(video_path); H0 = vr[0].shape[0]
    H = W = H0 * SCALE; PW = 540
    instr = c.get("instruction", "")
    runs, prev = [], None
    for f in frames:
        if f.get("subtask") != prev: runs.append(f["subtask"]); prev = f["subtask"]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PW, H))
    trail = []
    for i, f in enumerate(frames):
        fi = f.get("frame_index", i)
        img = vr[min(fi, len(vr) - 1)].asnumpy().astype("uint8")
        img = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
        tb = f.get("task_obj_bbox")
        if tb: cv2.rectangle(img, (tb[0]*SCALE, tb[1]*SCALE), (tb[2]*SCALE, tb[3]*SCALE), (0, 255, 0), 2)
        for db in (f.get("distractor_bboxes") or []):
            cv2.rectangle(img, (db[0]*SCALE, db[1]*SCALE), (db[2]*SCALE, db[3]*SCALE), (255, 120, 0), 1)
        g = f.get("gripper_2d")
        if g:
            x, y = int(g[0]*SCALE), int(g[1]*SCALE); trail.append((x, y))
            for px, py in trail[-30:]: cv2.circle(img, (px, py), 2, (0, 180, 255), -1)
            cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        panel = np.zeros((H, PW, 3), "uint8"); yy = [24]
        def put(txt, col=(255, 255, 255), dy=19, sc=0.44):
            for ln in wrap(txt, 46):
                cv2.putText(panel, ln, (8, yy[0]), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1); yy[0] += dy
        put("INSTRUCTION:", (0, 255, 255)); put(instr)
        yy[0] += 6; put("SUBTASK:", (0, 255, 255)); put(f.get("subtask", ""), (0, 255, 0))
        yy[0] += 6; put("REASON:", (0, 255, 255)); put(f.get("reason", "")[:280], (200, 200, 200), 17, 0.4)
        yy[0] += 6; put("LONG PLAN:", (0, 255, 255))
        for j, s in enumerate(runs):
            put(f"{j+1}. {s}", (0, 255, 0) if s == f.get("subtask") else (160, 160, 160), 17, 0.4)
        vw.write(np.hstack([img, panel]))
    vw.release()
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-root", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="*"); ap.add_argument("--ep", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = a.tasks or sorted(d.name for d in Path(a.cot_root).iterdir() if d.is_dir())
    for t in tasks:
        cot = f"{a.cot_root}/{t}/extras/episode_{a.ep:06d}/cot_annotations.json"
        vid = f"{a.data_root}/{t}/videos/chunk-{a.ep//1000:03d}/robot0_agentview_left_image/episode_{a.ep:06d}.mp4"
        if not (Path(cot).exists() and Path(vid).exists()):
            print(f"[skip] {t}", flush=True); continue
        n = viz_one(cot, vid, f"{a.out}/{t}_ep{a.ep:06d}_cot.mp4")
        print(f"[viz] {t}: {n}f", flush=True)


if __name__ == "__main__":
    main()
