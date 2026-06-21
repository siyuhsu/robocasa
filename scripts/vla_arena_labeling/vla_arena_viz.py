#!/usr/bin/env python3
"""VLA-Arena COT pilot visualization: overlay subtask label + gripper_2d on MP4.

Per episode, renders a review video: the displayed agentview frames with the
current subtask text (from Stage-2 subtasks.json frame_annotations) and the
projected gripper_2d point + recent trail (from Stage-3 grounding). Lets the
user judge subtask + grounding quality before large-scale annotation.
"""
import argparse, json
from pathlib import Path
import numpy as np, cv2


def read_mp4(path):
    cap = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)  # keep BGR for cv2 draw/write
    cap.release(); return fr


def put_text(img, text, y, color=(255, 255, 255)):
    for x0, y0, c in [(5, y, (0, 0, 0)), (4, y - 1, color)]:
        cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2 if c == (0, 0, 0) else 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", type=Path, required=True)
    ap.add_argument("--subtasks", type=Path, required=True)
    ap.add_argument("--grounding-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    sub = json.load(open(args.subtasks))
    info = json.load(open(args.lerobot / "meta" / "info.json")); chunk = info.get("chunks_size", 1000)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    S = args.scale
    for ei in args.episodes:
        ch = ei // chunk
        frames = read_mp4(args.lerobot / f"videos/chunk-{ch:03d}/observation.images.image/episode_{ei:06d}.mp4")
        sk = f"{args.suite}|{ei}"
        fa = sub.get(sk, {}).get("frame_annotations", [])
        gnd = json.load(open(args.grounding_root / f"episode_{ei:06d}" / "annotations.json"))
        g2d = gnd.get("gripper_2d", [])
        instr = gnd.get("instruction", "")
        n = min(len(frames), len(fa), len(g2d)) if fa and g2d else len(frames)
        H, W = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(args.out_dir / f"{args.suite}_episode_{ei:06d}_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (W * S, H * S))
        for t in range(n):
            img = cv2.resize(frames[t], (W * S, H * S), interpolation=cv2.INTER_NEAREST)
            # gripper trail (last 12) + current point
            for k in range(max(0, t - 12), t + 1):
                p = g2d[k] if k < len(g2d) else None
                if p is None: continue
                x, y = int(p[0] * S), int(p[1] * S)
                age = t - k
                cv2.circle(img, (x, y), max(2, 6 - age // 2), (0, 200, 255), -1)
            if t < len(g2d) and g2d[t]:
                x, y = int(g2d[t][0] * S), int(g2d[t][1] * S)
                cv2.circle(img, (x, y), 9, (0, 0, 255), 2)
                cv2.drawMarker(img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
            subtask = fa[t][0] if t < len(fa) and fa[t] else ""
            reason = fa[t][1] if t < len(fa) and fa[t] and len(fa[t]) > 1 else ""
            put_text(img, f"[{t}/{n}] {instr[:52]}", 18)
            put_text(img, f"subtask: {subtask}", 40, (0, 255, 120))
            put_text(img, f"gripper_2d: {g2d[t] if t < len(g2d) else None}", 60, (0, 200, 255))
            # reasoning, wrapped to a few lines
            words = reason.split(); line = ""; ry = 82
            for w in words:
                if len(line) + len(w) + 1 > 64:
                    put_text(img, ("reason: " + line) if ry == 82 else ("        " + line), ry, (200, 200, 255))
                    ry += 18; line = w
                    if ry > 82 + 18 * 3: break
                else:
                    line = (line + " " + w).strip()
            if line and ry <= 82 + 18 * 3:
                put_text(img, ("reason: " + line) if ry == 82 else ("        " + line), ry, (200, 200, 255))
            vw.write(img)
        vw.release()
        print(f"  wrote {args.suite}_episode_{ei:06d}_review.mp4 ({n} frames)", flush=True)
    print(f"[done] videos -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
