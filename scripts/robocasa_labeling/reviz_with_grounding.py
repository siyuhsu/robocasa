#!/usr/bin/env python3
"""Re-render the robocasa COT review videos WITH gripper_2d overlay (cross +
short trail) on top of the existing cot subtask/reason/long-plan panel. Reads
the already-produced cot + grounding.json + lerobot video (no re-labeling)."""
import json, glob, argparse
from pathlib import Path
import numpy as np, cv2

FPS = 20
VIDEO_KEY = "robot0_agentview_left_image"


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release(); return out


def wrap(text, width, font, sc, th):
    words = (text or "").split(); lines = []; cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if cv2.getTextSize(t, font, sc, th)[0][0] > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    if cur: lines.append(cur)
    return lines


def reviz(task, cot_root, data_root, out_dir, scale=3):
    cotp = Path(cot_root) / task / "extras" / "episode_000000" / "cot_annotations.json"
    gp = Path(cot_root) / "grounding" / task / "grounding.json"
    vp = Path(data_root) / task / "videos" / "chunk-000" / VIDEO_KEY / "episode_000000.mp4"
    if not (cotp.exists() and vp.exists()):
        return f"[skip] {task}"
    cot = json.load(open(cotp)); fr = cot["frames"]
    grec = json.load(open(gp)) if gp.exists() else {}
    g2d = grec.get("gripper_2d")
    bboxes = grec.get("bboxes"); manip = grec.get("manipulated")
    frames = read_frames(vp)
    H = W = frames[0].shape[0] * scale; sc_ratio = scale
    panelW = 360; font = cv2.FONT_HERSHEY_SIMPLEX
    long_plan = fr[len(fr) // 2].get("assistant_plan_level", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / f"{task}_ep000000_cot.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W + panelW, H))
    trail = []
    for i, frame in enumerate(frames):
        if i >= len(fr): break
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
        if bboxes and i < len(bboxes) and bboxes[i]:
            for o, b in bboxes[i].items():
                col = (0, 255, 0) if o == manip else (255, 120, 0)
                cv2.rectangle(img, (int(b[0] * sc_ratio), int(b[1] * sc_ratio)),
                              (int(b[2] * sc_ratio), int(b[3] * sc_ratio)), col, 2)
        if g2d and i < len(g2d) and g2d[i]:
            x, y = int(round(g2d[i][0] * sc_ratio)), int(round(g2d[i][1] * sc_ratio))
            trail.append((x, y)); trail[:] = trail[-12:]
            for j, (tx, ty) in enumerate(trail):
                cv2.circle(img, (tx, ty), 2, (0, 200, 255), -1)
            cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        panel = np.zeros((H, panelW, 3), np.uint8); yy = [24]
        def put(t, col=(255, 255, 255), s=0.46, th=1, dy=19):
            for ln in wrap(t, panelW - 16, font, s, th):
                cv2.putText(panel, ln, (8, yy[0]), font, s, col, th, cv2.LINE_AA); yy[0] += dy
        put(f"frame {i} seg={fr[i].get('segment_id')}", (160, 160, 160), 0.42)
        put("INSTRUCTION:", (120, 200, 255), 0.42); put(cot.get("instruction", "")[:80], (200, 230, 255), 0.42)
        yy[0] += 5; put("SUBTASK (merged):", (120, 255, 160), 0.48); put(fr[i].get("subtask", ""), (180, 255, 200), 0.48)
        yy[0] += 3; put("REASON (per-segment):", (200, 200, 120), 0.4); put(fr[i].get("reason", ""), (220, 220, 180), 0.4)
        yy[0] += 6; put("LONG PLAN:", (255, 180, 120), 0.4)
        for ln in long_plan.replace("LONG PLAN:", "").strip().split("\n"):
            put(ln.strip(), (230, 200, 170), 0.38, dy=15)
        vw.write(cv2.cvtColor(np.hstack([img.astype(np.uint8), panel]), cv2.COLOR_RGB2BGR))
    vw.release()
    return f"[reviz] {task}: {len(frames)}f gripper_2d={'yes' if g2d else 'no'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-root", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_cot_sample")
    ap.add_argument("--data-root", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_v01_cosmos_lerobot")
    ap.add_argument("--out", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_cot_sample/viz_grounded")
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args()
    tasks = a.tasks or sorted(d.name for d in Path(a.cot_root).iterdir() if d.is_dir() and d.name not in ("viz", "viz_grounded", "grounding"))
    for t in tasks:
        print(reviz(t, a.cot_root, a.data_root, Path(a.out)), flush=True)


if __name__ == "__main__":
    main()
