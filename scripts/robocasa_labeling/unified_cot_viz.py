#!/usr/bin/env python3
"""Unified full-COT review viz for libero / VLA-Arena / robocasa.

LIBERO drawing style+logic (palette-coloured bbox per object with labels, red
past-trail + cyan future-trail + current gripper dot) on the video; text panel in
the ROBOCASA layout (right half of the frame): INSTRUCTION / SUBTASK / REASON /
LONG PLAN with the current subtask highlighted. Handles both data layouts:
  --layout suite : data_root/<unit>_no_noops_1.0.0_lerobot, key observation.images.image  (libero, vla_arena)
  --layout task  : data_root/<unit>,                          key robot0_agentview_left_image (robocasa)
One sample per task (first episode of each unique instruction).
"""
import sys, json, argparse, textwrap
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/ssd/sxu/workspace/code/robocasa/scripts/libero_labeling")
from utils_libero import read_video_frames

SCALE = 3
PANEL_W = 560

_PALETTE = [(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),(145,30,180),
            (70,240,240),(240,50,230),(210,245,60),(250,190,212),(0,128,128),(220,190,255),
            (170,110,40),(255,250,200),(128,0,0),(170,255,195)]
def cbgr(i): r=_PALETTE[i%len(_PALETTE)]; return (r[2],r[1],r[0])


def load_font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


# ---- libero-style overlays (drawn at SCALE'd coords) ----
def draw_traj(img, gh, idx, n, col, fut=False):
    rng = range(idx, min(idx+n, len(gh))) if fut else range(max(0, idx-n), idx+1)
    pts = [(int(gh[j][0]), int(gh[j][1])) for j in rng if j < len(gh) and gh[j] and len(gh[j]) >= 2]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], col, 2, cv2.LINE_AA)
    if fut and len(pts) >= 2:
        cv2.circle(img, pts[-1], 5, col, 2, cv2.LINE_AA)


def draw_overlay(img, rec, names, gh, obj_cat, idx):
    boxes = []
    tb = rec.get("task_obj_bbox")
    if tb: boxes.append((0, tb, obj_cat or "task_obj"))
    for di, db in enumerate(rec.get("distractor_bboxes") or []):
        nm = names[di+1] if di+1 < len(names) else f"distr_{di}"
        boxes.append((di+1, db, nm))
    for ci, b, lab in boxes:
        if not b or len(b) != 4: continue
        x1, y1, x2, y2 = [int(v*SCALE) for v in b]; c = cbgr(ci)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1-th-5)), (min(x1+tw+3, img.shape[1]-1), y1), c, -1)
        cv2.putText(img, lab, (x1+1, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    draw_traj(img, gh, idx, 24, (0,0,255))            # past = red
    draw_traj(img, gh, idx, 20, (255,255,0), fut=True)  # future = cyan
    cur = gh[idx] if idx < len(gh) else None
    if cur and len(cur) >= 2:
        cv2.circle(img, (int(cur[0]), int(cur[1])), 6, (0,0,255), -1)
        cv2.circle(img, (int(cur[0]), int(cur[1])), 7, (255,255,255), 1)


# ---- robocasa-layout right panel (PIL) ----
def make_panel(payload, runs, idx, pw, ph):
    rec = payload["frames"][idx]
    img = Image.new("RGB", (pw, ph), (12, 12, 12)); d = ImageDraw.Draw(img)
    F = load_font(15); Fs = load_font(13); Fh = load_font(14)
    y = [12]; mc = max(36, pw//9)
    def block(title, txt, tc, bc=(210,210,210), fnt=Fs, dy=19):
        d.text((10, y[0]), title, fill=tc, font=Fh); y[0] += 21
        for raw in str(txt).splitlines() or [""]:
            for ln in (textwrap.wrap(raw, mc) or [""]):
                d.text((18, y[0]), ln, fill=bc, font=fnt); y[0] += dy
        y[0] += 6
    d.text((10, y[0]), f"Frame {idx}/{len(payload['frames'])}   seg={rec.get('segment_id','?')}",
           fill=(130,130,130), font=Fs); y[0] += 24
    block("INSTRUCTION:", payload.get("instruction",""), (90,180,255))
    block("SUBTASK:", rec.get("subtask",""), (255,235,90), (120,255,120))
    block("REASON:", (rec.get("reason","") or "")[:300], (255,235,90))
    d.text((10, y[0]), "LONG PLAN:", fill=(255,160,80), font=Fh); y[0] += 21
    cur = rec.get("subtask")
    for j, s in enumerate(runs):
        on = (s == cur)
        for k, ln in enumerate(textwrap.wrap(f"{j+1}. {s}", mc) or [""]):
            pre = "" if k else ""
            d.text((18, y[0]), ln, fill=(120,255,120) if on else (150,150,150), font=F if on else Fs)
            y[0] += 20
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def render(payload, vp, out_path, fps=20):
    frames = payload["frames"]
    rgb = read_video_frames(vp)
    if not rgb: return False
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
    n = min(len(bgr), len(frames))
    h, w = bgr[0].shape[:2]; H, W = h*SCALE, w*SCALE
    gh = [[p[0]*SCALE, p[1]*SCALE] if (p and len(p) >= 2) else None for p in (f.get("gripper_2d") for f in frames)]
    names = payload.get("all_object_names", []); obj_cat = payload.get("obj_cat")
    runs, prev = [], None
    for f in frames:
        if f.get("subtask") != prev: runs.append(f["subtask"]); prev = f["subtask"]
    comp = []
    for t in range(n):
        fr = cv2.resize(bgr[t], (W, H), interpolation=cv2.INTER_NEAREST)
        draw_overlay(fr, frames[t], names, gh, obj_cat, t)
        cv2.putText(fr, f"f{t}/{n}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        panel = make_panel(payload, runs, t, PANEL_W, H)
        comp.append(cv2.cvtColor(np.hstack([fr, panel]), cv2.COLOR_BGR2RGB))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(str(out_path), np.stack(comp), plugin="pyav", codec="libx264", fps=fps, out_pixel_format="yuv420p")
    return True


def one_per_task(meta_dir):
    seen = {}
    for l in open(Path(meta_dir) / "episodes.jsonl"):
        r = json.loads(l); i = (r.get("tasks") or [""])[0]
        if i and i not in seen: seen[i] = r["episode_index"]
    return sorted(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-root", required=True); ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--layout", choices=["suite", "task"], required=True)
    ap.add_argument("--units", nargs="+", required=True); ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--episodes", type=int, nargs="*", help="explicit episode indices (overrides default per-task selection)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for u in a.units:
        if a.layout == "suite":
            lr = Path(a.data_root) / f"{u}_no_noops_1.0.0_lerobot"; vkey = "observation.images.image"
            eps = a.episodes if a.episodes else one_per_task(lr / "meta")
        else:
            lr = Path(a.data_root) / u; vkey = "robot0_agentview_left_image"; eps = a.episodes if a.episodes else [0]
        nok = 0
        for ep in eps:
            cot = Path(a.cot_root) / u / "extras" / f"episode_{ep:06d}" / "cot_annotations.json"
            vid = lr / "videos" / f"chunk-{ep//1000:03d}" / vkey / f"episode_{ep:06d}.mp4"
            if not (cot.exists() and vid.exists()): continue
            try:
                if render(json.load(open(cot)), vid, Path(a.out) / f"{u}_ep{ep:04d}_cot.mp4", a.fps): nok += 1
            except Exception as e:
                print(f"  [err] {u} ep{ep}: {repr(e)[:80]}", flush=True)
        print(f"[{u}] {nok}/{len(eps)} videos", flush=True)


if __name__ == "__main__":
    main()
