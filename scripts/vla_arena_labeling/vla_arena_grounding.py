#!/usr/bin/env python3
"""VLA-Arena COT Stage 3 (gripper_2d): project recorded eef_pos -> pixel.

Pure-numpy projection (run in starVLA env) using per-instruction agentview
camera params dumped by vla_arena_dump_cameras.py. gripper_2d is the
load-bearing grounding field for qwenlap; object bboxes are a separate
best-effort pass (replay-based, added later). Output annotations.json:
  {sample_key: {num_frames, camera, coord_frame, gripper_2d: [[u,v]|None,...]}}
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd


def project_to_pixel(eef_pos, cam_pos, cam_xmat, fovy, H, W):
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    diff = np.asarray(eef_pos) - np.asarray(cam_pos)
    c = np.asarray(cam_xmat).T @ diff  # mujoco cam looks along -z
    if c[2] >= 0:
        return None
    u = f * c[0] / -c[2] + W / 2
    v = -f * c[1] / -c[2] + H / 2
    return float(u), float(v)


def to_mp4_frame_pt(pt, W, H):
    if pt is None:
        return None
    return [int(round(W - 1 - pt[0])), int(round(pt[1]))]   # 180deg: only u flips (v pre-flipped)


def list_episodes(lerobot, one_per_task, episodes_arg):
    meta = lerobot / "meta"
    info = json.load(open(meta / "info.json")); chunk = info.get("chunks_size", 1000)
    rows = []
    for l in open(meta / "episodes.jsonl"):
        r = json.loads(l)
        rows.append((r["episode_index"], (r.get("tasks") or [None])[0]))
    if episodes_arg:
        want = set(episodes_arg); rows = [x for x in rows if x[0] in want]
    elif one_per_task:
        seen = set(); out = []
        for ei, instr in rows:
            if instr in seen: continue
            seen.add(instr); out.append((ei, instr))
        rows = out
    return rows, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", type=Path, required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--one-per-task", action="store_true")
    ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args()

    cameras = json.load(open(args.cameras))
    rows, chunk = list_episodes(args.lerobot, args.one_per_task, args.episodes)
    print(f"[info] {args.suite}: grounding {len(rows)} episodes", flush=True)

    def project_all(eef, cam):
        H, W = cam["H"], cam["W"]
        g = [to_mp4_frame_pt(project_to_pixel(eef[t], cam["cam_pos"], cam["cam_xmat"], cam["fovy"], H, W), W, H)
             for t in range(len(eef))]
        inb = sum(1 for p in g if p and 0 <= p[0] < W and 0 <= p[1] < H)
        return g, inb

    done = 0
    for ei, instr in rows:
        cands = cameras.get(instr)
        if not cands:
            print(f"[warn] no camera for {instr!r}", flush=True); continue
        if isinstance(cands, dict):       # back-compat single-camera format
            cands = [cands]
        ch = ei // chunk
        df = pd.read_parquet(args.lerobot / f"data/chunk-{ch:03d}/episode_{ei:06d}.parquet")
        eef = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:, 0:3]
        # pick the candidate scene whose camera maximizes in-bounds projection
        best = None
        for cam in cands:
            g2d, inb = project_all(eef, cam)
            if best is None or inb > best[1]:
                best = (g2d, inb, cam)
        g2d, inb, cam = best
        n = len(g2d)
        rec = {"sample_key": f"{args.suite}|{ei}", "instruction": instr, "num_frames": len(eef),
               "coord_frame": "raw_mp4", "all_object_names": [], "obj_cat": None, "distr_cats": [],
               "cameras": {"agentview": {"gripper_2d": g2d, "all_object_bboxes": [[] for _ in range(n)]}},
               "bddl": cam["bddl"], "n_cands": len(cands)}
        out = args.out_root / f"episode_{ei:06d}" / "annotations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(out, "w"))
        done += 1
        print(f"  ep{ei} {instr[:30]!r} frames={len(eef)} in-bounds={inb}/{len(g2d)} bddl={cam['suite']}/L{cam['level']} ({len(cands)} cand)", flush=True)
    print(f"[done] {args.suite}: {done} episodes grounded -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
