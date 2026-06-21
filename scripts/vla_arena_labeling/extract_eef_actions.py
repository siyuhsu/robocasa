#!/usr/bin/env python3
"""Pre-dump per-episode eef_pos(0:3) + 7-D actions from the lerobot parquet
(starVLA env has pandas; the VLA-Arena sim venv does not). The bbox replay
script reads these .npz to (a) match the init state and (b) replay actions.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd


def list_one_per_task(lerobot):
    info = json.load(open(lerobot / "meta" / "info.json")); chunk = info.get("chunks_size", 1000)
    seen = set(); rows = []
    for l in open(lerobot / "meta" / "episodes.jsonl"):
        r = json.loads(l); i = (r.get("tasks") or [None])[0]
        if i and i not in seen:
            seen.add(i); rows.append((r["episode_index"], i))
    return rows, chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--one-per-task", action="store_true")
    ap.add_argument("--all", action="store_true", help="all episodes (not one-per-task)")
    ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args()
    import json as _j
    chunk = _j.load(open(args.lerobot/"meta"/"info.json")).get("chunks_size",1000)
    if args.all:
        rows=[]
        for l in open(args.lerobot/"meta"/"episodes.jsonl"):
            r=_j.loads(l); rows.append((r["episode_index"], (r.get("tasks") or [None])[0]))
    else:
        rows, chunk = list_one_per_task(args.lerobot)
    if args.episodes:
        want = set(args.episodes); rows = [(e, i) for e, i in rows if e in want] or [(e, None) for e in args.episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    for ei, instr in rows:
        ch = ei // chunk
        df = pd.read_parquet(args.lerobot / f"data/chunk-{ch:03d}/episode_{ei:06d}.parquet")
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        np.savez(args.out / f"episode_{ei:06d}.npz", eef=state[:, 0:3], actions=action,
                 instruction=np.array(instr if instr else "", dtype=object))
    print(f"[eef] dumped {len(rows)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
