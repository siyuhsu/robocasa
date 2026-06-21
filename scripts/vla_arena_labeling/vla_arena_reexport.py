#!/usr/bin/env python3
"""Re-export a VLA-Arena lerobot_openpi dataset (PNG-in-parquet, columns
`image`/`wrist_image`/`state`/`actions`) into a LIBERO-keyed LeRobot dataset
(MP4 videos + columns `observation.images.image`/`observation.images.wrist_image`/
`observation.state`/`action`) that the starVLA libero_cot / qwenlap loader ingests.

VLA-Arena state-8 [eef_pos3, axisangle3, gripper_qpos2] and action-7 are ALREADY
bit-identical to the LIBERO layout, so this is a pure repackage (rename cols,
PNG->MP4, key-rename meta). On-disk PNGs are already 180-deg flipped; written as-is.

Usage:
  python vla_arena_reexport.py --src <SRC> --dst <DST> --suite vla_arena_l0 \
      [--fps 10] [--limit N] [--workers 16]
"""
import argparse, io, json, shutil
from pathlib import Path
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
import cv2
from PIL import Image

IMG_KEY_MAP = {"image": "observation.images.image",
               "wrist_image": "observation.images.wrist_image"}
COL_MAP = {"state": "observation.state", "actions": "action"}

MODALITY = {
    "state": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
              "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
              "pad": {"start": 6, "end": 7}, "gripper": {"start": 7, "end": 8}},
    "action": {"x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
               "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
               "gripper": {"start": 6, "end": 7}},
    "video": {"primary_image": {"original_key": "observation.images.image"},
              "wrist_image": {"original_key": "observation.images.wrist_image"}},
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def decode_png(cell):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)


def write_mp4(path: Path, frames, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed: {path}")
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()


def convert_one(ep, src, out, data_path_tmpl, chunk_size, fps):
    chunk = ep // chunk_size
    df = pd.read_parquet(src / data_path_tmpl.format(episode_chunk=chunk, episode_index=ep))
    for src_key, dst_key in IMG_KEY_MAP.items():
        frames = [decode_png(c) for c in df[src_key].tolist()]
        write_mp4(out / f"videos/chunk-{chunk:03d}/{dst_key}/episode_{ep:06d}.mp4", frames, fps)
    keep = {c: df[c] for c in df.columns if c not in IMG_KEY_MAP}  # keeps source `index`
    ndf = pd.DataFrame(keep).rename(columns=COL_MAP)
    dpath = out / data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
    dpath.parent.mkdir(parents=True, exist_ok=True)
    ndf.to_parquet(dpath, index=False)
    return ep, len(ndf)


def transform_info(src_info, fps):
    new_feats = {}
    for k, v in src_info["features"].items():
        if k in IMG_KEY_MAP:
            shp = v.get("shape", [256, 256, 3])
            new_feats[IMG_KEY_MAP[k]] = {
                "dtype": "video", "shape": shp, "names": ["height", "width", "rgb"],
                "info": {"video.height": shp[0], "video.width": shp[1], "video.codec": "mp4v",
                          "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                          "video.fps": fps, "video.channels": 3, "has_audio": False}}
        elif k in COL_MAP:
            nv = dict(v)
            nv["names"] = {"motors": (["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]
                                       if k == "state" else
                                       ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"])}
            new_feats[COL_MAP[k]] = nv
        else:
            new_feats[k] = v
    out = dict(src_info)
    out["features"] = new_feats
    out["fps"] = fps
    out["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    out["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    return out


def rename_stat_keys(stat_obj):
    return {IMG_KEY_MAP.get(k, COL_MAP.get(k, k)): v for k, v in stat_obj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    src = args.src
    out = args.dst / f"{args.suite}_no_noops_1.0.0_lerobot"
    (out / "meta").mkdir(parents=True, exist_ok=True)
    src_info = json.load(open(src / "meta" / "info.json"))
    chunk_size = src_info.get("chunks_size", 1000)
    n_total = src_info["total_episodes"]
    n = n_total if args.limit <= 0 else min(args.limit, n_total)
    dtmpl = src_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    print(f"[reexport] {args.suite}: {n}/{n_total} eps, fps={args.fps}, workers={args.workers} -> {out}", flush=True)

    fn = partial(convert_one, src=src, out=out, data_path_tmpl=dtmpl, chunk_size=chunk_size, fps=args.fps)
    total_frames = 0
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, (ep, nf) in enumerate(pool.imap_unordered(fn, range(n), chunksize=4)):
                total_frames += nf
                if (i + 1) % 200 == 0 or i + 1 == n:
                    print(f"  [{i+1}/{n}] done (last ep{ep} {nf}f)", flush=True)
    else:
        for ep in range(n):
            _, nf = fn(ep); total_frames += nf

    info = transform_info(src_info, args.fps)
    info.update(total_episodes=n, total_frames=total_frames,
                total_videos=n * len(IMG_KEY_MAP), total_chunks=(n - 1) // chunk_size + 1)
    json.dump(info, open(out / "meta" / "info.json", "w"), indent=2)
    json.dump(MODALITY, open(out / "meta" / "modality.json", "w"), indent=2)
    shutil.copy(src / "meta" / "tasks.jsonl", out / "meta" / "tasks.jsonl")
    with open(src / "meta" / "episodes.jsonl") as f, open(out / "meta" / "episodes.jsonl", "w") as g:
        for i, line in enumerate(f):
            if i >= n: break
            g.write(line)
    es = src / "meta" / "episodes_stats.jsonl"
    if es.exists():
        with open(es) as f, open(out / "meta" / "episodes_stats.jsonl", "w") as g:
            for i, line in enumerate(f):
                if i >= n: break
                rec = json.loads(line)
                if "stats" in rec: rec["stats"] = rename_stat_keys(rec["stats"])
                g.write(json.dumps(rec) + "\n")
    print(f"[reexport] DONE {args.suite}: {n} eps, {total_frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
