#!/usr/bin/env python3
"""Phase 2 (run in starVLA env): build LeRobot parquet + meta from the per-demo
npz dumped by render_robocasa_224.py (Phase 1, which already wrote the MP4s into
out-root/<task>/videos/). Cosmos format: state-9, action-7, 3 cams @224, fps20."""
import json, glob, argparse
from pathlib import Path
import numpy as np, pandas as pd

VIDEO_KEYS = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]
STATE_DIM, ACTION_DIM, ROBOT_TYPE = 9, 7, "panda_robocasa_cosmos"


def feat_stats(a):
    a = np.asarray(a, np.float64)
    if a.ndim == 1: a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def make_info(n_ep, n_frames, n_tasks, fps, res):
    vf = {"dtype": "video", "shape": [res, res, 3], "names": ["height", "width", "channel"], "video_info": {"video.fps": fps}}
    return {"codebase_version": "v2.0", "robot_type": ROBOT_TYPE, "fps": fps, "total_episodes": n_ep,
            "total_frames": n_frames, "total_tasks": n_tasks, "total_videos": n_ep * len(VIDEO_KEYS), "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
                         "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
                         "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
                         "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
                         "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": ["robot"]},
                         "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ["robot"]},
                         **{k: vf for k in VIDEO_KEYS}}}


def make_modality():
    return {"video": {"agentview_left": {"original_key": "robot0_agentview_left_image"},
                      "agentview_right": {"original_key": "robot0_agentview_right_image"},
                      "wrist": {"original_key": "robot0_eye_in_hand_image"}},
            "state": {"gripper_qpos": {"original_key": "observation.state", "start": 0, "end": 2, "dtype": "float32", "absolute": True},
                      "eef_position": {"original_key": "observation.state", "start": 2, "end": 5, "dtype": "float32", "absolute": True},
                      "eef_quaternion": {"original_key": "observation.state", "start": 5, "end": 9, "rotation_type": "quaternion", "dtype": "float32", "absolute": True}},
            "action": {"end_effector_position": {"original_key": "action", "start": 0, "end": 3, "dtype": "float32", "absolute": False},
                       "end_effector_rotation": {"original_key": "action", "start": 3, "end": 6, "rotation_type": "axis_angle", "dtype": "float32", "absolute": False},
                       "gripper_close": {"original_key": "action", "start": 6, "end": 7, "dtype": "float32", "absolute": False}},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}}}


def build_task(task, out_root, npz_root, fps, res):
    out = Path(out_root) / task
    npzs = sorted(glob.glob(str(Path(npz_root) / task / "episode_*.npz")))
    if not npzs: return f"[skip] {task}: no npz"
    task_to_idx = {}; ep_rows = []; stats_rows = []; n_frames = 0
    for npz in npzs:
        ei = int(Path(npz).stem.split("_")[1]); d = np.load(npz, allow_pickle=True)
        state = d["state"].astype(np.float32); action = d["action"].astype(np.float32)
        lang = str(d["lang"]) or task; T = len(action); ch = ei // 1000
        if lang not in task_to_idx: task_to_idx[lang] = len(task_to_idx)
        ti = task_to_idx[lang]
        rows = [{"timestamp": t / fps, "episode_index": ei, "frame_index": t, "task_index": ti,
                 "observation.state": state[t].tolist(), "action": action[t].tolist()} for t in range(T)]
        pq = out / "data" / f"chunk-{ch:03d}" / f"episode_{ei:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(pq, index=False)
        ep_rows.append({"episode_index": ei, "tasks": [lang], "length": T})
        stats_rows.append({"episode_index": ei, "stats": {
            "observation.state": feat_stats(state), "action": feat_stats(action),
            "timestamp": feat_stats(np.arange(T) / fps), "frame_index": feat_stats(np.arange(T)),
            "episode_index": feat_stats(np.full(T, ei)), "task_index": feat_stats(np.full(T, ti))}})
        n_frames += T
    meta = out / "meta"; meta.mkdir(parents=True, exist_ok=True)
    with (meta / "episodes.jsonl").open("w") as f:
        for r in ep_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for t, i in task_to_idx.items(): f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")
    with (meta / "episodes_stats.jsonl").open("w") as f:
        for r in stats_rows: f.write(json.dumps(r) + "\n")
    (meta / "info.json").write_text(json.dumps(make_info(len(ep_rows), n_frames, len(task_to_idx), fps, res), indent=2))
    (meta / "modality.json").write_text(json.dumps(make_modality(), indent=2))
    nmp4 = len(glob.glob(str(out / "videos" / "chunk-000" / VIDEO_KEYS[0] / "*.mp4")))
    return f"[lerobot] {task}: {len(ep_rows)} ep, {n_frames} frames, {len(task_to_idx)} lang, mp4(cam0)={nmp4}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True); ap.add_argument("--npz-root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args()
    for t in a.tasks: print(build_task(t, a.out_root, a.npz_root, a.fps, a.res), flush=True)


if __name__ == "__main__":
    main()
