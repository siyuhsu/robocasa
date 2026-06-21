#!/usr/bin/env python3
"""Convert RoboCasa v0.1 robomimic HDF5 -> LeRobot v2, in the NVIDIA Cosmos-Policy
action/state format (for fair comparison with Cosmos-Policy-RoboCasa).

Cosmos format (verified from robocasa_dataset_statistics.json + dataset card):
  - action  7-D = raw actions[:, 0:7]  (eef Δpos3 + Δrot_axisangle3 + gripper1)
                  i.e. drop base_motion(4) + control_mode(1) from the 12-D action.
  - state   9-D = robot0_gripper_qpos(2) + robot0_eef_pos(3, world) + robot0_eef_quat(4)
  - 3 cams: robot0_agentview_left/right_image + robot0_eye_in_hand_image (128, native)

Per task we MERGE human (all demos) + MimicGen (mask, default 300_demos) into one
LeRobot dataset. No-op frames are dropped (consistent with the 'no_noops' convention).
Outputs per-task LeRobot dirs for easy mixing with the LIBERO/VLA-Arena datasets.
"""
from __future__ import annotations
import argparse, json
from collections import OrderedDict
from pathlib import Path
from multiprocessing import Pool
import h5py, cv2
import numpy as np
import pandas as pd

ATOMIC_TASKS = [
    "PnPCabToCounter", "PnPCounterToCab", "PnPCounterToMicrowave", "PnPCounterToSink",
    "PnPCounterToStove", "PnPMicrowaveToCounter", "PnPSinkToCounter", "PnPStoveToCounter",
    "OpenSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor", "CloseSingleDoor",
    "OpenDrawer", "CloseDrawer", "TurnOnStove", "TurnOffStove",
    "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave", "CoffeeServeMug", "CoffeeSetupMug",
]  # 24 atomic, NavigateKitchen excluded (nav-only)

VIDEO_KEYS = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]
STATE_KEYS = ["robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat"]  # 2+3+4 = 9 (Cosmos order)
STATE_DIM, ACTION_DIM = 9, 7
ROBOT_TYPE = "panda_robocasa_cosmos"


def _natural(k):
    try: return int(k.split("_")[-1])
    except ValueError: return 10**9


def _jattr(v):
    if isinstance(v, bytes): v = v.decode("utf-8")
    return json.loads(v)


def find_hdf5(input_root: Path, task: str, mg: bool):
    if mg:
        m = sorted(input_root.glob(f"*/{task}/mg/*/*.hdf5"))
    else:
        m = [p for p in input_root.glob(f"*/{task}/*/*.hdf5") if "/mg/" not in str(p)]
    return m[0] if m else None


def demo_list(h5, filter_key):
    if filter_key:
        mp = f"mask/{filter_key}"
        if mp not in h5:
            raise KeyError(f"{mp} missing; have {sorted(h5.get('mask',{}).keys())}")
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h5[mp][()]]
    else:
        names = list(h5["data"].keys())
    return sorted(names, key=_natural)


def build_state(obs):
    arrs = [np.asarray(obs[k], dtype=np.float32) for k in STATE_KEYS]
    s = np.concatenate(arrs, axis=1)
    assert s.shape[1] == STATE_DIM, f"state dim {s.shape[1]} != {STATE_DIM}"
    return s


def noop_keep(actions7, thr=1e-3):
    """Keep frames with eef motion or a gripper toggle; drop idle no-ops."""
    motion = np.abs(actions7[:, :6]).max(axis=1) > thr
    grip = actions7[:, 6]
    gch = np.zeros(len(grip), bool); gch[1:] = np.abs(np.diff(grip)) > 0.5
    keep = motion | gch
    keep[0] = keep[-1] = True
    return keep


def write_video(path: Path, frames, fps, res=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = (res, res) if res else (int(frames.shape[1]), int(frames.shape[2]))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened(): raise RuntimeError(f"VideoWriter failed: {path}")
    try:
        for fr in frames:
            a = np.asarray(fr, np.uint8)
            if res and (a.shape[0] != res or a.shape[1] != res):
                a = cv2.resize(a, (res, res), interpolation=cv2.INTER_CUBIC)
            vw.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    finally: vw.release()


def feat_stats(arr):
    a = np.asarray(arr, np.float64)
    if a.ndim == 1: a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def make_info(n_ep, n_frames, n_tasks, fps, res):
    vfeat = {"dtype": "video", "shape": [res, res, 3], "names": ["height", "width", "channel"],
             "video_info": {"video.fps": fps}}
    return {"codebase_version": "v2.0", "robot_type": ROBOT_TYPE, "fps": fps,
            "total_episodes": n_ep, "total_frames": n_frames, "total_tasks": n_tasks,
            "total_videos": n_ep * len(VIDEO_KEYS), "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
                "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
                "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
                "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
                "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": ["robot"]},
                "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ["robot"]},
                **{k: vfeat for k in VIDEO_KEYS}}}


def make_modality():
    return {
        "video": {"agentview_left": {"original_key": "robot0_agentview_left_image"},
                  "agentview_right": {"original_key": "robot0_agentview_right_image"},
                  "wrist": {"original_key": "robot0_eye_in_hand_image"}},
        "state": {
            "gripper_qpos": {"original_key": "observation.state", "start": 0, "end": 2, "dtype": "float32", "absolute": True},
            "eef_position": {"original_key": "observation.state", "start": 2, "end": 5, "dtype": "float32", "absolute": True},
            "eef_quaternion": {"original_key": "observation.state", "start": 5, "end": 9, "rotation_type": "quaternion", "dtype": "float32", "absolute": True}},
        "action": {
            "end_effector_position": {"original_key": "action", "start": 0, "end": 3, "dtype": "float32", "absolute": False},
            "end_effector_rotation": {"original_key": "action", "start": 3, "end": 6, "rotation_type": "axis_angle", "dtype": "float32", "absolute": False},
            "gripper_close": {"original_key": "action", "start": 6, "end": 7, "dtype": "float32", "absolute": False}},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}}}


def gather_demos(h5path, filter_key, fps, noop):
    """Yield (state9, action7, {vkey: frames}, lang) per kept demo."""
    out = []
    with h5py.File(h5path, "r") as h5:
        for dn in demo_list(h5, filter_key):
            g = h5[f"data/{dn}"]; obs = g["obs"]
            actions = np.asarray(g["actions"], np.float32)[:, :ACTION_DIM]
            state = build_state(obs)
            T = len(actions)
            assert len(state) == T, f"{dn}: {len(state)} vs {T}"
            vids = {vk: np.asarray(obs[vk]) for vk in VIDEO_KEYS}
            keep = noop_keep(actions) if noop else np.ones(T, bool)
            lang = str(_jattr(g.attrs["ep_meta"]).get("lang") or "") if "ep_meta" in g.attrs else ""
            out.append((state[keep], actions[keep], {vk: vids[vk][keep] for vk in VIDEO_KEYS}, lang, T, int(keep.sum())))
    return out


def convert_task(args_tuple):
    task, input_root, output_root, human_filter, mg_filter, fps, noop, overwrite, res = args_tuple
    out = Path(output_root) / task
    if out.exists():
        if not overwrite: return f"[skip] {task} exists"
        import shutil; shutil.rmtree(out)
    sources = []
    hp = find_hdf5(Path(input_root), task, mg=False)
    mp = find_hdf5(Path(input_root), task, mg=True)
    if hp: sources.append((hp, human_filter, "human"))
    if mp: sources.append((mp, mg_filter, "mg"))
    if not sources: return f"[MISS] {task}: no hdf5"

    demos = []
    raw_tot = kept_tot = 0
    for path, fk, tier in sources:
        try:
            ds = gather_demos(path, fk, fps, noop)
        except KeyError as e:
            return f"[ERR] {task} {tier}: {e}"
        for s, a, v, lang, T, k in ds:
            demos.append((s, a, v, lang)); raw_tot += T; kept_tot += k

    task_to_idx = OrderedDict(); ep_rows = []; stats_rows = []; n_frames = 0
    for ei, (state, actions, vids, lang) in enumerate(demos):
        lang = lang or task
        if lang not in task_to_idx: task_to_idx[lang] = len(task_to_idx)
        ti = task_to_idx[lang]; T = len(actions); ch = ei // 1000
        rows = [{"timestamp": t / fps, "episode_index": ei, "frame_index": t, "task_index": ti,
                 "observation.state": state[t].astype(np.float32).tolist(),
                 "action": actions[t].astype(np.float32).tolist()} for t in range(T)]
        pq = out / "data" / f"chunk-{ch:03d}" / f"episode_{ei:06d}.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_parquet(pq, index=False)
        for vk in VIDEO_KEYS:
            write_video(out / "videos" / f"chunk-{ch:03d}" / vk / f"episode_{ei:06d}.mp4", vids[vk], fps, res)
        ep_rows.append({"episode_index": ei, "tasks": [lang], "length": T})
        stats_rows.append({"episode_index": ei, "stats": {
            "observation.state": feat_stats(state), "action": feat_stats(actions),
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
    info = make_info(len(ep_rows), n_frames, len(task_to_idx), fps, res)
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "modality.json").write_text(json.dumps(make_modality(), indent=2))
    return f"[done] {task}: {len(ep_rows)} ep, {n_frames} frames (kept {kept_tot}/{raw_tot}={100*kept_tot/max(raw_tot,1):.0f}%), {len(task_to_idx)} lang -> {out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/ssd/sxu/workspace/sizhe/robocasa/datasetFul/v0.1/single_stage")
    ap.add_argument("--output-root", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_v01_cosmos_lerobot")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--all-atomic", action="store_true")
    ap.add_argument("--human-filter", default="", help="'' = all human demos")
    ap.add_argument("--mg-filter", default="300_demos")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-noop-filter", action="store_true", help="disable no-op filtering")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--res", type=int, default=224, help="output video resolution (resize from native 128)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tasks = ATOMIC_TASKS if a.all_atomic else (a.tasks or ["OpenDrawer"])
    jobs = [(t, a.input_root, a.output_root, a.human_filter, a.mg_filter, a.fps,
             not a.no_noop_filter, a.overwrite, a.res) for t in tasks]
    if a.workers > 1 and len(jobs) > 1:
        with Pool(min(a.workers, len(jobs))) as p:
            for r in p.imap_unordered(convert_task, jobs): print(r, flush=True)
    else:
        for j in jobs: print(convert_task(j), flush=True)


if __name__ == "__main__":
    main()
