#!/usr/bin/env python3
"""State-replay grounding for LIBERO / VLA-Arena — the FIXED method.

The original `extract_libero_grounding.py` rebuilds the trajectory by OPEN-LOOP
action replay (`env.step(parquet actions)`), which diverges → grasps fail → the
moving object never moves in the replay → its bbox sticks at the initial position
(gripper_2d still ~tracks). This script instead **replays the recorded sim states**
(`sim.set_state_from_flattened(states[t]); sim.forward()`) so every object is at its
true pose → bbox tracks. gripper_2d + bbox are written INTO the cot frames.

ep → source demo is matched by frame-count + action[:,:6] (the gripper action dim
is sign-flipped between lerobot and the source HDF5). Use the `*_no_noops` source
HDF5 so its `states` are 1:1 with the lerobot frames.

Run in the libero_eval conda env (ForceVLA libero on PYTHONPATH/LIBERO_CONFIG_PATH):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl libero_eval/bin/python extract_libero_grounding_sr.py \
    --suite libero_spatial --episodes 0 \
    --cot_root /…/LIBERO_COT --data_root /…/LEROBOT_LIBERO_DATA \
    --hdf5_dir /…/LIBERO-datasets/libero_spatial_no_noops
"""
import os, sys, json, glob, argparse
os.environ.setdefault("LIBERO_CONFIG_PATH", "/ssd/sxu/workspace/fengnianzhang/ForceVLA/third_party/libero/libero")
os.environ.setdefault("MUJOCO_GL", "egl")
LIBERO_REPO = "/ssd/sxu/workspace/fengnianzhang/ForceVLA/third_party/libero"
sys.path.insert(0, LIBERO_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, h5py, pyarrow.parquet as pq
from pathlib import Path
from libero.libero.envs import SegmentationRenderEnv
from extract_libero_grounding import (project_to_pixel, bbox_from_seg, to_mp4_frame_bbox,
                                      to_mp4_frame_pt, find_bddl_for_instruction, LIBERO_BDDL_DIR)


def match_demo(hdf5_path, pqa):
    """ep → source demo: same frame count + matching action[:,:6] (gripper dim sign-flipped)."""
    N = pqa.shape[0]
    H = h5py.File(hdf5_path, "r")["data"]
    same = [dn for dn in H if H[dn]["actions"].shape[0] == N]
    for dn in same:
        ha = np.asarray(H[dn]["actions"], np.float32)
        if np.allclose(ha[:, :6], pqa[:, :6], atol=1e-2):
            return dn, np.asarray(H[dn]["states"])
    return (same[0], np.asarray(H[same[0]]["states"])) if same else (None, None)


def ground_ep(env, cot_path, states, cam):
    c = json.load(open(cot_path)); frames = c["frames"]; n = len(frames)
    sim = env.env.sim
    ooi = list(env.obj_of_interest)
    allo = ooi + [o for o in env.instance_to_id if o not in ooi and o != "MountedPanda0"]
    manip = ooi[0] if ooi else (allo[0] if allo else None)
    H = W = 256
    for t in range(min(len(states), n)):
        sim.set_state_from_flattened(states[t]); sim.forward()      # ← state replay (the fix)
        try: obs = env.env._get_observations(force_update=True)
        except TypeError: obs = env.env._get_observations()
        fr = frames[t]
        try:
            eef = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")]
            fr["gripper_2d"] = to_mp4_frame_pt(project_to_pixel(eef, sim, cam, H=H, W=W), W, H)
        except Exception:
            fr["gripper_2d"] = None
        seg = obs.get(f"{cam}_segmentation_instance")
        boxes = {}
        if seg is not None:
            for o in allo:
                iid = env.instance_to_id.get(o)
                if iid is None: continue
                b = bbox_from_seg(seg, iid)
                if b: boxes[o] = to_mp4_frame_bbox(b, W, H)
        fr["task_obj_bbox"] = boxes.get(manip)
        fr["distractor_bboxes"] = [boxes[o] for o in boxes if o != manip]
        fr["assistant_object_level"] = "OBJECT:\n" + "".join(f"{o}: {boxes[o]}\n" for o in boxes)
    c["all_object_names"] = allo; c["obj_cat"] = manip
    c["distr_cats"] = [o for o in allo if o != manip]
    c["grounding_camera"] = cam; c["grounding_method"] = "state_replay"; c["grounding_coord_frame"] = "raw_mp4"
    json.dump(c, open(cot_path, "w"), ensure_ascii=False, indent=2)
    g2 = sum(1 for f in frames if f.get("gripper_2d"))
    bb = sum(1 for f in frames if f.get("task_obj_bbox"))
    return f"ok(g2d={g2}/{n} obj_bbox={bb}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--cot_root", required=True)
    ap.add_argument("--data_root", required=True, help="LEROBOT_LIBERO_DATA or vla_arena_libero_cot")
    ap.add_argument("--hdf5_dir", required=True, help="*_no_noops source HDF5 dir (states + model_file)")
    ap.add_argument("--camera", default="agentview")
    a = ap.parse_args()
    lr = Path(a.data_root) / f"{a.suite}_no_noops_1.0.0_lerobot"
    eps_meta = [json.loads(l) for l in open(lr / "meta" / "episodes.jsonl")]
    env_cache = {}
    for ep in a.episodes:
        instr = eps_meta[ep]["tasks"][0]
        pqa = np.stack([np.asarray(x, np.float32) for x in
                        pq.read_table(lr / "data" / "chunk-000" / f"episode_{ep:06d}.parquet").column("action").to_pylist()])
        stem = instr.lower().replace(" ", "_").replace("'", "_")
        hf = [f for f in glob.glob(f"{a.hdf5_dir}/*.hdf5") if stem in Path(f).stem.lower()]
        if not hf:
            print(f"[ep{ep}] NO source HDF5 for '{instr}' under {a.hdf5_dir}", flush=True); continue
        dn, states = match_demo(hf[0], pqa)
        if states is None:
            print(f"[ep{ep}] no matching demo (frame-count) in {Path(hf[0]).name}", flush=True); continue
        bddl = find_bddl_for_instruction(LIBERO_BDDL_DIR / a.suite, instr)
        if str(bddl) not in env_cache:
            env_cache[str(bddl)] = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                                         camera_widths=256, camera_segmentations="instance")
            env_cache[str(bddl)].reset()
        cot = Path(a.cot_root) / a.suite / "extras" / f"episode_{ep:06d}" / "cot_annotations.json"
        if not cot.exists():
            print(f"[ep{ep}] no cot at {cot}", flush=True); continue
        print(f"[ep{ep}] {instr[:42]} demo={dn} -> {ground_ep(env_cache[str(bddl)], str(cot), states, a.camera)}", flush=True)
    for e in env_cache.values():
        try: e.close()
        except Exception: pass


if __name__ == "__main__":
    main()
