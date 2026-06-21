#!/usr/bin/env python3
"""Phase 1 (run in robocasa0.2 env): re-render robocasa v0.1 demos at 224, writing
3-cam MP4 directly into the LeRobot videos/ layout + per-demo npz (state9/action7/
lang, no-op filtered). Bypasses the v0.2 task env (incompatible with v0.1 models):
raw MjSim on the path-rewritten MJCF, visual-geom-only render.

  cd /ssd/sxu/workspace/sizhe/robocasa && CUDA_VISIBLE_DEVICES=g MUJOCO_GL=egl \
    python render_robocasa_224.py --tasks OpenDrawer --out-root <dir> --npz-root <dir>
"""
import os, re, json, glob, argparse, traceback
from pathlib import Path
import numpy as np, h5py, cv2, mujoco
from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen

VIDEO_KEYS = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
STATE_KEYS = ["robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat"]  # 2+3+4=9 (Cosmos order)
ACTION_DIM = 7
LOCAL = "/ssd/sxu/workspace/sizhe"


def fix_xml(xml):
    """Rewrite ANY dev-machine asset prefix (/Users/abhishek/..., /data1/aaronl/...,
    etc.) ending in /<pkg>/models/assets/ to the local install."""
    if isinstance(xml, bytes): xml = xml.decode()
    for pkg in ("robosuite", "robocasa"):
        xml = re.sub(rf'"[^"]*?/{pkg}/models/assets/', f'"{LOCAL}/{pkg}/{pkg}/models/assets/', xml)
    return xml


def noop_keep(a, thr=1e-3):
    motion = np.abs(a[:, :6]).max(1) > thr
    grip = a[:, 6]; gch = np.zeros(len(grip), bool); gch[1:] = np.abs(np.diff(grip)) > 0.5
    keep = motion | gch; keep[0] = keep[-1] = True
    return keep


def find_hdf5(root, task, mg):
    if mg:
        m = sorted(glob.glob(f"{root}/*/{task}/mg/*/*.hdf5"))
    else:
        m = [p for p in glob.glob(f"{root}/*/{task}/*/*.hdf5") if "/mg/" not in p]
    return m[0] if m else None


def demo_list(h5, fk):
    if fk:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h5[f"mask/{fk}"][()]]
    else:
        names = list(h5["data"].keys())
    return sorted(names, key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 10**9)


def write_mp4(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames: vw.write(cv2.cvtColor(np.asarray(f, np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def render_demo(xml, states, keep, res):
    m = mujoco.MjModel.from_xml_string(fix_xml(xml))
    sim = MjSim(m); ctx = MjRenderContextOffscreen(sim, device_id=0); sim.add_render_context(ctx)
    ctx.vopt.geomgroup[:] = 0; ctx.vopt.geomgroup[1] = 1   # visual geoms only
    frames = {c: [] for c in CAMS}
    for t in range(len(states)):
        if not keep[t]: continue
        sim.set_state_from_flattened(states[t]); sim.forward()
        for c in CAMS:
            frames[c].append(np.asarray(sim.render(camera_name=c, width=res, height=res), np.uint8)[::-1])
    # free GL
    try: ctx.close()
    except Exception: pass
    return {c: np.asarray(frames[c]) for c in CAMS}


def render_task(task, root, out_root, npz_root, human_fk, mg_fk, res, fps, noop):
    out = Path(out_root) / task; npzd = Path(npz_root) / task
    npzd.mkdir(parents=True, exist_ok=True)
    sources = []
    hp = find_hdf5(root, task, False); mp = find_hdf5(root, task, True)
    if hp: sources.append((hp, human_fk))
    if mp: sources.append((mp, mg_fk))
    if not sources: return f"[MISS] {task}"
    ei = 0; ok = 0; fail = 0
    for path, fk in sources:
        with h5py.File(path, "r") as f:
            for dn in demo_list(f, fk):
                g = f[f"data/{dn}"]; obs = g["obs"]
                try:
                    actions = np.asarray(g["actions"], np.float32)[:, :ACTION_DIM]
                    state = np.concatenate([np.asarray(obs[k], np.float32) for k in STATE_KEYS], 1)
                    states = np.asarray(g["states"]); xml = g.attrs["model_file"]
                    lang = json.loads(g.attrs["ep_meta"]).get("lang", "") if "ep_meta" in g.attrs else ""
                    keep = noop_keep(actions) if noop else np.ones(len(actions), bool)
                    vids = render_demo(xml, states, keep, res)
                    ch = ei // 1000
                    for c, vk in zip(CAMS, VIDEO_KEYS):
                        write_mp4(out / "videos" / f"chunk-{ch:03d}" / vk / f"episode_{ei:06d}.mp4", vids[c], fps)
                    np.savez(npzd / f"episode_{ei:06d}.npz", state=state[keep], action=actions[keep], lang=lang)
                    ei += 1; ok += 1
                except Exception as e:
                    fail += 1
                    print(f"  [err] {task}/{dn}: {repr(e)[:80]}", flush=True)
    return f"[render] {task}: ok={ok} fail={fail} -> {out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/ssd/sxu/workspace/sizhe/robocasa/datasetFul/v0.1/single_stage")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--npz-root", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--human-filter", default="")
    ap.add_argument("--mg-filter", default="300_demos")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-noop-filter", action="store_true")
    a = ap.parse_args()
    for t in a.tasks:
        print(render_task(t, a.input_root, a.out_root, a.npz_root, a.human_filter, a.mg_filter,
                          a.res, a.fps, not a.no_noop_filter), flush=True)


if __name__ == "__main__":
    main()
