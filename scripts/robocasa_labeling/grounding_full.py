#!/usr/bin/env python3
"""Full-scale RoboCasa grounding (robocasa0.2, raw mujoco): gripper_2d + object
bbox for ALL episodes, written directly into the cot frames (libero format).
Sharded across GPUs (run one process per GPU with --shard-id/--num-shards).

ep->demo mapping reconstructs the converter order: human demos (sorted) then
MimicGen 300_demos (sorted). Per kept (no-op-filtered) frame: project world eef
-> gripper_2d, segmentation -> obj/distr bboxes. Writes per-frame gripper_2d,
task_obj_bbox, distractor_bboxes, assistant_object_level + top-level names.
"""
import os, re, json, glob, argparse, math
from pathlib import Path
import numpy as np, h5py, mujoco

LOCAL = "/ssd/sxu/workspace/sizhe"; CAM = "robot0_agentview_left"; RES = 256


def fix_xml(x):
    x = x.decode() if isinstance(x, bytes) else x
    for p in ("robosuite", "robocasa"):
        x = re.sub(rf'"[^"]*?/{p}/models/assets/', f'"{LOCAL}/{p}/{p}/models/assets/', x)
    return x


def _nat(k):
    try: return int(k.split("_")[-1])
    except: return 10**9


def noop_keep(a, thr=1e-3):
    mo = np.abs(a[:, :6]).max(1) > thr; gr = a[:, 6]
    gc = np.zeros(len(gr), bool); gc[1:] = np.abs(np.diff(gr)) > 0.5
    k = mo | gc; k[0] = k[-1] = True; return k


def find_hdf5(root, task, mg):
    if mg: m = sorted(glob.glob(f"{root}/*/{task}/mg/*/*.hdf5"))
    else: m = [p for p in glob.glob(f"{root}/*/{task}/*/*.hdf5") if "/mg/" not in p]
    return m[0] if m else None


def ep_to_demo(task, root, mg_filter):
    mp = {}; ei = 0
    hp = find_hdf5(root, task, False); gp = find_hdf5(root, task, True)
    if hp:
        with h5py.File(hp, "r") as f:
            for dn in sorted(f["data"].keys(), key=_nat): mp[ei] = (hp, dn); ei += 1
    if gp:
        with h5py.File(gp, "r") as f:
            names = [x.decode() if isinstance(x, bytes) else str(x) for x in f[f"mask/{mg_filter}"][()]]
            for dn in sorted(names, key=_nat): mp[ei] = (gp, dn); ei += 1
    return mp


def project(eef, cp, cm, fovy, H, W):
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    c = cm.T @ (np.asarray(eef) - np.asarray(cp))
    if -c[2] <= 1e-6: return None
    return [float(f * (c[0] / -c[2]) + W / 2), float(-f * (c[1] / -c[2]) + H / 2)]


def ground_ep(cot_path, hdf5, demo_name, do_bbox):
    c = json.load(open(cot_path)); frames = c["frames"]; n = len(frames)
    with h5py.File(hdf5, "r") as f:
        g = f[f"data/{demo_name}"]; obs = g["obs"]
        actions = np.asarray(g["actions"], np.float32)[:, :7]
        eef = np.asarray(obs["robot0_eef_pos"], np.float64)
        states = np.asarray(g["states"]); xml = g.attrs["model_file"]
        em = json.loads(g.attrs["ep_meta"]) if "ep_meta" in g.attrs else {}
    keep = noop_keep(actions)
    if int(keep.sum()) != n:        # alignment guard
        return f"misalign({int(keep.sum())}!={n})"
    m = mujoco.MjModel.from_xml_string(fix_xml(xml))
    for cam, cfg in em.get("cam_configs", {}).items():
        ci = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if ci < 0: continue
        if "pos" in cfg: m.cam_pos[ci] = np.array(cfg["pos"], float)
        if "quat" in cfg: m.cam_quat[ci] = np.array(cfg["quat"], float)
        fv = cfg.get("camera_attribs", {}).get("fovy")
        if fv is not None: m.cam_fovy[ci] = float(fv)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM); fovy = float(m.cam_fovy[cid])
    nq, nv = m.nq, m.nv; data = mujoco.MjData(m)
    objs = [o.get("name") for o in em.get("object_cfgs", []) if o.get("name")]
    gn = [m.geom(i).name or "" for i in range(m.ngeom)]
    obj_geoms = {o: set(i for i, nm in enumerate(gn) if nm.startswith(o + "_g") or nm == o) for o in objs}
    obj_geoms = {o: s for o, s in obj_geoms.items() if s}
    renderer = mujoco.Renderer(m, height=RES, width=RES) if do_bbox else None
    if renderer: renderer.enable_segmentation_rendering()
    kept = np.where(keep)[0]
    manip = objs[0] if objs else None
    for fi, t in enumerate(kept):
        data.qpos[:] = states[t][1:1 + nq]; data.qvel[:] = states[t][1 + nq:1 + nq + nv]
        mujoco.mj_forward(m, data)
        p = project(eef[t], data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy(), fovy, RES, RES)
        fr = frames[fi]; fr["gripper_2d"] = ([int(round(p[0])), int(round(p[1]))] if p else None)
        boxes = {}
        if renderer:
            try:
                renderer.update_scene(data, camera=cid); seg = renderer.render()[:, :, 0]
                for o, gids in obj_geoms.items():
                    mk = np.isin(seg, list(gids))
                    if mk.any():
                        ys, xs = np.where(mk); boxes[o] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            except Exception: pass
        fr["task_obj_bbox"] = boxes.get(manip)
        fr["distractor_bboxes"] = [boxes[o] for o in boxes if o != manip]
        fr["assistant_object_level"] = "OBJECT:\n" + "".join(
            f"{o}: [{b[0]},{b[1]}], [{b[2]},{b[3]}]\n" for o, b in boxes.items())
    if renderer:
        try: renderer.close()
        except: pass
    c["all_object_names"] = list(obj_geoms.keys()); c["obj_cat"] = manip
    c["distr_cats"] = [o for o in obj_geoms if o != manip]
    c["grounding_camera"] = CAM
    json.dump(c, open(cot_path, "w"), ensure_ascii=False, indent=2)
    g2 = sum(1 for f in frames if f.get("gripper_2d"))
    return f"ok(g2d={g2}/{n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-root", required=True); ap.add_argument("--input-root", default="/ssd/sxu/workspace/sizhe/robocasa/datasetFul/v0.1/single_stage")
    ap.add_argument("--mg-filter", default="300_demos")
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--no-bbox", action="store_true"); ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--episodes", type=int, nargs="*", help="only ground these episode indices (default: all)")
    a = ap.parse_args()
    tasks = a.tasks or sorted(d.name for d in Path(a.cot_root).iterdir() if d.is_dir())
    eps_want = set(a.episodes) if a.episodes else None
    jobs = []
    for t in tasks:
        e2d = ep_to_demo(t, a.input_root, a.mg_filter)
        for ep, (hp, dn) in e2d.items():
            if eps_want is not None and ep not in eps_want: continue
            cot = Path(a.cot_root) / t / "extras" / f"episode_{ep:06d}" / "cot_annotations.json"
            if cot.exists(): jobs.append((t, ep, str(cot), hp, dn))
    jobs.sort()
    per = math.ceil(len(jobs) / a.num_shards)
    jobs = jobs[a.shard_id * per:(a.shard_id + 1) * per]
    print(f"[shard {a.shard_id}/{a.num_shards}] {len(jobs)} eps", flush=True)
    from collections import Counter; st = Counter()
    for i, (t, ep, cot, hp, dn) in enumerate(jobs):
        try: r = ground_ep(cot, hp, dn, not a.no_bbox)
        except Exception as e: r = f"ERR:{repr(e)[:60]}"
        st[r.split("(")[0].split(":")[0]] += 1
        if i % 100 == 0: print(f"  [s{a.shard_id} {i}/{len(jobs)}] {t} ep{ep}: {r}", flush=True)
    print(f"[shard {a.shard_id}] DONE {dict(st)}", flush=True)


if __name__ == "__main__":
    main()
