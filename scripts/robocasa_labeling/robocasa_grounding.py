#!/usr/bin/env python3
"""RoboCasa grounding (robocasa0.2 env, RAW mujoco): per-frame gripper_2d (eef
projection through the mobile-base camera) + object bboxes (segmentation render).

Raw mujoco (MjModel/MjData + mujoco.Renderer) avoids robosuite's render wrapper.
For each task ep0 (= human demo sorted[0], matching the cosmos converter):
rewrite MJCF asset paths, apply ep_meta cam_configs, replay the NO-OP-FILTERED
states, and per frame: mj_forward -> cam_xpos/cam_xmat -> project world eef ->
gripper_2d(224); segmentation render -> per-object (obj + distr_*) bbox.
"""
import re, json, glob, argparse
from pathlib import Path
import numpy as np, h5py, mujoco

LOCAL = "/ssd/sxu/workspace/sizhe"
CAM = "robot0_agentview_left"
RES = 256   # libero256 version (was 224 for cosmos)


def fix_xml(x):
    x = x.decode() if isinstance(x, bytes) else x
    for p in ("robosuite", "robocasa"):
        x = re.sub(rf'"[^"]*?/{p}/models/assets/', f'"{LOCAL}/{p}/{p}/models/assets/', x)
    return x


def noop_keep(a, thr=1e-3):
    mo = np.abs(a[:, :6]).max(1) > thr
    gr = a[:, 6]; gc = np.zeros(len(gr), bool); gc[1:] = np.abs(np.diff(gr)) > 0.5
    k = mo | gc; k[0] = k[-1] = True
    return k


def find_human_hdf5(task):
    m = [p for p in glob.glob(f"/ssd/sxu/workspace/sizhe/robocasa/datasetFul/v0.1/single_stage/*/{task}/*/*.hdf5") if "/mg/" not in p]
    return m[0] if m else None


def project(eef, campos, cammat, fovy, H, W):
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    c = cammat.T @ (np.asarray(eef) - np.asarray(campos))
    if -c[2] <= 1e-6:
        return None
    return [float(f * (c[0] / -c[2]) + W / 2), float(-f * (c[1] / -c[2]) + H / 2)]


def ground_task(task, out_root, do_bbox=True):
    hp = find_human_hdf5(task)
    if not hp:
        return f"[MISS] {task}"
    g = h5py.File(hp, "r")["data"]
    dn = sorted(g.keys(), key=lambda k: int(k.split("_")[-1]))[0]
    demo = g[dn]; states = np.asarray(demo["states"])
    actions = np.asarray(demo["actions"], np.float32)[:, :7]
    eef = np.asarray(demo["obs"]["robot0_eef_pos"], np.float64)
    keep = noop_keep(actions)
    em = json.loads(demo.attrs["ep_meta"]) if "ep_meta" in demo.attrs else {}
    m = mujoco.MjModel.from_xml_string(fix_xml(demo.attrs["model_file"]))
    for cam, cfg in em.get("cam_configs", {}).items():
        ci = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if ci < 0:
            continue
        if "pos" in cfg: m.cam_pos[ci] = np.array(cfg["pos"], float)
        if "quat" in cfg: m.cam_quat[ci] = np.array(cfg["quat"], float)
        fv = cfg.get("camera_attribs", {}).get("fovy")
        if fv is not None: m.cam_fovy[ci] = float(fv)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    fovy = float(m.cam_fovy[cid]); nq, nv = m.nq, m.nv
    data = mujoco.MjData(m)
    # object name -> geom id set  (obj=manipulated, distr_*=distractors)
    objs = [o.get("name") for o in em.get("object_cfgs", []) if o.get("name")]
    gn = [m.geom(i).name or "" for i in range(m.ngeom)]
    obj_geoms = {o: set(i for i, n in enumerate(gn) if n.startswith(o + "_g") or n == o) for o in objs}
    obj_geoms = {o: s for o, s in obj_geoms.items() if s}
    renderer = None
    if do_bbox:
        renderer = mujoco.Renderer(m, height=RES, width=RES); renderer.enable_segmentation_rendering()
    g2d = []; inb = 0
    all_bboxes = []  # per frame: {obj_name: [x1,y1,x2,y2]}
    for t in np.where(keep)[0]:
        data.qpos[:] = states[t][1:1 + nq]; data.qvel[:] = states[t][1 + nq:1 + nq + nv]
        mujoco.mj_forward(m, data)
        p = project(eef[t], data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy(), fovy, RES, RES)
        g2d.append(p)
        if p and 0 <= p[0] < RES and 0 <= p[1] < RES:
            inb += 1
        if renderer is not None:
            fb = {}
            try:
                renderer.update_scene(data, camera=cid)
                seg = renderer.render()[:, :, 0]   # geom id per pixel (-1 = none)
                for o, gids in obj_geoms.items():
                    mask = np.isin(seg, list(gids))
                    if mask.any():
                        ys, xs = np.where(mask)
                        fb[o] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            except Exception:
                pass   # skip bad frame (seg-render IndexError/EGL on some scenes)
            all_bboxes.append(fb)
    rec = {"task": task, "demo": dn, "camera": CAM, "res": RES, "num_frames": len(g2d),
           "in_bounds": inb, "gripper_2d": g2d, "objects": list(obj_geoms.keys()),
           "manipulated": objs[0] if objs else None, "bboxes": all_bboxes}
    if renderer is not None:
        try: renderer.close()        # free GL context (else exhausts after ~16 tasks)
        except Exception: pass
    od = Path(out_root) / task; od.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(od / "grounding.json", "w"))
    cov = sum(1 for fb in all_bboxes if objs and objs[0] in fb) / max(len(all_bboxes), 1) if all_bboxes else 0
    return f"[ground] {task}: f={len(g2d)} g2d_inb={inb}/{len(g2d)} objs={list(obj_geoms.keys())} manip_bbox_cov={cov:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_cot_sample/grounding")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--no-bbox", action="store_true")
    a = ap.parse_args()
    for t in a.tasks:
        print(ground_task(t, a.out, not a.no_bbox), flush=True)


if __name__ == "__main__":
    main()
