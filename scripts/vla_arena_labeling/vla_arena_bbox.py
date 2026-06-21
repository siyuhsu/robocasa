#!/usr/bin/env python3
"""VLA-Arena object-bbox grounding via sim replay (run in the VLA-Arena venv).

For each episode: build the env from its chosen bddl (written by
vla_arena_grounding.py) with ELEMENT segmentation (geom-id per pixel), match the
init state by eef (vs the recorded parquet eef[0], dumped to npz), replay the
recorded actions, and extract per-object bboxes via env.env.model.instances_to_ids
(object -> geom ids). Merges all_object_bboxes + all_object_names into the
existing grounding annotations.json (which already holds gripper_2d), in the
LIBERO backfill format, with the 180-deg MP4-frame mapping.
"""
import argparse, json, glob
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "/ssd/sxu/workspace/sizhe/VLA-Arena")
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena import get_vla_arena_path

DUMMY = [0.0] * 6 + [-1.0]
ROBOTISH = ("MountedPanda0", "RethinkMount0", "PandaGripper0_right", "PandaGripper0", "Panda0")


def to_mp4_bbox(b, W, H):
    if b is None:
        return None
    x1, y1, x2, y2 = b
    return [W - 1 - int(x2), H - 1 - int(y2), W - 1 - int(x1), H - 1 - int(y1)]


def bbox_for_geoms(seg, gids):
    m = np.isin(seg, list(gids))
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_inits(bddl_rel):
    # bddl_rel = "<suite>/level_X/<stem>.bddl"
    suite = bddl_rel.split("/")[0]
    stem = Path(bddl_rel).name.replace(".bddl", "")
    sub = "/".join(bddl_rel.split("/")[1:-1])  # level_X
    base = Path(get_vla_arena_path("init_states")) / suite / sub
    import torch
    for ext in (".pruned_init", ".init"):
        p = base / (stem + ext)
        if p.exists():
            return torch.load(str(p), weights_only=False)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounding-root", required=True, help=".../VLA_ARENA_GROUNDING/<suite>/extras")
    ap.add_argument("--eef-dir", required=True, help="dir of episode_*.npz (eef+actions)")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="redo even if bbox present")
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.grounding_root) / "episode_*/annotations.json")))
    if args.episodes:
        want = {f"episode_{e:06d}" for e in args.episodes}
        paths = [p for p in paths if Path(p).parent.name in want]
    if args.num_shards > 1:
        import math
        paths.sort()
        per = math.ceil(len(paths) / args.num_shards)
        paths = paths[args.shard_id*per:(args.shard_id+1)*per]

    env = None; cur_bddl = None
    for ap_ in paths:
        gj = json.load(open(ap_))
        ei = int(Path(ap_).parent.name.split("_")[1])
        _ex = gj.get("cameras",{}).get("agentview",{}).get("all_object_bboxes")
        if _ex and any(any(b for b in fr) for fr in _ex) and not args.force:
            print(f"  ep{ei}: skip (bbox present)", flush=True); continue
        bddl_rel = gj.get("bddl")
        npz = Path(args.eef_dir) / f"episode_{ei:06d}.npz"
        if not bddl_rel or not npz.exists():
            print(f"  ep{ei}: skip (no bddl/npz)"); continue
        d = np.load(npz, allow_pickle=True); eef0 = d["eef"][0]; actions = d["actions"]
        bddl_abs = str(Path(get_vla_arena_path("bddl_files")) / bddl_rel)
        if bddl_rel != cur_bddl:
            if env is not None: env.close()
            env = OffScreenRenderEnv(bddl_file_name=bddl_abs, camera_heights=args.res,
                                     camera_widths=args.res, camera_names=["agentview"],
                                     camera_segmentations="element")
            cur_bddl = bddl_rel
        geom_map = env.env.model.instances_to_ids
        ooi = list(env.obj_of_interest)
        all_objs = ooi + [o for o in geom_map if o not in ooi and o not in ROBOTISH]
        gids = {o: set(geom_map.get(o, {}).get("geom", [])) for o in all_objs}
        # init-match by eef vs recorded eef[0]
        inits = load_inits(bddl_rel)
        best_i, best_d = 0, 1e9
        if inits is not None:
            for i in range(len(inits)):
                obs = env.set_init_state(inits[i])
                e = np.asarray(obs.get("robot0_eef_pos"))
                if e is not None and e.shape == (3,):
                    dist = float(np.linalg.norm(e - eef0))
                    if dist < best_d:
                        best_d, best_i = dist, i
        # replay from best init
        env.reset()
        if inits is not None:
            env.set_init_state(inits[best_i])
        for _ in range(10):
            env.step(DUMMY)
        W = H = args.res
        all_bboxes = []
        n = len(actions)
        for t in range(n):
            obs, *_ = env.step(actions[t].tolist())
            seg = obs.get("agentview_segmentation_element")
            if seg is None:
                all_bboxes.append([None] * len(all_objs)); continue
            seg = np.asarray(seg)
            if seg.ndim == 3: seg = seg[..., 0]
            all_bboxes.append([to_mp4_bbox(bbox_for_geoms(seg, gids[o]), W, H) for o in all_objs])
        # merge into annotations.json (libero backfill format)
        cam = gj.setdefault("cameras", {}).setdefault("agentview", {})
        cam["all_object_bboxes"] = all_bboxes
        gj["all_object_names"] = all_objs
        gj["obj_cat"] = all_objs[0] if all_objs else None
        gj["distr_cats"] = all_objs[1:]
        json.dump(gj, open(ap_, "w"))
        nb = sum(1 for fr in all_bboxes for b in fr if b)
        print(f"  ep{ei} init={best_i}(d={best_d:.3f}) objs={all_objs} frames={n} nonnull_boxes={nb}", flush=True)
    if env is not None:
        env.close()
    print("[bbox] done", flush=True)


if __name__ == "__main__":
    main()
