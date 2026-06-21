#!/usr/bin/env python3
"""Copy subtask-independent grounding (gripper_2d + object bboxes) from the
ORIGINAL LIBERO_COT into the freshly LNDS-relabeled COT, matched by frame_index.
Grounding is a function of the trajectory (eef projection + sim seg), not of the
subtask labels, so it is identical between the two versions — reusing it avoids
re-running the libero MuJoCo sim for a quick comparison sample."""
import argparse, json
from pathlib import Path

FRAME_KEYS = ["gripper_2d", "task_obj_bbox", "distractor_bboxes", "assistant_object_level"]
TOP_KEYS = ["all_object_names", "obj_cat", "distr_cats", "grounding_camera", "grounding_coord_frame"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_root", required=True)
    ap.add_argument("--new_root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes", type=int, nargs="+")
    ap.add_argument("--all", action="store_true", help="all episodes present in new_root/<suite>/extras")
    a = ap.parse_args()
    if a.all:
        nb = Path(a.new_root) / a.suite / "extras"
        eps = sorted(int(d.name.split("_")[1]) for d in nb.glob("episode_*") if d.is_dir())
    else:
        eps = a.episodes or []
    done = 0
    for e in eps:
        op = Path(a.orig_root) / a.suite / "extras" / f"episode_{e:06d}" / "cot_annotations.json"
        np_ = Path(a.new_root) / a.suite / "extras" / f"episode_{e:06d}" / "cot_annotations.json"
        if not op.exists() or not np_.exists():
            print(f"  {a.suite} ep{e}: SKIP (orig={op.exists()} new={np_.exists()})"); continue
        o = json.load(open(op)); n = json.load(open(np_))
        of = {f["frame_index"]: f for f in o["frames"]}
        for fr in n["frames"]:
            src = of.get(fr["frame_index"])
            if not src:
                continue
            for k in FRAME_KEYS:
                if k in src:
                    fr[k] = src[k]
        for k in TOP_KEYS:
            if k in o:
                n[k] = o[k]
        json.dump(n, open(np_, "w"), ensure_ascii=False, indent=2)
        done += 1
    print(f"[merge] {a.suite}: grounding merged into {done}/{len(eps)} episodes")


if __name__ == "__main__":
    main()
