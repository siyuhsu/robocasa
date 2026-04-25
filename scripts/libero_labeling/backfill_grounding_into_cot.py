"""
Phase D.3: Backfill bbox + gripper_2d into cot_annotations.json.

For each episode that has both:
  - LIBERO_COT/<suite>/extras/episode_XXXXXX/cot_annotations.json  (Phase B,
    motion-only — bbox/gripper_2d fields are null)
  - LIBERO_GROUNDING/<suite>/extras/episode_XXXXXX/annotations.json (Phase D.2,
    bbox + gripper_2d in raw-MP4 frame)

merge the grounding into the per-frame records of cot_annotations.json:
  - frames[t]["gripper_2d"]         ← grounding agentview gripper_2d[t]
  - frames[t]["task_obj_bbox"]      ← grounding agentview all_object_bboxes[t][0]
                                       (first task object, == obj_cat)
  - frames[t]["distractor_bboxes"]  ← grounding agentview all_object_bboxes[t][1:]
  - frames[t]["assistant_position_level"]  ← rendered "NEXT GRIPPER: [x, y]\n"
  - frames[t]["assistant_object_level"]    ← rendered "OBJECT:\n<obj>: [x1,y1], [x2,y2]\n..."

Top-level adds:
  - all_object_names, obj_cat, distr_cats
  - grounding_camera = "agentview"
  - grounding_coord_frame = "raw_mp4"

Updates files in place. Idempotent — running twice is a no-op aside from
recomputing the templates.

Usage:
  python scripts/libero_labeling/backfill_grounding_into_cot.py \
      --cot_root playground/Datasets/LIBERO_COT \
      --grounding_root playground/Datasets/LIBERO_GROUNDING \
      --suite all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


LIBERO_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]


# Templates — match robocasa CoT prompt style for downstream training compat.
POSITION_LEVEL_TEMPLATE = "NEXT GRIPPER: {gripper_2d_next}\n"
OBJECT_LEVEL_TEMPLATE = "OBJECT:\n{objects}\n"


def _objects_text(frame_boxes: list, object_names: list[str]) -> str:
    """Render bbox dict block: 'name: [x1,y1], [x2,y2]\n...'."""
    lines = []
    for i, bbox in enumerate(frame_boxes):
        if bbox is None or len(bbox) != 4:
            continue
        name = object_names[i] if i < len(object_names) else f"obj_{i}"
        x1, y1, x2, y2 = [int(v) for v in bbox]
        lines.append(f"{name}: [{x1},{y1}], [{x2},{y2}]")
    return "\n".join(lines)


def backfill_episode(cot_path: Path, grounding_path: Path,
                     camera: str = "agentview") -> tuple[bool, str]:
    """Merge one episode's grounding into its cot_annotations.json. Returns
    (changed, reason). Writes back atomically (.tmp + rename)."""
    if not cot_path.exists():
        return (False, "cot missing")
    if not grounding_path.exists():
        return (False, "grounding missing")

    with open(cot_path) as f:
        cot = json.load(f)
    with open(grounding_path) as f:
        gr = json.load(f)

    cam = gr.get("cameras", {}).get(camera)
    if cam is None:
        return (False, f"camera {camera} missing in grounding")

    gripper_2d = cam.get("gripper_2d", [])
    all_obj_bboxes = cam.get("all_object_bboxes", [])
    all_object_names = gr.get("all_object_names", [])
    obj_cat = gr.get("obj_cat")
    distr_cats = gr.get("distr_cats", [])

    n_cot = len(cot.get("frames", []))
    n_gr = min(len(gripper_2d), len(all_obj_bboxes))
    n = min(n_cot, n_gr)
    if n == 0:
        return (False, "no overlap")

    # Top-level metadata
    cot["all_object_names"] = all_object_names
    cot["obj_cat"] = obj_cat
    cot["distr_cats"] = distr_cats
    cot["grounding_camera"] = camera
    cot["grounding_coord_frame"] = gr.get("coord_frame", "raw_mp4")

    # Per-frame backfill
    for t in range(n):
        rec = cot["frames"][t]
        frame_boxes = all_obj_bboxes[t] if t < len(all_obj_bboxes) else []
        rec["gripper_2d"] = gripper_2d[t] if t < len(gripper_2d) else None
        rec["task_obj_bbox"] = frame_boxes[0] if frame_boxes else None
        rec["distractor_bboxes"] = frame_boxes[1:] if len(frame_boxes) > 1 else []

        # Rendered text fields for prompt
        # NEXT GRIPPER references the gripper_2d at frame t+1 (or t if at end)
        next_t = t + 1 if t + 1 < n else t
        next_gp = gripper_2d[next_t] if next_t < len(gripper_2d) else None
        if next_gp is not None and len(next_gp) == 2:
            rec["assistant_position_level"] = POSITION_LEVEL_TEMPLATE.format(
                gripper_2d_next=f"[{int(next_gp[0])}, {int(next_gp[1])}]"
            )
        else:
            rec["assistant_position_level"] = None

        objects_text = _objects_text(frame_boxes, all_object_names)
        rec["assistant_object_level"] = (
            OBJECT_LEVEL_TEMPLATE.format(objects=objects_text) if objects_text else None
        )

    # Atomic write
    tmp = cot_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cot, f, ensure_ascii=False, indent=2)
    tmp.replace(cot_path)
    return (True, f"merged {n} frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cot_root", required=True,
                   help="LIBERO_COT root (Phase B output)")
    p.add_argument("--grounding_root", required=True,
                   help="LIBERO_GROUNDING root (Phase D.2 output)")
    p.add_argument("--suite", default="all", choices=["all", *LIBERO_SUITES])
    p.add_argument("--camera", default="agentview",
                   help="Which camera's bbox/gripper to backfill")
    args = p.parse_args()

    cot_root = Path(args.cot_root).resolve()
    gr_root = Path(args.grounding_root).resolve()
    suites = LIBERO_SUITES if args.suite == "all" else [args.suite]

    total = {"merged": 0, "skipped": 0, "missing_grounding": 0, "missing_cot": 0}

    for suite in suites:
        cot_dir = cot_root / suite / "extras"
        gr_dir = gr_root / suite / "extras"
        if not cot_dir.exists():
            print(f"[warn] {suite}: no LIBERO_COT/{suite}/extras")
            continue

        ep_dirs = sorted([d for d in cot_dir.iterdir()
                          if d.is_dir() and d.name.startswith("episode_")])
        print(f"\n[suite {suite}] {len(ep_dirs)} cot episodes")

        for ep_dir in tqdm(ep_dirs, desc=suite):
            cot_path = ep_dir / "cot_annotations.json"
            gr_path = gr_dir / ep_dir.name / "annotations.json"
            ok, reason = backfill_episode(cot_path, gr_path, camera=args.camera)
            if ok:
                total["merged"] += 1
            elif reason.startswith("grounding"):
                total["missing_grounding"] += 1
            elif reason.startswith("cot"):
                total["missing_cot"] += 1
            else:
                total["skipped"] += 1

    print(f"\n[done] merged={total['merged']}, "
          f"missing_grounding={total['missing_grounding']}, "
          f"missing_cot={total['missing_cot']}, "
          f"skipped={total['skipped']}")


if __name__ == "__main__":
    main()
