#!/usr/bin/env python3
"""Offline subtask-order re-correction (NO VLM, NO re-label).

label_vla_arena_episodes.py saves the RAW per-segment VLM labels
(`segment_labels_raw`) + `subtask_list` into each cot_annotations.json. This
script re-applies `correct_subtask_order` (weighted-LNDS backbone + position
fill) to those raw labels and rewrites the per-frame subtask/reason +
assistant_short_plan + assistant_plan_level (LONG PLAN). Iterate the correction
algorithm freely without re-running the VLM.

Usage:
  python fix_subtask_order.py --cot_root <VLA_ARENA_COT> --suite vla_arena_l0 [--episodes ...]
"""
import argparse, json, glob, sys
from pathlib import Path

sys.path.insert(0, "/ssd/sxu/workspace/code/robocasa/scripts/vla_arena_labeling")
sys.path.insert(0, "/ssd/sxu/workspace/code/robocasa/scripts/robocasa_labeling")
from subtask_order import correct_subtask_order          # noqa: E402
from generate_subtasks import build_frame_annotations     # noqa: E402

LONG_PLAN_TEMPLATE = "LONG PLAN:\n{long_plan}\n"
SHORT_PLAN_TEMPLATE = "SHORT PLAN:\nSubtask: {subtask}\nSubtask Reasoning: {reasoning}\n"


def build_long_plan_text(frame_annotations):
    items, prev = [], None
    for ann in frame_annotations:
        st = ann[0] if isinstance(ann, list) and len(ann) >= 1 else "unknown"
        if st == prev:
            continue
        items.append(st); prev = st
    return "\n".join(f"{j + 1}. {st}" for j, st in enumerate(items))


def fix_one(cot_path):
    cot = json.load(open(cot_path))
    raw = cot.get("segment_labels_raw"); subs = cot.get("subtask_list")
    if not raw or not subs:
        return None
    raw = {int(k): v for k, v in raw.items()}
    overall = [f.get("segment_id") for f in cot["frames"]]
    corrected = correct_subtask_order(raw, overall, subs)
    fa = build_frame_annotations(overall, corrected)
    alp = LONG_PLAN_TEMPLATE.format(long_plan=build_long_plan_text(fa))
    for i, f in enumerate(cot["frames"]):
        st, reason = (fa[i][0], fa[i][1]) if i < len(fa) and len(fa[i]) >= 2 else ("unknown", "unknown")
        f["subtask"] = st; f["reason"] = reason
        f["assistant_short_plan"] = SHORT_PLAN_TEMPLATE.format(subtask=st, reasoning=reason)
        f["assistant_plan_level"] = alp
    json.dump(cot, open(cot_path, "w"), ensure_ascii=False, indent=2)
    return len(set(a[0] for a in fa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot_root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    a = ap.parse_args()
    base = Path(a.cot_root) / a.suite / "extras"
    paths = sorted(glob.glob(str(base / "episode_*/cot_annotations.json")))
    if a.episodes:
        want = {f"episode_{e:06d}" for e in a.episodes}
        paths = [p for p in paths if Path(p).parent.name in want]
    ok = 0; skip = 0
    for p in paths:
        r = fix_one(p)
        if r is None: skip += 1
        else: ok += 1
    print(f"[fix] corrected {ok}, skipped(no raw) {skip}, of {len(paths)}")


if __name__ == "__main__":
    main()
