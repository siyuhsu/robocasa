#!/usr/bin/env python3
"""Fix truncated RoboCasa cots so the FULL Stage-1 plan is realized. RoboCasa
tasks execute linearly (approach->...->place->release, incl. open/close container),
so lay the complete plan proportionally across the segments in order: guarantees
the final subtask is reached (no truncation). Applied ONLY to flagged cots; OK
cots are left untouched. Offline, no VLM/sim. Reason stays per-segment (raw)."""
import sys, json, glob, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/ssd/sxu/workspace/code/robocasa/scripts/robocasa_labeling")
from check_subtask_coverage import check_one

LONG = "LONG PLAN:\n{lp}\n"
SHORT = "SHORT PLAN:\nSubtask: {s}\nSubtask Reasoning: {r}\n"


def realize_full(cot_path):
    c = json.load(open(cot_path)); plan = c.get("subtask_list") or []
    raw = {int(k): v for k, v in (c.get("segment_labels_raw") or {}).items()}
    if len(plan) < 3:
        return "no_plan"
    frames = c["frames"]
    seg_ids = []
    for f in frames:
        if not seg_ids or seg_ids[-1] != f["segment_id"]:
            seg_ids.append(f["segment_id"])
    seg_pos = {s: i + 1 for i, s in enumerate(seg_ids)}
    M, P = len(seg_ids), len(plan)
    new_sub = {}
    for j, s in enumerate(seg_ids):
        loc = round(j * (P - 1) / max(M - 1, 1)) if M > 1 else 0
        new_sub[s] = plan[min(P - 1, loc)]
    raw_reason = {int(k): (v[1] if len(v) >= 2 else "") for k, v in raw.items()}
    runs, prev = [], None
    for f in frames:
        st = new_sub.get(f["segment_id"], f.get("subtask"))
        rs = raw_reason.get(seg_pos.get(f["segment_id"], -1), f.get("reason", ""))
        f["subtask"] = st; f["reason"] = rs
        if st != prev:
            runs.append(st); prev = st
    alp = LONG.format(lp="\n".join(f"{i+1}. {s}" for i, s in enumerate(runs)))
    for f in frames:
        f["assistant_short_plan"] = SHORT.format(s=f["subtask"], r=f["reason"])
        f["assistant_plan_level"] = alp
    json.dump(c, open(cot_path, "w"), ensure_ascii=False, indent=2)
    return "fixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot-root", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--episodes", nargs="*")  # task:ep for testing
    a = ap.parse_args()
    if a.episodes:
        for te in a.episodes:
            t, e = te.split(":")
            p = f"{a.cot_root}/{t}/extras/episode_{int(e):06d}/cot_annotations.json"
            print(te, check_one(p), "->", realize_full(p), check_one(p))
        return
    tasks = a.tasks or sorted(d.name for d in Path(a.cot_root).iterdir() if d.is_dir())
    st = Counter()
    for t in tasks:
        for p in sorted(glob.glob(f"{a.cot_root}/{t}/extras/episode_*/cot_annotations.json")):
            if check_one(p).startswith("trunc"):
                st[realize_full(p)] += 1
            else:
                st["ok_skip"] += 1
    print(f"[reapply_robocasa] {dict(st)}")


if __name__ == "__main__":
    main()
