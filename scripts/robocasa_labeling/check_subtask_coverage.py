#!/usr/bin/env python3
"""QC: flag cots whose merged LONG PLAN does NOT cover the full Stage-1 plan
(subtask_list) — i.e., the instruction was not fully realized into subtasks
(typically long-horizon 2nd-object phases merged/mislabeled away)."""
import json, glob, argparse, re
from pathlib import Path
from collections import Counter

VERBS_START = ("approach", "reach", "move to", "move toward", "go to", "navigate")


def covered_max(plan, longplan):
    """max index of plan subtask realized in longplan (exact, else token-fuzzy)."""
    idx = -1
    pl = [p.lower() for p in plan]
    for s in longplan:
        s = s.lower()
        if s in pl:
            idx = max(idx, pl.index(s))
        else:
            best, bi = 0, -1
            st = set(re.findall(r"[a-z]+", s))
            for j, p in enumerate(pl):
                pt = set(re.findall(r"[a-z]+", p))
                ov = len(st & pt) / max(len(st | pt), 1)
                if ov > best:
                    best, bi = ov, j
            if best >= 0.6:
                idx = max(idx, bi)
    return idx


def is_multiphase(plan):
    """plan has >=2 object phases (a 2nd approach/reach after a place/release)."""
    starts = sum(1 for p in plan if any(p.lower().startswith(v) for v in ("approach", "reach")))
    return starts >= 2


def check_one(p):
    c = json.load(open(p))
    plan = c.get("subtask_list") or []
    if len(plan) < 3:
        return "no_plan"
    runs, prev = [], None
    for f in c["frames"]:
        if f.get("subtask") != prev:
            runs.append(f.get("subtask")); prev = f.get("subtask")
    mc = covered_max(plan, runs)
    truncated = mc < len(plan) - 2          # didn't reach the final 1-2 plan subtasks
    multi = is_multiphase(plan)
    if truncated and multi:
        return "trunc_multi"
    if truncated:
        return "trunc_single"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    for root in a.roots:
        paths = sorted(glob.glob(str(Path(root) / "**" / "extras" / "episode_*" / "cot_annotations.json"), recursive=True))
        st = Counter(); flagged = []
        for p in paths:
            r = check_one(p)
            st[r] += 1
            if r.startswith("trunc"):
                flagged.append(p)
        name = Path(root).name
        print(f"[{name}] {dict(st)} (total {len(paths)})", flush=True)
        if a.list:
            for f in flagged[:15]:
                print("   FLAG", "/".join(f.split("/")[-3:]))
    return 0


if __name__ == "__main__":
    main()
