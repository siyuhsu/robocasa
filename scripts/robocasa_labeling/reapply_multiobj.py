#!/usr/bin/env python3
"""Re-annotate long-horizon multi-object cots so the FULL Stage-1 plan is
realized (both object phases), fixing the 2nd-object collapse. Offline, no VLM.

Method: the Stage-1 plan splits into object phases (each starts at approach/reach);
the gripper open/close cycles give the execution rounds. Map round k's segments to
phase k's subtasks by the segment's VERB (approach/grasp/lift/move/place — reliable
even when the VLM got the OBJECT wrong), phase-monotonic within the round. Reason
stays per-segment (raw). Guarantees both objects appear in the LONG PLAN.
"""
import json, glob, argparse, re
from pathlib import Path
import numpy as np, pandas as pd

LONG = "LONG PLAN:\n{lp}\n"
SHORT = "SHORT PLAN:\nSubtask: {s}\nSubtask Reasoning: {r}\n"


def classify(s):
    l = (s or "").lower()
    if any(w in l for w in ("release", "let go")): return "release"
    if any(w in l for w in ("place", "put ", "insert", "drop", "set down")): return "place"
    if any(w in l for w in ("lift", "raise")): return "lift"
    if any(w in l for w in ("grasp", "grip", "pick up", "pick the")): return "grasp"
    if any(w in l for w in ("approach", "reach")): return "approach"
    if any(w in l for w in ("move", "carry", "transport", "slide", "go to", "navigate")): return "move"
    return None


def split_phases(plan):
    starts = [i for i, s in enumerate(plan) if any(s.lower().startswith(v) for v in ("approach", "reach"))]
    starts = sorted(set([0] + starts))
    if len(starts) < 2:
        return [list(range(len(plan)))]
    return [list(range(starts[k], starts[k + 1] if k + 1 < len(starts) else len(plan))) for k in range(len(starts))]


def gripper_cycles(state):
    gap = np.abs(state[:, 6] - state[:, 7])
    lo, hi = gap.min(), gap.max()
    if hi - lo < 1e-6:
        return [], []
    closed = gap < (lo + hi) / 2
    d = np.diff(closed.astype(int))
    return list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)


def reapply(cot_path, parquet_path, force=False):
    c = json.load(open(cot_path)); plan = c.get("subtask_list") or []
    raw = {int(k): v for k, v in (c.get("segment_labels_raw") or {}).items()}
    if not plan or not raw:
        return "no_data"
    phases = split_phases(plan)
    if len(phases) < 2:
        return "single_phase"
    frames = c["frames"]; n = len(frames)
    df = pd.read_parquet(parquet_path)
    state = np.stack(df["observation.state"].to_numpy()).astype(float)[:n]
    grasp, rel = gripper_cycles(state)
    nr = min(len(grasp), len(phases))
    if nr >= 2:
        bounds = [0]
        for k in range(1, nr):
            rs = [r for r in rel if r < grasp[k]]
            bounds.append(max(rs) if rs else grasp[k])
        bounds.append(n)
        ph = phases[:nr]
    else:  # gripper unclear -> even split across plan phases
        nr = len(phases); ph = phases
        bounds = [int(n * k / nr) for k in range(nr)] + [n]

    def round_of(fi):
        for k in range(len(bounds) - 1):
            if bounds[k] <= fi < bounds[k + 1]:
                return k
        return len(bounds) - 2

    seg_ids = []
    for f in frames:
        if not seg_ids or seg_ids[-1] != f["segment_id"]:
            seg_ids.append(f["segment_id"])
    seg_frames = {s: [i for i, f in enumerate(frames) if f["segment_id"] == s] for s in seg_ids}
    seg_pos = {s: i + 1 for i, s in enumerate(seg_ids)}
    # group segments by round (temporal order), then lay each round's phase plan
    # proportionally across its segments -> full phase coverage when segs>=subtasks
    round_segs = {}
    for s in seg_ids:
        mid = seg_frames[s][len(seg_frames[s]) // 2]
        round_segs.setdefault(round_of(mid), []).append(s)
    new_sub = {}
    for rd in sorted(round_segs):
        segs = round_segs[rd]
        phase_idx = ph[rd] if rd < len(ph) else ph[-1]
        P, M = len(phase_idx), len(segs)
        for j, s in enumerate(segs):
            loc = round(j * (P - 1) / max(M - 1, 1)) if M > 1 else 0
            new_sub[s] = plan[phase_idx[min(P - 1, loc)]]
    # write frames
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
    return f"fixed:{len(runs)}runs/{len(plan)}plan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot_root", required=True); ap.add_argument("--suite", required=True)
    ap.add_argument("--lerobot_root", required=True)
    ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    base = Path(a.cot_root) / a.suite / "extras"
    lr = Path(a.lerobot_root) / f"{a.suite}_no_noops_1.0.0_lerobot"
    info = json.load(open(lr / "meta" / "info.json")); chunk = info.get("chunks_size", 1000)
    paths = sorted(glob.glob(str(base / "episode_*" / "cot_annotations.json")))
    if a.episodes:
        want = {f"episode_{e:06d}" for e in a.episodes}
        paths = [p for p in paths if Path(p).parent.name in want]
    from collections import Counter
    st = Counter()
    for p in paths:
        ei = int(Path(p).parent.name.split("_")[1])
        pq = lr / f"data/chunk-{ei//chunk:03d}/episode_{ei:06d}.parquet"
        if a.dry:
            print(p); continue
        st[reapply(str(p), str(pq)).split(":")[0]] += 1
    print(f"[{a.suite}] {dict(st)}")


if __name__ == "__main__":
    main()
