#!/usr/bin/env python3
"""Trajectory+grounding multi-object subtask correction (offline, no VLM).

For genuine multi-object / long-horizon episodes, a text-based LNDS over the
candidate-plan order is unsound: it imposes the plan's object order (not the
real execution order) and inherits the VLM's object-color guess. This corrector
instead grounds everything in the trajectory:

  - ROUNDS  = gripper open->close->open cycles (each = one legit pick-place).
  - OBJECT  per round = the object whose bbox the gripper_2d is inside at the
            round's grasp frame (grounding truth, not the VLM's color guess).
  - PHASE   per segment = verb classified from the VLM label; within a round the
            phase is forced monotonic (LNDS over approach<grasp<lift<move<place
            <release); repeats are allowed ACROSS rounds, in execution order.
  - subtask = templated from (phase, grounded object, VLM destination).

Single-object episodes (1 round) reduce to plain within-round monotonic — same
as pure LNDS — so this is safe to run on everything, but we gate on >1 round.

Runs on a cot_annotations.json that already has grounding backfilled.
"""
import argparse, json, re, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROBOT_SUBSTR = ("Panda", "Mount", "Gripper", "Rethink", "OnTheGround", "robot", "table", "Table")
PHASE_ORDER = {"approach": 0, "grasp": 1, "lift": 2, "move": 3, "place": 4, "release": 5}


def classify_phase(label):
    l = (label or "").lower()
    if any(w in l for w in ("release", "let go", "open the gripper")):
        return "release"
    if any(w in l for w in ("place", "put ", "insert", "deposit", "drop", "set down", "lower")):
        return "place"
    if any(w in l for w in ("push", "slide")):
        return "push"
    if any(w in l for w in ("lift", "raise", "pick up and lift")):
        return "lift"
    if any(w in l for w in ("grasp", "grip", "pick up", "pick the", "close the gripper")):
        return "grasp"
    if any(w in l for w in ("approach", "reach")):
        return "approach"
    if any(w in l for w in ("move", "carry", "transport", "bring")):
        return "move"
    return None


def parse_dest(label):
    m = re.search(r"\b(?:to|on|onto|in|into|toward|towards)\s+the\s+(.+?)\s*$", (label or "").strip(), re.I)
    return m.group(1).strip().rstrip(".") if m else None


def sim2human(name):
    return re.sub(r"_\d+$", "", name).replace("_", " ").strip()


def parse_objs(txt):
    out = {}
    for ln in (txt or "").splitlines():
        m = re.match(r"\s*([\w]+):\s*\[(\d+),(\d+)\],\s*\[(\d+),(\d+)\]", ln)
        if m:
            out[m.group(1)] = tuple(map(int, m.group(2, 3, 4, 5)))
    return out


def is_robot(n):
    return any(s in n for s in ROBOT_SUBSTR)


def gripper_events(parquet):
    df = pd.read_parquet(parquet)
    st = np.stack(df["observation.state"].to_numpy()).astype(float)
    gap = np.abs(st[:, 6] - st[:, 7]) if st.shape[1] >= 8 else np.abs(st[:, -1])
    lo, hi = gap.min(), gap.max()
    if hi - lo < 1e-6:
        return [], [], gap
    closed = gap < (lo + hi) / 2.0
    d = np.diff(closed.astype(int))
    grasp_f = list(int(x) for x in np.where(d == 1)[0] + 1)    # open->close
    release_f = list(int(x) for x in np.where(d == -1)[0] + 1)  # close->open
    return grasp_f, release_f, gap


def obj_at(frame_idx, frames):
    fr = min(frames, key=lambda f: abs(f["frame_index"] - frame_idx))
    g2d = fr.get("gripper_2d"); objs = parse_objs(fr.get("assistant_object_level", ""))
    objs = {n: b for n, b in objs.items() if not is_robot(n)}
    if not (g2d and objs):
        return None
    inside = [n for n, b in objs.items() if b[0] - 8 <= g2d[0] <= b[2] + 8 and b[1] - 8 <= g2d[1] <= b[3] + 8]
    if inside:  # smallest containing bbox = the held object
        return min(inside, key=lambda n: (objs[n][2] - objs[n][0]) * (objs[n][3] - objs[n][1]))
    return min(objs, key=lambda n: abs((objs[n][0] + objs[n][2]) / 2 - g2d[0]) + abs((objs[n][1] + objs[n][3]) / 2 - g2d[1]))


def lnds_phase(idxs, weights):
    n = len(idxs)
    valid = [i for i in range(n) if idxs[i] is not None]
    if not valid:
        return set(range(n))
    best = {i: weights[i] for i in valid}
    par = {i: -1 for i in valid}
    for a, i in enumerate(valid):
        for j in valid[:a]:
            if idxs[j] <= idxs[i] and best[j] + weights[i] > best[i]:
                best[i] = best[j] + weights[i]; par[i] = j
    end = max(valid, key=lambda i: best[i]); bb = set(); k = end
    while k != -1:
        bb.add(k); k = par[k]
    return bb


def make_subtask(phase, obj, dest):
    if phase == "approach":
        return f"approach the {obj}"
    if phase == "grasp":
        return f"grasp the {obj}"
    if phase == "lift":
        return f"lift the {obj}"
    if phase == "move":
        return f"move to the {dest}" if dest else f"move the {obj}"
    if phase == "place":
        return f"place the {obj} on the {dest}" if dest else f"place the {obj}"
    if phase == "release":
        return f"release the {obj}"
    return None


def fix_one(cot_path, parquet, dry=False, verbose=False):
    c = json.load(open(cot_path))
    frames = c["frames"]

    # OBJECT-MOTION rounds: the grasped object is the one that MOVES with the
    # gripper (static plates/boxes/tables don't move). Gripper cycles are
    # unreliable (the gripper may barely open between grasps), so we segment by
    # which object is being carried. Only fire when >=2 objects clearly move.
    objpos = {}   # name -> [(frame_index, cx, cy)]
    for f in frames:
        for n, b in parse_objs(f.get("assistant_object_level", "")).items():
            if is_robot(n):
                continue
            objpos.setdefault(n, []).append((f["frame_index"], (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
    MIN_MOTION = 60.0
    moving = {}   # name -> velocity-weighted centroid frame (robust carry time)
    for n, pts in objpos.items():
        if len(pts) < 3:
            continue
        a = np.array(pts); fr = a[:, 0]; vel = np.r_[0.0, np.abs(np.diff(a[:, 1:], axis=0)).sum(1)]
        if vel.sum() < MIN_MOTION:
            continue
        moving[n] = float((fr * vel).sum() / max(vel.sum(), 1e-6))   # when it's actively carried
    order = sorted(moving, key=lambda n: moving[n])   # by carry centroid (blip-robust)
    if len(order) < 2:
        return ("single_round", None)   # <2 moving objects; leave to pure LNDS
    rounds = [(moving[n], n) for n in order]

    # round boundary = midpoint between consecutive carry centroids
    boundaries = [0]
    for k in range(1, len(order)):
        boundaries.append(int((moving[order[k - 1]] + moving[order[k]]) / 2))
    boundaries.append(len(frames) + max((f["frame_index"] for f in frames), default=0))
    bounds = [(boundaries[i], boundaries[i + 1]) for i in range(len(order))]

    def round_of(fidx):
        for r, (a, b) in enumerate(bounds):
            if a <= fidx < b:
                return r
        return len(bounds) - 1

    round_obj = {i: sim2human(order[i]) for i in range(len(order))}

    # group segments by round, classify phase, enforce within-round monotonic
    seg_ids = []
    for f in frames:
        if not seg_ids or seg_ids[-1] != f["segment_id"]:
            seg_ids.append(f["segment_id"])
    seg_frames = {s: [f for f in frames if f["segment_id"] == s] for s in seg_ids}
    seg_round = {s: round_of(int(np.median([f["frame_index"] for f in seg_frames[s]]))) for s in seg_ids}
    seg_phase = {}
    raw = {int(k): v for k, v in (c.get("segment_labels_raw") or {}).items()}
    # map seg_id -> exec position (1-based) for raw lookup
    seg_pos = {s: i + 1 for i, s in enumerate(seg_ids)}
    for s in seg_ids:
        lbl = raw.get(seg_pos[s], [None])[0] if raw else seg_frames[s][0].get("subtask")
        seg_phase[s] = classify_phase(lbl)

    new_label = {}
    for r in range(len(bounds)):
        rsegs = [s for s in seg_ids if seg_round[s] == r]
        if not rsegs:
            continue
        idxs = [PHASE_ORDER.get(seg_phase[s]) for s in rsegs]
        wts = [len(seg_frames[s]) for s in rsegs]
        bb = lnds_phase(idxs, wts)
        cur = None
        for i, s in enumerate(rsegs):
            if i in bb and seg_phase[s] is not None:
                cur = seg_phase[s]
            ph = cur if cur is not None else seg_phase[s]
            obj = round_obj[r]
            dest = parse_dest(raw.get(seg_pos[s], [None])[0] if raw else seg_frames[s][0].get("subtask"))
            st = make_subtask(ph, obj, dest)
            new_label[s] = st or (seg_frames[s][0].get("subtask"))

    # rewrite frames
    for f in frames:
        st = new_label.get(f["segment_id"])
        if st:
            f["subtask"] = st
            f["assistant_short_plan"] = f"SHORT PLAN:\nSubtask: {st}\nSubtask Reasoning: {f.get('reason','')}\n"
    # long plan
    items, prev = [], None
    for f in frames:
        if f["subtask"] != prev:
            items.append(f["subtask"]); prev = f["subtask"]
    lp = "LONG PLAN:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(items)) + "\n"
    for f in frames:
        f["assistant_plan_level"] = lp

    if verbose:
        print(f"  rounds={len(rounds)} objs={[round_obj[r] for r in range(len(bounds))]}")
        print("  LONG PLAN:"); print("   " + lp.replace("\n", "\n   "))
    if not dry:
        json.dump(c, open(cot_path, "w"), ensure_ascii=False, indent=2)
    return ("fixed", len(rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", help="single cot_annotations.json")
    ap.add_argument("--cot_root"); ap.add_argument("--suite"); ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--lerobot", required=True, help="lerobot dir (for parquet gripper)")
    ap.add_argument("--dry", action="store_true"); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    info = json.load(open(Path(a.lerobot) / "meta" / "info.json")); chunk = info.get("chunks_size", 1000)

    def pq(ei):
        return Path(a.lerobot) / f"data/chunk-{ei//chunk:03d}/episode_{ei:06d}.parquet"

    if a.cot:
        ei = int(Path(a.cot).parent.name.split("_")[1])
        print(a.cot, fix_one(a.cot, pq(ei), a.dry, True)); return
    base = Path(a.cot_root) / a.suite / "extras"
    paths = sorted(glob.glob(str(base / "episode_*/cot_annotations.json")))
    if a.episodes:
        want = {f"episode_{e:06d}" for e in a.episodes}
        paths = [p for p in paths if Path(p).parent.name in want]
    stats = {}
    for p in paths:
        ei = int(Path(p).parent.name.split("_")[1])
        r, n = fix_one(p, pq(ei), a.dry, a.verbose)
        stats[r] = stats.get(r, 0) + 1
        if a.verbose:
            print(f"ep{ei}: {r} rounds={n}")
    print("[multiobj]", stats)


if __name__ == "__main__":
    main()
