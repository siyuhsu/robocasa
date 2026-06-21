#!/usr/bin/env python3
"""Dump agentview camera params per instruction (VLA-Arena venv), level-aware.

The agentview camera varies by scene; an instruction can appear in multiple
bddls (different suite/level) with DIFFERENT cameras. We resolve candidates,
prefer the dataset's level, and dump ALL surviving candidates' camera params so
the grounding step can disambiguate by projection in-bounds. Output JSON:
  {instruction: [ {bddl, level, suite, cam_pos, cam_xmat, fovy, H, W}, ... ]}
Run with the VLA-Arena envs/base/.venv python.
"""
import sys, glob, re, json, argparse, math
from collections import Counter
sys.path.insert(0, "/ssd/sxu/workspace/sizhe/VLA-Arena")
import numpy as np
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena import get_vla_arena_path

Q = chr(34)


def norm(s):
    s = s.strip().strip(Q).strip().lower()
    return re.sub(r"\s+\d+$", "", s)


def level_of(bddl_rel):
    m = re.search(r"level_(\d)", bddl_rel)
    return int(m.group(1)) if m else -1


def build_lang2bddl():
    root = get_vla_arena_path("bddl_files")
    out = {}
    for b in glob.glob(root + "/**/*.bddl", recursive=True):
        m = re.search(r"\(:language\s+(.+?)\)", open(b).read(), re.S)
        if m:
            out.setdefault(norm(m.group(1)), []).append(b)
    return out


STOP_T = set("the a an and or on in to of it with up turned off is are".split())


def _toks(x):
    return set(w for w in re.findall(r"[a-z]+", x) if w not in STOP_T)


def fuzzy_cands(instr, lang2bddl):
    it = _toks(norm(instr))
    if not it:
        return []
    N = len(lang2bddl); df = Counter()
    for key in lang2bddl:
        for t in set(_toks(key)):
            df[t] += 1
    idf = {t: math.log(N / (1 + df[t])) + 1.0 for t in df}
    def w(ts):
        return sum(idf.get(t, 1.0) for t in ts)
    scored = []
    for key, bs in lang2bddl.items():
        kt = _toks(key)
        if not kt:
            continue
        score = w(it & kt) / (w(it | kt) or 1.0)
        scored.append((score, key, bs))
    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] >= 0.35:
        top = scored[0][0]
        return [b for sc, k, bs in scored if sc >= top - 0.10 for b in bs]
    return []


def dump_camera(bddl, res):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res,
                             camera_widths=res, camera_names=["agentview"])
    env.reset()
    sim = env.sim; cid = sim.model.camera_name2id("agentview")
    rel = bddl.split("bddl_files/")[1]
    rec = {"bddl": rel, "level": level_of(rel), "suite": rel.split("/")[0],
           "cam_pos": sim.model.cam_pos[cid].tolist(),
           "cam_xmat": sim.data.cam_xmat[cid].reshape(3, 3).tolist(),
           "fovy": float(sim.model.cam_fovy[cid]), "H": res, "W": res}
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-jsonl")
    ap.add_argument("--instructions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=None, help="dataset level (0 for L0, 1 for L1)")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    lang2bddl = build_lang2bddl()
    instrs = args.instructions or [json.loads(l)["task"] for l in open(args.tasks_jsonl)]
    out = {}
    if not args.overwrite:
        try: out = json.load(open(args.out))
        except Exception: pass
    for i, instr in enumerate(instrs):
        if instr in out and out[instr]:
            continue
        cands = lang2bddl.get(norm(instr), [])
        if not cands:
            cands = fuzzy_cands(instr, lang2bddl)
        if not cands:
            print(f"[warn] no bddl for {instr!r}", flush=True); continue
        # prefer dataset level; if none at that level, keep all
        if args.level is not None:
            lvl = [c for c in cands if level_of(c.split("bddl_files/")[1]) == args.level]
            if lvl:
                cands = lvl
        recs = []
        for b in cands:
            try:
                recs.append(dump_camera(b, args.resolution))
            except Exception as e:
                print(f"[err] {b.split('bddl_files/')[1]}: {repr(e)[:80]}", flush=True)
        if recs:
            out[instr] = recs
            json.dump(out, open(args.out, "w"), indent=2)
            tag = recs[0]["suite"] + f"/L{recs[0]['level']}" + (f" (+{len(recs)-1} alt)" if len(recs) > 1 else "")
            print(f"  [{i+1}/{len(instrs)}] {instr[:40]!r} -> {tag}", flush=True)
    print(f"[done] {len(out)} instructions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
