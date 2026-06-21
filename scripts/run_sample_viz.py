#!/usr/bin/env python3
"""Render one full-COT review video PER TASK for libero + VLA-Arena (viz only,
no VLM/sim) to verify the Task-A per-segment-reasoning re-correction. Uses the
existing cots (subtask-merged + reason-per-segment + gripper_2d + bbox baked in)
via visualize_vla_arena_full_cot.py."""
import json, subprocess, argparse
from pathlib import Path

SCR = "/ssd/sxu/workspace/code/robocasa/scripts"
VIZ = f"{SCR}/vla_arena_labeling/visualize_vla_arena_full_cot.py"
PY = "/ssd/sxu/miniconda3/envs/starVLA/bin/python"
LIB_DATA = "/ssd/sxu/workspace/code/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA"
LIB_COT = "/ssd/sxu/workspace/sizhe/starVLA/playground/Datasets/LIBERO_COT_LNDS"
VA_DATA = "/ssd/sxu/workspace/sizhe/datasets/vla_arena_libero_cot"
VA_COT = "/ssd/sxu/workspace/sizhe/starVLA/playground/Datasets/VLA_ARENA_COT"

SUITES = [
    ("libero_object", LIB_DATA, LIB_COT, 20),
    ("libero_goal", LIB_DATA, LIB_COT, 20),
    ("libero_spatial", LIB_DATA, LIB_COT, 20),
    ("libero_10", LIB_DATA, LIB_COT, 20),
    ("vla_arena_l0", VA_DATA, VA_COT, 10),
    ("vla_arena_l1", VA_DATA, VA_COT, 10),
]


def one_per_task(lerobot_dir):
    seen = {}
    ep = Path(lerobot_dir) / "meta" / "episodes.jsonl"
    for l in open(ep):
        r = json.loads(l); instr = (r.get("tasks") or [""])[0]
        if instr and instr not in seen:
            seen[instr] = r["episode_index"]
    return sorted(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/ssd/sxu/workspace/sizhe/datasets/taskA_sample_viz")
    ap.add_argument("--suites", nargs="*")
    a = ap.parse_args()
    suites = [s for s in SUITES if (not a.suites or s[0] in a.suites)]
    for suite, data, cot, fps in suites:
        lr = Path(data) / f"{suite}_no_noops_1.0.0_lerobot"
        eps = one_per_task(lr)
        comma = ",".join(map(str, eps))
        outdir = Path(a.out) / suite
        print(f"[{suite}] {len(eps)} tasks -> eps {eps[:6]}{'...' if len(eps)>6 else ''}", flush=True)
        cmd = [PY, VIZ, "--cot_root", cot, "--data_root", data, "--output", str(outdir),
               "--suite", suite, "--episodes", comma, "--fps", str(fps)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(list(outdir.glob("*.mp4"))) if outdir.exists() else 0
        print(f"[{suite}] DONE videos={n} {'' if r.returncode==0 else 'ERR:'+r.stderr[-200:]}", flush=True)


if __name__ == "__main__":
    main()
