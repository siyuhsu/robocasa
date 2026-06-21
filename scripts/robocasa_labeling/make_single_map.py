#!/usr/bin/env python3
"""Inline Stage-1 for end-to-end single-demo annotation: decompose the given
episodes' instruction(s) into a subtask plan and write a (small) instruction→subtask
mapping JSON. Lets LIBERO/VLA-Arena annotate a NEW task without a pre-built map
(mirrors RoboCasa's inline Stage-1). Run in the starVLA env with a VLM endpoint."""
import sys, os, json, argparse
from pathlib import Path
SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
from utils import QwenVLLM
from generate_instruction_subtasks import decompose_instruction_vlm, extract_keyframes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--video_key", default="observation.images.image")
    ap.add_argument("--api_url", required=True)
    ap.add_argument("--model_name", default="Qwen3.5-9B")
    a = ap.parse_args()
    lr = Path(a.data_root) / f"{a.suite}_no_noops_1.0.0_lerobot"
    eps = [json.loads(l) for l in open(lr / "meta" / "episodes.jsonl")]
    model = QwenVLLM(api_url=a.api_url, model_name=a.model_name,
                     chat_template_kwargs={"enable_thinking": False})
    # merge into existing map if present (so repeated calls accumulate)
    m = json.load(open(a.out_map)) if os.path.exists(a.out_map) else {}
    for ep in a.episodes:
        instr = eps[ep]["tasks"][0]
        if instr in m:
            continue
        vid = lr / "videos" / f"chunk-{ep // 1000:03d}" / a.video_key / f"episode_{ep:06d}.mp4"
        kf = extract_keyframes(vid, 1.0)
        subs = decompose_instruction_vlm(instr, kf, model) or ["approach", "interact", "complete"]
        # canonical mapping schema (matches generate_*_instruction_subtasks output):
        # {instruction: {"task_type", "category", "subtasks": [...]}}
        m[instr] = {"task_type": a.suite, "category": "atomic", "subtasks": subs}
        print(f"  [{instr[:50]}] -> {subs}", flush=True)
    Path(a.out_map).parent.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(a.out_map, "w"), ensure_ascii=False, indent=2)
    print(f"wrote {len(m)} instruction(s) -> {a.out_map}", flush=True)


if __name__ == "__main__":
    main()
