"""
Generate instruction → subtask-list mapping for LIBERO 4 suites.

For each unique LIBERO task instruction (10 per suite × 4 suites = 40 unique),
sample uniformly-spaced keyframes from one demo episode and ask Qwen3-VL to
decompose the task into a sequential list of high-level subtasks.

Output: scripts/libero_labeling/libero_instruction_subtask_mapping.json
  {
    "put the bowl on the plate": {
      "subtasks": ["approach the bowl", "grasp the bowl", "lift the bowl",
                   "move to the plate", "place the bowl on the plate"],
      "subtask_count": 5,
      "sample_episode": "libero_goal/episode_000000"
    },
    ...
  }

Usage:
  python scripts/libero_labeling/generate_libero_instruction_subtasks.py \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA \
      --api_url http://localhost:8101/v1/chat/completions \
      --model_name <abs_path_to_Qwen3-VL-4B-Instruct>

  # Dry-run (no VLM, just scan instructions):
  python scripts/libero_labeling/generate_libero_instruction_subtasks.py \
      --data_root playground/Datasets/LEROBOT_LIBERO_DATA --dry_run
"""

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_libero import (
    HFQwenVLClient,
    QwenVLLM,
    discover_libero_suites,
    load_episode_lengths,
    load_task_index_to_instruction,
    read_video_frames,
    video_path,
)


VLM_PLAN_PROMPT = """You are a robotics task planning expert. A single robotic arm in a tabletop kitchen environment completed the following LIBERO task:

Task instruction: "{instruction}"

Below are keyframes sampled uniformly from a demonstration video showing the robot executing this task from start to finish.

Based on the instruction and the visual demonstration, decompose this task into a sequential plan of high-level subtasks. Each subtask should describe one distinct phase of the robot's execution.

Guidelines:
1. Each subtask is a short imperative phrase (3-8 words), e.g. "approach the mug", "grasp the bowl", "move to the plate".
2. Subtasks must be sequential and cover the full task from start to finish.
3. Use specific action verbs: approach, reach, grasp, lift, move, carry, place, release, open, close, push, pull, slide, turn, press.
4. Include the relevant objects mentioned in the instruction (e.g. "the bowl", "the wine bottle", "the cabinet door").
5. Typical LIBERO tasks need 3-7 subtasks: simple open/close → 2-4, pick-and-place → 4-6, multi-stage (open + place) → 5-7.

Output ONLY a JSON list of subtask strings, nothing else. Example:
["approach the bowl", "grasp the bowl", "lift the bowl", "move to the plate", "place the bowl on the plate"]"""


def collect_unique_instructions(data_root: Path) -> dict[str, tuple[str, int]]:
    """
    Scan all 4 LIBERO suites; return {instruction: (suite_name, ep_idx)}
    keeping the FIRST suite/episode where the instruction appears (for sampling video).
    """
    instructions: dict[str, tuple[str, int]] = {}
    for suite_dir in discover_libero_suites(data_root):
        suite_name = suite_dir.name.replace("_no_noops_1.0.0_lerobot", "")
        task_map = load_task_index_to_instruction(suite_dir)
        eps = load_episode_lengths(suite_dir)
        # First episode for each unique instruction in this suite
        seen_in_suite = set()
        for ep_idx, _length, inst in eps:
            if not inst or inst in seen_in_suite:
                continue
            seen_in_suite.add(inst)
            if inst not in instructions:
                instructions[inst] = (suite_name, ep_idx)
        # Also include any task_map instructions that didn't appear in episodes (edge case)
        for inst in task_map.values():
            instructions.setdefault(inst, (suite_name, 0))
    return instructions


def sample_keyframes(suite_dir: Path, ep_idx: int, num_frames: int = 8) -> list[Image.Image]:
    """Decode the primary camera video and sample uniformly-spaced keyframes."""
    vp = video_path(suite_dir, ep_idx, "observation.images.image")
    if not vp.exists():
        raise FileNotFoundError(vp)
    frames = read_video_frames(vp)
    if not frames:
        raise RuntimeError(f"no frames decoded from {vp}")
    n = len(frames)
    indices = np.linspace(0, n - 1, num=min(num_frames, n), dtype=int)
    return [Image.fromarray(frames[i], "RGB") for i in indices]


def parse_vlm_subtasks(text: str) -> list[str] | None:
    """Extract a JSON list of subtask strings from VLM output."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # Try to find a JSON list
    m = re.search(r"\[[\s\S]*?\]", text)
    if m is None:
        return None
    try:
        parsed = ast.literal_eval(m.group(0))
    except Exception:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(parsed, list):
        return None
    out = [str(x).strip() for x in parsed if isinstance(x, (str,)) and str(x).strip()]
    return out or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to LEROBOT_LIBERO_DATA/")
    p.add_argument("--output", type=str,
                   default=str(Path(__file__).resolve().parent / "libero_instruction_subtask_mapping.json"))
    p.add_argument("--vlm_backend", choices=["http", "hf_local"], default="hf_local",
                   help="http=requests OpenAI-compat (vLLM server); hf_local=in-process transformers (no server)")
    p.add_argument("--api_url", type=str, default="http://localhost:8101/v1/chat/completions",
                   help="Used when --vlm_backend http")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                   help="vLLM registered name (http) or absolute model path (hf_local)")
    p.add_argument("--device", type=str, default="cuda:0", help="Used when --vlm_backend hf_local")
    p.add_argument("--num_keyframes", type=int, default=8)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-query VLM even for instructions already in output JSON")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()

    # Resume-aware: load existing mapping if present
    existing = {}
    if output_path.exists() and not args.overwrite:
        with open(output_path) as f:
            existing = json.load(f)
        print(f"[info] Loaded {len(existing)} existing entries from {output_path}")

    # 1. Scan all unique instructions
    instructions = collect_unique_instructions(data_root)
    print(f"[info] Found {len(instructions)} unique LIBERO instructions across all suites")

    if args.dry_run:
        for inst, (suite, ep) in sorted(instructions.items()):
            mark = "✓" if inst in existing else " "
            print(f"  [{mark}] [{suite} ep{ep:03d}] {inst}")
        return

    # 2. VLM session
    if args.vlm_backend == "http":
        model = QwenVLLM(api_url=args.api_url, model_name=args.model_name)
    else:
        model = HFQwenVLClient(model_path=args.model_name, device=args.device)

    # 3. Iterate
    out = dict(existing)
    pending = [inst for inst in instructions if inst not in out]
    if not pending:
        print("[info] All instructions already mapped; use --overwrite to re-generate")
        return

    for inst in tqdm(pending, desc="VLM subtask decomposition"):
        suite_name, ep_idx = instructions[inst]
        suite_dir = data_root / f"{suite_name}_no_noops_1.0.0_lerobot"
        try:
            keyframes = sample_keyframes(suite_dir, ep_idx, args.num_keyframes)
        except Exception as e:
            tqdm.write(f"[warn] '{inst[:50]}': {e}")
            continue

        prompt = VLM_PLAN_PROMPT.format(instruction=inst)
        subtasks = None
        for retry in range(args.max_retries):
            try:
                resp = model.generate_content([prompt, *keyframes], max_tokens=512)
                subtasks = parse_vlm_subtasks(resp.text)
                if subtasks:
                    break
            except Exception as e:
                tqdm.write(f"[retry {retry+1}/{args.max_retries}] '{inst[:40]}': {e}")
                time.sleep(min(5 * (retry + 1), 30))

        if not subtasks:
            tqdm.write(f"[fail] '{inst[:50]}' — no valid subtask list after {args.max_retries} retries")
            continue

        out[inst] = {
            "subtasks": subtasks,
            "subtask_count": len(subtasks),
            "sample_episode": f"{suite_name}/episode_{ep_idx:06d}",
        }

        # Periodic save (every 5 instructions)
        if len(out) % 5 == 0 or inst == pending[-1]:
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] {len(out)} instruction → subtasks saved to {output_path}")


if __name__ == "__main__":
    main()
