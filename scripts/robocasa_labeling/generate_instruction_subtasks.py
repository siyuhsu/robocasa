"""
Generate instruction → subtask (plan) mapping for all RoboCasa tasks.

Inspired by ECoT (Embodied Chain-of-Thought): for each unique task
instruction, extract video keyframes from a sample episode and use
Qwen VLM to decompose the task into sequential subtask phases.

Steps:
  1. Scan all episodes.jsonl across datasets to collect unique instructions
  2. For each unique instruction, find a sample episode video
  3. Extract uniformly-spaced keyframes from the video
  4. Send keyframes + instruction to Qwen VLM for plan generation
  5. Save instruction → subtask list mapping

Output:
    scripts/robocasa_labeling/instruction_subtask_mapping.json

Usage:
    # Dry-run (no VLM, scan only):
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_instruction_subtasks.py \
        --datasets_root datasets/v1.0 --dry_run

    # Full run with VLM:
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_instruction_subtasks.py \
        --datasets_root datasets/v1.0 \
        --api_url http://localhost:8100/v1/chat/completions

    # Process a limited number for testing:
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_instruction_subtasks.py \
        --datasets_root datasets/v1.0 --max_instructions 10

    # Only label specific subsets (default):
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_instruction_subtasks.py \
        --datasets_root datasets/v1.0 \
        --subsets target_no_nav pretrain_human300_no_nav
"""

import argparse
import json
import re
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import QwenVLLM


# ─── VLM prompt ──────────────────────────────────────────────────────────────

VLM_PLAN_PROMPT = """You are a robotics task planning expert. A single robotic arm in a kitchen environment completed the following task:

Task instruction: "{instruction}"

Below are keyframes sampled uniformly from a demonstration video showing the robot executing this task from start to finish.

Based on the instruction and the visual demonstration, decompose this task into a sequential plan of high-level subtasks. Each subtask should describe one distinct phase of the robot's execution.

Guidelines:
1. Each subtask is a short imperative phrase (3-8 words), e.g. "approach the mug", "grasp the handle", "move to the shelf".
2. Subtasks must be sequential and cover the full task from start to finish.
3. Use specific action verbs: approach, reach, grasp, lift, move, carry, place, release, open, close, turn, press, slide, push, pull, etc.
4. Include the relevant objects mentioned in the instruction (e.g. "the mug", "the cabinet door").
5. Typical tasks require 3-8 subtasks. Simple open/close tasks need 2-4, pick-and-place need 5-7, composite multi-object tasks need 6-10.

Output ONLY a JSON list of subtask strings, nothing else. Example:
["approach the watermelon", "grasp the watermelon", "lift the watermelon", "move to the towel", "place the watermelon on the towel"]"""


# ─── scan all unique instructions ────────────────────────────────────────────

def resolve_subset_paths(subsets: list[str]) -> set[str]:
    """
    Resolve a list of DATASET_SOUP_REGISTRY subset names to a set of
    canonical lerobot directory paths (resolved to real paths where possible).
    """
    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

    allowed: set[str] = set()
    for name in subsets:
        if name not in DATASET_SOUP_REGISTRY:
            raise ValueError(
                f"Unknown subset '{name}'. "
                f"Available: {sorted(DATASET_SOUP_REGISTRY.keys())}"
            )
        for ds_meta in DATASET_SOUP_REGISTRY[name]:
            p = ds_meta.get("path")
            if p:
                # Normalise so comparisons work regardless of trailing slash etc.
                allowed.add(str(Path(p).resolve()))
    return allowed


def scan_instructions(
    datasets_root: Path,
    allowed_paths: set[str] | None = None,
) -> dict[str, dict]:
    """
    Scan episodes.jsonl files and collect unique instructions.

    Args:
        datasets_root: Root directory to search recursively.
        allowed_paths: If provided, only process lerobot directories whose
            resolved path is in this set.  Pass the output of
            ``resolve_subset_paths`` to restrict scanning to specific subsets.

    Returns dict keyed by unique instruction string:
        {
            "task_type": str,
            "category": "atomic" | "composite",
            "split": str,
            "sample_episode": int,
            "dataset_path": str,  # path to lerobot dir
            "episode_length": int,
        }
    """
    instructions: dict[str, dict] = {}

    for ep_file in sorted(datasets_root.rglob("meta/episodes.jsonl")):
        lerobot_dir = ep_file.parent.parent

        # Filter to allowed subsets when requested
        if allowed_paths is not None:
            if str(lerobot_dir.resolve()) not in allowed_paths:
                continue

        task_dir = lerobot_dir.parent.parent
        task_type = task_dir.name
        category = task_dir.parent.name   # atomic / composite
        split = task_dir.parent.parent.name  # pretrain / target

        with open(ep_file) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                ep_idx = d["episode_index"]
                ep_len = d.get("length", 0)
                for t in d.get("tasks", []):
                    t = t.strip()
                    # Skip empty or bare task-type names (e.g. "AddIceCubes")
                    if not t or t == task_type:
                        continue
                    if t not in instructions:
                        # Pick the first episode we encounter for this instruction
                        instructions[t] = {
                            "task_type": task_type,
                            "category": category,
                            "split": split,
                            "sample_episode": ep_idx,
                            "dataset_path": str(lerobot_dir),
                            "episode_length": ep_len,
                        }

    return instructions


# ─── video keyframe extraction ───────────────────────────────────────────────

def get_video_path(dataset_path: str, ep_idx: int,
                   camera: str = "robot0_agentview_left") -> Path | None:
    """Locate the MP4 video file for a given episode."""
    ds = Path(dataset_path)
    for chunk_dir in sorted(ds.glob("videos/chunk-*")):
        video = chunk_dir / f"observation.images.{camera}" / f"episode_{ep_idx:06d}.mp4"
        if video.exists():
            return video
    return None


def extract_keyframes(video_path: Path, fps: float = 1.0) -> list[Image.Image]:
    """Extract keyframes at a fixed rate (fps frames per second) from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    if total <= 0:
        cap.release()
        return []

    step = max(1, int(round(video_fps / fps)))
    indices = np.arange(0, total, step, dtype=int)
    frames: list[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
    cap.release()

    # # Debug: save all keyframes to /tmp for color-channel inspection
    # if frames:
    #     import os
    #     debug_dir = Path("./tmp/keyframe_debug")
    #     debug_dir.mkdir(exist_ok=True)
    #     stem = video_path.stem  # e.g. "episode_000000"
    #     for fi, f in enumerate(frames):
    #         out = debug_dir / f"{stem}_frame{fi:02d}.png"
    #         f.save(out)
    #     print(f"  [debug] saved {len(frames)} keyframes to {debug_dir}/")

    return frames


# ─── prefetch worker ─────────────────────────────────────────────────────────

def _load_keyframes_worker(
    info: dict, camera: str, keyframe_fps: float
) -> tuple[Path | None, list, float, float]:
    """Thread worker: locate video and decode keyframes. Returns (path, frames, t_locate, t_extract)."""
    t0 = time.perf_counter()
    video_path = get_video_path(info["dataset_path"], info["sample_episode"], camera)
    t1 = time.perf_counter()
    if video_path is None:
        return None, [], t1 - t0, 0.0
    keyframes = extract_keyframes(video_path, keyframe_fps)
    t2 = time.perf_counter()
    return video_path, keyframes, t1 - t0, t2 - t1


# ─── VLM-based subtask decomposition ─────────────────────────────────────────

def decompose_instruction_vlm(
    instruction: str,
    keyframes: list[Image.Image],
    model: QwenVLLM,
    max_retries: int = 3,
) -> list[str] | None:
    """
    Send keyframes + instruction to VLM and parse the subtask plan.
    Returns a list of subtask strings, or None on failure.
    """
    prompt = VLM_PLAN_PROMPT.format(instruction=instruction)
    content = [prompt] + keyframes

    for attempt in range(max_retries):
        try:
            response = model.generate_content(content, max_tokens=1024)
            text = response.text
            # Extract the JSON list from the response
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                subtasks = json.loads(match.group())
                if isinstance(subtasks, list) and len(subtasks) >= 1 and all(isinstance(s, str) for s in subtasks):
                    return subtasks
        except Exception as e:
            print(f"    [warn] VLM attempt {attempt + 1} failed: {e}")

    return None


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate instruction → subtask mapping using VLM + video keyframes"
    )
    parser.add_argument("--datasets_root", type=str, default="/tmp/sx0401/workspace/datasets/v1.0/",
                        help="Root dir containing pretrain/ and target/")
    parser.add_argument("--output", type=str,
                        default="scripts/robocasa_labeling/instruction_subtask_mapping.json",
                        help="Output JSON path")
    parser.add_argument("--keyframe_fps", type=float, default=1.0,
                        help="Keyframe sampling rate in frames per second (default: 1 fps)")
    parser.add_argument("--camera", type=str, default="robot0_agentview_left",
                        help="Camera view to extract keyframes from")
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8002/v1/chat/completions")
    parser.add_argument("--model_name", type=str,
                        default="/jobfs/163093094.gadi-pbs/models/models--Qwen--Qwen3-VL-8B-Instruct"
                                "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--dry_run", action="store_true",
                        help="Scan only, skip VLM calls")
    parser.add_argument("--max_instructions", type=int, default=0,
                        help="Max instructions to process (0 = all)")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="Number of videos to prefetch in background threads "
                             "while the main thread runs VLM inference (default: 4)")
    parser.add_argument(
        "--subsets", nargs="+",
        default=["target_no_nav", "pretrain_human300_no_navigation"],
        help="DATASET_SOUP_REGISTRY subset names to label. "
             "Pass an empty string '' to scan all datasets without filtering.",
    )
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve subset filter
    allowed_paths: set[str] | None = None
    if args.subsets and args.subsets != [""]:
        allowed_paths = resolve_subset_paths(args.subsets)
        print(f"[info] Filtering to subsets: {args.subsets} "
              f"({len(allowed_paths)} dataset paths)")

    # Step 1: scan all unique instructions
    print(f"[info] Scanning datasets under {datasets_root} ...")
    instructions = scan_instructions(datasets_root, allowed_paths=allowed_paths)
    print(f"[info] Found {len(instructions)} unique task instructions")

    # Show distribution by task_type
    type_counts: dict[str, int] = {}
    for info in instructions.values():
        tt = info["task_type"]
        type_counts[tt] = type_counts.get(tt, 0) + 1
    print(f"[info] Across {len(type_counts)} task types")
    top5 = sorted(type_counts.items(), key=lambda x: -x[1])[:5]
    for tt, cnt in top5:
        print(f"  {tt}: {cnt} instructions")

    # Step 2: load checkpoint (resume support)
    mapping: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            mapping = json.load(f)
        print(f"[info] Loaded {len(mapping)} existing mappings (checkpoint)")

    # Step 3: init VLM
    model = None
    if not args.dry_run:
        model = QwenVLLM(api_url=args.api_url, model_name=args.model_name)
    else:
        print("[info] Dry-run mode — VLM calls will be skipped")

    # Step 4: process each unique instruction
    sorted_instrs = sorted(instructions.keys())
    if args.max_instructions > 0:
        sorted_instrs = sorted_instrs[:args.max_instructions]

    # Separate already-done entries so prefetch only covers actual work
    todo: list[str] = [instr for instr in sorted_instrs if instr not in mapping]
    skipped = len(sorted_instrs) - len(todo)
    total_todo = len(todo)
    success, failed = 0, 0

    print(f"[info] prefetch={args.prefetch} threads  todo={total_todo}  skipped(existing)={skipped}")

    # ── prefetch pipeline ────────────────────────────────────────────────────
    # While the main thread calls the VLM (slow network I/O), background
    # threads decode the next `prefetch` videos so video loading is free.
    with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
        pending: deque[tuple[str, Future]] = deque()
        submit_ptr = 0

        def _submit(instr: str) -> None:
            fut = pool.submit(
                _load_keyframes_worker,
                instructions[instr], args.camera, args.keyframe_fps,
            )
            pending.append((instr, fut))

        # Seed the prefetch window
        for instr in todo[:args.prefetch]:
            _submit(instr)
        submit_ptr = min(args.prefetch, total_todo)

        pbar = tqdm(total=total_todo, desc="VLM labeling", unit="instr", dynamic_ncols=True)

        for j in range(total_todo):
            instr, fut = pending.popleft()

            # Keep the window full: submit the next instruction immediately
            if submit_ptr < total_todo:
                _submit(todo[submit_ptr])
                submit_ptr += 1

            # Retrieve result (blocks only if not yet ready)
            t_wait0 = time.perf_counter()
            video_path, keyframes, t_locate, t_extract = fut.result()
            t_wait = time.perf_counter() - t_wait0

            if video_path is None:
                tqdm.write(f"  [{j+1}/{total_todo}] [skip] no video: {instr[:60]}...")
                failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=failed)
                continue

            if not keyframes:
                tqdm.write(f"  [{j+1}/{total_todo}] [skip] no frames: {instr[:60]}...")
                failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=failed)
                continue

            # VLM decomposition
            if args.dry_run:
                subtasks = None
            else:
                subtasks = decompose_instruction_vlm(instr, keyframes, model)

            mapping[instr] = {
                "task_type": instructions[instr]["task_type"],
                "category": instructions[instr]["category"],
                "subtasks": subtasks or [],
                "subtask_count": len(subtasks) if subtasks else 0,
            }
            success += 1

            pbar.set_postfix(ok=success, fail=failed, subtasks=len(subtasks) if subtasks else 0)
            pbar.update(1)

            # Periodic checkpoint save
            if (j + 1) % 50 == 0:
                with open(output_path, "w") as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)
                tqdm.write(f"  [checkpoint] {len(mapping)} entries saved")

    # Final save
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Saved {len(mapping)} instruction mappings to {output_path}")
    print(f"  new={success}, skipped(already_done)={skipped}, failed={failed}")
    if mapping:
        counts = [m["subtask_count"] for m in mapping.values() if m["subtask_count"] > 0]
        if counts:
            print(f"  Subtask counts: min={min(counts)}, max={max(counts)}, "
                  f"mean={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    main()
