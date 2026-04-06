"""
Generate per-frame subtask annotations for RoboCasa episodes.

Uses pre-generated instruction→subtask mapping
(instruction_subtask_mapping.json) to get the subtask list and count.
Triple segmentation (spatial + orient + gripper, Emma-X approach)
is used to split the trajectory, then the VLM selects from the
predefined subtask list for each segment.  Finally segment-level
labels are propagated to every frame.

Output per episode: a list of [subtask, reason] pairs, one per frame.
With --visualize, also renders debug MP4 videos.

Usage:
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_subtasks.py \
        --dataset datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
        --output  datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/subtasks \
        --max_episodes 5

    # With visualization (end-to-end debug)
    conda run -n siyu_robocasa python scripts/robocasa_labeling/generate_subtasks.py \
        --dataset ... --output ... --max_episodes 3 \
        --visualize --viz_output tmp/subtask_viz --dry_run
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import QwenVLLM, triple_segment, get_key_frames_per_segment


# ─── LeRobot data loading ────────────────────────────────────────────────────

def load_lerobot_dataset(dataset_path: Path):
    """Load a LeRobotDataset from a local directory.

    Reference: https://robocasa.ai/docs/build/html/datasets/using_datasets.html
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="robocasa365", root=str(dataset_path))
    return ds


def load_dataset_meta(dataset_path: Path) -> tuple[dict, int]:
    """Load modality.json and info.json to get state layout and fps.

    Returns:
        modality_dict: parsed modality.json
        fps: frames per second from info.json
    """
    modality_path = dataset_path / "meta" / "modality.json"
    with open(modality_path) as f:
        modality_dict = json.load(f)

    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps", 20)

    return modality_dict, fps


def resolve_subset_paths(subsets: list[str]) -> set[str]:
    """
    Resolve DATASET_SOUP_REGISTRY subset names to canonical lerobot paths.
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
                allowed.add(str(Path(p).resolve()))
    return allowed


def discover_lerobot_datasets(
    datasets_root: Path,
    allowed_paths: set[str] | None = None,
) -> list[Path]:
    """
    Discover all lerobot dataset directories under datasets_root.
    If allowed_paths is given, keep only datasets in that resolved-path set.
    """
    discovered: list[Path] = []
    seen: set[str] = set()

    for ep_file in sorted(datasets_root.rglob("meta/episodes.jsonl")):
        lerobot_dir = ep_file.parent.parent
        resolved = str(lerobot_dir.resolve())
        if allowed_paths is not None and resolved not in allowed_paths:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        discovered.append(lerobot_dir)

    return discovered


def get_segmentation_indices(modality_dict: dict) -> tuple[list[int], list[int], list[int]]:
    """Build column indices for triple segmentation from modality state definitions.

    Returns three sets of indices for the Emma-X triple-segmentation:
      - spatial_indices: eef position (for HDBSCAN on position deltas)
      - orient_indices: eef rotation (for HDBSCAN on rotation deltas)
      - gripper_indices: gripper joint positions (for state-change detection)
    """
    state_info = modality_dict["state"]

    def _collect(keys):
        indices = []
        for key in keys:
            if key in state_info:
                s, e = state_info[key]["start"], state_info[key]["end"]
                indices.extend(range(s, e))
        return indices

    spatial_indices = _collect(["end_effector_position_relative"])
    orient_indices = _collect(["end_effector_rotation_relative"])
    gripper_indices = _collect(["gripper_qpos"])

    return spatial_indices, orient_indices, gripper_indices


def load_episode_data(
    ds, ep_idx: int, cameras: list[str], num_workers: int = 8
) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    """
    Load states and multi-camera frames for an episode using a thread pool.

    A single ds[idx] call decodes all requested cameras, so there is no
    extra per-camera overhead.

    Args:
        cameras: list of camera names, e.g.
            ["robot0_agentview_left", "robot0_eye_in_hand"]

    Returns:
        states: (N, D) numpy array
        frames_dict: {camera_name: list of (H, W, 3) uint8 arrays}
    """
    from_idx = int(ds.episode_data_index["from"][ep_idx])
    to_idx = int(ds.episode_data_index["to"][ep_idx])
    n = to_idx - from_idx

    img_keys = [f"observation.images.{c}" for c in cameras]

    def _load_one(local_idx: int):
        sample = ds[from_idx + local_idx]
        state = sample["observation.state"].numpy()
        imgs = {}
        for cam, key in zip(cameras, img_keys):
            t = sample[key]  # (3, H, W) float32 [0, 1]
            imgs[cam] = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return local_idx, state, imgs

    states: list = [None] * n
    frames_dict: dict[str, list] = {c: [None] * n for c in cameras}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_one, i): i for i in range(n)}
        for future in as_completed(futures):
            local_idx, state, imgs = future.result()
            states[local_idx] = state
            for cam in cameras:
                frames_dict[cam][local_idx] = imgs[cam]

    return np.stack(states), frames_dict


# ─── VLM annotation ──────────────────────────────────────────────────────────

def check_valid(response_text: str, segment_count: int,
                subtask_list: list[str] | None = None):
    """Validate VLM response format. Returns True or error string."""
    search = re.search(r"\{[\s\S]*\}", response_text)
    if search is None:
        return "no dict"
    try:
        match = ast.literal_eval(search.group(0))
    except Exception:
        return "no valid dict"
    for k, v in match.items():
        if not isinstance(v, list) or len(v) != 2:
            return "wrong format"
        if subtask_list and v[0] not in subtask_list:
            return "subtask not in list"
    if len(match) != segment_count:
        return "wrong segment number"
    return True


def segment_to_subtask(instruction, subtask_list, key_frames, segment_count,
                       model: QwenVLLM, max_retries=5):
    """
    Ask VLM to label each segment by selecting from the predefined
    subtask list.  Returns the raw text output or an error string.
    """
    subtask_list_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtask_list))
    prompt = (
        f"The robot successfully completed a task specified by the instruction: '{instruction}'. "
        f"This task can be decomposed into the following subtasks:\n{subtask_list_str}\n\n"
        "Here are segments of images showing the robot performing the task in sequential order. "
        "Each image is a third-person camera view of the robot and workspace. "
        "For each segment, select which subtask from the list above best describes "
        "the robot's behavior and explain your reasoning. "
        "You MUST choose exactly from the subtask strings listed above. "
        "You can assign the same subtask to multiple segments. "
        "Output in dictionary format: "
        "{segment_number: [subtask, reason], ...}, "
        f"segment_number starts from 1 (integer), total segments = {segment_count}. "
        "subtask must be exactly one of the subtask strings from the list."
    )

    for attempt in range(3):
        retry = 0
        while retry < max_retries:
            try:
                response = model.generate_content([prompt, *key_frames])
                break
            except Exception:
                retry += 1
                if retry < max_retries:
                    time.sleep(min(10 * retry, 60))
                else:
                    return "no response"

        valid = check_valid(response.text, segment_count, subtask_list)
        if valid is True:
            return response.text

    return valid  # error string


def segment_to_subtask_single(
    instruction: str,
    subtask_list: list[str],
    segment_key_frames: list,
    segment_num: int,
    segment_count: int,
    frame_range: tuple[int, int],
    total_frames: int,
    model: QwenVLLM,
    prev_subtask: str | None = None,
    use_dual_view: bool = False,
    max_retries: int = 5,
) -> list | str:
    """
    Ask VLM to label ONE segment. Input is only this segment's key frames.
    Returns [subtask, reason] or an error string.
    """
    subtask_list_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtask_list))

    prev_subtask_desc = prev_subtask if prev_subtask is not None else "None (first segment)"

    view_desc = (
        "Each image contains two views side-by-side: LEFT=third-person, RIGHT=wrist (eye-in-hand). "
        if use_dual_view
        else "Each image is a third-person camera view of the robot and workspace. "
    )
    start_idx, end_idx = frame_range
    prompt = (
        "You are an expert robotics video annotator. "
        "Your job is to label segmented video clips from a robot manipulation episode. "
        "For each segment, output the best-matching subtask and a concise visual reason.\n\n"
        f"Task instruction: '{instruction}'\n"
        f"Candidate subtasks (in required execution order):\n{subtask_list_str}\n\n"
        "[Context for inference only; do not mention this context in reason]\n"
        f"Full episode: {total_frames} frames, split into {segment_count} segments.\n"
        f"Current segment: {segment_num}/{segment_count}, frames {start_idx} to {end_idx}.\n"
        f"Previous segment subtask: {prev_subtask_desc}\n"
        f"Visual input: key frames from this segment. {view_desc}\n"
        "Decision rules:\n"
        "1. Select exactly one subtask string from the candidate list.\n"
        "2. Prefer direct visual evidence: end-effector motion, contact state, manipulated object, object pose change, and scene state.\n"
        "3. If evidence is ambiguous across candidates, choose the one most consistent with visible interaction progress.\n"
        "4. Reason must describe only visible content, not segment IDs, frame numbers, or temporal metadata.\n\n"
        "Output JSON only with this schema:\n"
        "{\"subtask\": \"<exact subtask string>\", \"reason\": \"<brief image-grounded explanation>\"}"
    )

    for attempt in range(3):
        retry = 0
        while retry < max_retries:
            try:
                response = model.generate_content([prompt, *segment_key_frames])
                break
            except Exception:
                retry += 1
                if retry < max_retries:
                    time.sleep(min(10 * retry, 60))
                else:
                    return "no response"

        text = response.text
        # Parse {"subtask": "...", "reason": "..."}
        search = re.search(r"\{[^{}]*\"subtask\"[^{}]*\"reason\"[^{}]*\}", text)
        if search:
            try:
                parsed = ast.literal_eval(search.group(0))
                st = str(parsed.get("subtask", "")).strip()
                reason = str(parsed.get("reason", "")).strip()
                if st in subtask_list:
                    return [st, reason]
                return "subtask not in list"
            except Exception:
                pass
        # Fallback: try generic dict
        search = re.search(r"\{[\s\S]*?\}", text)
        if search:
            try:
                parsed = ast.literal_eval(search.group(0))
                st = str(parsed.get("subtask", parsed.get("Subtask", ""))).strip()
                reason = str(parsed.get("reason", parsed.get("Reason", ""))).strip()
                if st in subtask_list:
                    return [st, reason]
                return "subtask not in list"
            except Exception:
                pass
    return "no valid dict"


def parse_segment_labels(response_text: str) -> dict[int, list]:
    """Extract {segment_number: [subtask, reason]} from VLM response."""
    search = re.search(r"\{[\s\S]*\}", response_text)
    if search is None:
        return {}
    try:
        raw = ast.literal_eval(search.group(0))
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def build_frame_annotations(processed_segs: list, segment_labels: dict[int, list]):
    """
    Map segment-level labels to per-frame [subtask, reason] pairs.

    processed_segs: length-N list of cluster IDs from HDBSCAN.
    segment_labels: {1: [subtask, reason], 2: [...], ...} from VLM.

    Returns a length-N list of [subtask, reason].
    """
    seg_frames: dict[int, list[int]] = {}
    for i, seg in enumerate(processed_segs):
        seg_frames.setdefault(seg, []).append(i)
    sorted_segs = sorted(seg_frames.items(), key=lambda kv: kv[1][0])

    cluster_to_seg_num = {
        cluster_id: idx + 1
        for idx, (cluster_id, _) in enumerate(sorted_segs)
    }

    n_frames = len(processed_segs)
    frame_annotations: list[list[str]] = [["unknown", "unknown"]] * n_frames
    for i, seg in enumerate(processed_segs):
        seg_num = cluster_to_seg_num[seg]
        label = segment_labels.get(seg_num, ["unknown", "unknown"])
        frame_annotations[i] = label

    return frame_annotations


# ─── visualisation ────────────────────────────────────────────────────────────

_SEGMENT_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def _subtask_color_rgb(name: str) -> tuple:
    return _SEGMENT_PALETTE[hash(name) % len(_SEGMENT_PALETTE)]


def _rgb2bgr(c):
    return (c[2], c[1], c[0])


def _load_font(size=14):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_segment_bar(frame, subtask, seg_idx, total_segs, t, n_frames):
    h, w = frame.shape[:2]
    bar_h = 20
    bgr = _rgb2bgr(_subtask_color_rgb(subtask))
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)
    prog = int(w * t / max(n_frames - 1, 1))
    cv2.rectangle(frame, (0, 0), (prog, bar_h), bgr, -1)
    cv2.putText(frame, f"Seg {seg_idx}/{total_segs}  Frame {t}/{n_frames}",
                (6, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)


def _draw_subtask_label(frame, subtask):
    h, w = frame.shape[:2]
    banner_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    bgr = _rgb2bgr(_subtask_color_rgb(subtask))
    cv2.putText(frame, subtask, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, bgr, 1, cv2.LINE_AA)


def _make_text_panel(instruction, subtask, reason, t, width):
    panel_h = 160
    panel = np.ones((panel_h, width, 3), dtype=np.uint8) * 245
    pil_img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(13)
    font_s = _load_font(11)
    y, lh = 6, 16

    draw.text((8, y), f"Frame {t}", fill=(100, 100, 100), font=font_s); y += lh
    draw.text((8, y), "INSTRUCTION:", fill=(0, 0, 180), font=font); y += lh
    for ln in textwrap.wrap(instruction, width=80):
        draw.text((12, y), ln, fill=(30, 30, 30), font=font_s); y += lh - 2
    y += 4
    draw.text((8, y), "SUBTASK:", fill=_subtask_color_rgb(subtask), font=font); y += lh
    for ln in textwrap.wrap(subtask, width=80):
        draw.text((12, y), ln, fill=(30, 30, 30), font=font_s); y += lh - 2
    y += 4
    draw.text((8, y), "REASON:", fill=(128, 0, 128), font=font); y += lh
    for ln in textwrap.wrap(reason, width=80):
        draw.text((12, y), ln, fill=(60, 60, 60), font=font_s); y += lh - 2

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_episode_video(
    all_frames: list[np.ndarray],
    overall_segment: list,
    segment_count: int,
    frame_annotations: list[list[str]],
    instruction: str,
    out_path: Path,
    fps: int = 20,
):
    """Render an annotated debug video from in-memory RGB frames.

    Uses ffmpeg (libx264 / H.264) so that the output plays directly in
    VSCode's built-in video player.
    """
    n = len(all_frames)
    if n == 0:
        return

    h, w = all_frames[0].shape[:2]
    text_h = 160
    out_h, out_w = h + text_h, w

    # segment value → sequential 1-based index
    seen: dict[float, int] = {}
    order = 0
    for s in overall_segment:
        if s not in seen:
            order += 1
            seen[s] = order

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for t in range(n):
        frame = cv2.cvtColor(all_frames[t], cv2.COLOR_RGB2BGR)

        subtask, reason = frame_annotations[t] if t < len(frame_annotations) else ("unknown", "unknown")
        seg_idx = seen.get(overall_segment[t], 0) if t < len(overall_segment) else 0

        _draw_segment_bar(frame, subtask, seg_idx, segment_count, t, n)
        _draw_subtask_label(frame, subtask)

        panel = _make_text_panel(instruction, subtask, reason, t, w)
        combined = np.vstack([frame, panel])
        proc.stdin.write(combined.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate per-frame subtask annotations for RoboCasa")
    parser.add_argument("--dataset", type=str, default="",
                        help="Optional: single lerobot dataset dir. If set, overrides --datasets_root/--subsets.")
    parser.add_argument("--datasets_root", type=str, default="/tmp/sx0401/workspace/datasets/v1.0/",
                        help="Root dir containing pretrain/ and target/")
    parser.add_argument("--output", type=str, default="tmp/robocasa_subtasks",
                        help="Output directory. In multi-dataset mode, keeps dataset-relative folder structure.")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--camera", type=str, default="robot0_agentview_left",
                        help="Main camera view for VLM")
    parser.add_argument("--wrist_camera", type=str, default="robot0_eye_in_hand",
                        help="Optional wrist camera (e.g. robot0_eye_in_hand). "
                             "If set, stitches main+wrist side-by-side for VLM. Default: single view only")
    parser.add_argument("--instruction_mapping", type=str,
                        default="scripts/robocasa_labeling/instruction_subtask_mapping.json",
                        help="Path to instruction→subtask mapping JSON")
    # VLM options
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8002/v1/chat/completions")
    parser.add_argument("--model_name", type=str,
                        default="/jobfs/163093094.gadi-pbs/models/models--Qwen--Qwen3-VL-8B-Instruct"
                                "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip VLM calls (for testing)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of threads for parallel frame loading")
    # Visualisation
    parser.add_argument("--visualize", action="store_true",
                        help="Render debug videos for each episode")
    parser.add_argument("--viz_output", type=str, default="tmp/subtask_viz",
                        help="Output directory for debug videos")
    parser.add_argument(
        "--subsets", nargs="+",
        default=["target_no_nav", "pretrain_human300_no_navigation"],
        help="DATASET_SOUP_REGISTRY subset names to label. "
             "Pass an empty string '' to scan all datasets without filtering.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards for parallel runs (default: 1, disabled)",
    )
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Current shard index in [0, num_shards-1]",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard_id must satisfy 0 <= shard_id < num_shards")

    datasets_root = Path(args.datasets_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir: Path | None = None
    if args.visualize:
        viz_dir = Path(args.viz_output)
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Visualisation enabled → {viz_dir}")

    # Resolve datasets to process
    if args.dataset:
        dataset_paths = [Path(args.dataset)]
        print(f"[info] Single-dataset mode: {dataset_paths[0]}")
    else:
        allowed_paths: set[str] | None = None
        if args.subsets and args.subsets != [""]:
            allowed_paths = resolve_subset_paths(args.subsets)
            print(f"[info] Filtering to subsets: {args.subsets} ({len(allowed_paths)} dataset paths)")
        dataset_paths = discover_lerobot_datasets(datasets_root, allowed_paths=allowed_paths)
        print(f"[info] Found {len(dataset_paths)} datasets to annotate")

    if args.num_shards > 1:
        if args.dataset:
            print(
                f"[info] Shard mode enabled: {args.shard_id}/{args.num_shards} "
                "(single-dataset mode: dataset-level sharding is ignored)"
            )
        else:
            dataset_paths = [
                p for i, p in enumerate(dataset_paths)
                if i % args.num_shards == args.shard_id
            ]
            print(
                f"[info] Shard mode enabled: {args.shard_id}/{args.num_shards}, "
                f"assigned {len(dataset_paths)} datasets"
            )

    if not dataset_paths:
        print("[warn] No datasets found to process.")
        return

    # Load instruction→subtask mapping
    instr_mapping_path = Path(args.instruction_mapping)
    instr_mapping: dict = {}
    if instr_mapping_path.exists():
        with open(instr_mapping_path) as f:
            instr_mapping = json.load(f)
        print(f"[info] Loaded instruction mapping with {len(instr_mapping)} entries")
    else:
        print(f"[error] Instruction mapping not found at {instr_mapping_path}")
        print("       Run generate_instruction_subtasks.py first.")
        sys.exit(1)

    # Init VLM
    if not args.dry_run:
        model = QwenVLLM(api_url=args.api_url, model_name=args.model_name)
    else:
        model = None
        print("[info] Dry-run mode — VLM calls will be skipped")

    error_counts: dict[str, int] = defaultdict(int)
    total_saved = 0
    for dataset_i, dataset_path in enumerate(dataset_paths, 1):
        dataset_path = Path(dataset_path)
        task_name = dataset_path.parent.parent.name

        if args.dataset:
            try:
                rel = dataset_path.relative_to(datasets_root)
                result_dir = output_dir / rel
                result_dir.mkdir(parents=True, exist_ok=True)
                result_path = result_dir / "subtasks.json"
            except ValueError:
                # Fallback when dataset_path is not under datasets_root.
                result_path = output_dir / f"subtasks_{task_name}.json"
        else:
            rel = dataset_path.relative_to(datasets_root)
            result_dir = output_dir / rel
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / "subtasks.json"

        print(f"\n[dataset {dataset_i}/{len(dataset_paths)}] {dataset_path}")

        # Load dataset metadata (modality layout + fps)
        try:
            modality_dict, dataset_fps = load_dataset_meta(dataset_path)
        except Exception as e:
            print(f"[warn] Failed to load metadata for {dataset_path}: {e}")
            error_counts["load meta failed"] += 1
            continue
        spatial_indices, orient_indices, gripper_indices = get_segmentation_indices(modality_dict)
        print(f"[info] Triple segmentation (Emma-X): "
              f"spatial={len(spatial_indices)}D, orient={len(orient_indices)}D, "
              f"gripper={len(gripper_indices)}D, fps={dataset_fps}")

        # Load episodes metadata
        episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
        ep_metas = []
        try:
            with open(episodes_jsonl) as f:
                for line in f:
                    ep_metas.append(json.loads(line))
        except Exception as e:
            print(f"[warn] Failed to read {episodes_jsonl}: {e}")
            error_counts["load episodes failed"] += 1
            continue
        total = len(ep_metas)
        print(f"[info] Found {total} episodes for {task_name}")

        end_ep = total if args.max_episodes <= 0 else min(args.start_ep + args.max_episodes, total)
        ep_indices = list(range(args.start_ep, end_ep))

        # Checkpoint resume
        results: dict = {}
        if result_path.exists():
            with open(result_path) as f:
                results = json.load(f)
            print(f"[info] Loaded {len(results)} existing subtask results")

        # Load LeRobot dataset
        print("[info] Loading LeRobot dataset ...")
        try:
            ds = load_lerobot_dataset(dataset_path)
        except Exception as e:
            print(f"[warn] Failed to load LeRobot dataset {dataset_path}: {e}")
            error_counts["load dataset failed"] += 1
            continue
        print(f"[info] Dataset loaded: {ds.num_episodes} episodes, {ds.num_frames} frames")
        viz_done_for_task = False

        for ep_idx in tqdm(ep_indices, desc=f"Subtask annotation [{task_name}]"):
            key = f"{task_name}|{ep_idx}"
            if key in results:
                continue

            # Get instruction (needed for mapping lookup)
            ep_meta_json = dataset_path / "extras" / f"episode_{ep_idx:06d}" / "ep_meta.json"
            if ep_meta_json.exists():
                with open(ep_meta_json) as f:
                    ep_meta = json.load(f)
                instruction = ep_meta.get("lang", "")
            else:
                instruction = ""
            if not instruction:
                task_desc = ep_metas[ep_idx].get("tasks", [""])[0] if ep_idx < len(ep_metas) else ""
                instruction = task_desc

            # Look up subtask list and count from mapping
            if not instruction or instruction not in instr_mapping:
                tqdm.write(f"[warn] No mapping for ep {ep_idx}: '{instruction[:60]}', skipping")
                error_counts["no mapping"] += 1
                continue

            entry = instr_mapping[instruction]
            subtask_list = entry.get("subtasks", [])
            subtask_count = entry.get("subtask_count", 0)

            if not subtask_list or subtask_count == 0:
                tqdm.write(f"[warn] Empty subtask list for ep {ep_idx}, skipping")
                error_counts["empty subtasks"] += 1
                continue

            # Load states + frames (main + optional wrist camera)
            cameras = [args.camera]
            if args.wrist_camera:
                cameras.append(args.wrist_camera)
            try:
                obs_state, frames_dict = load_episode_data(
                    ds, ep_idx, cameras, num_workers=args.num_workers
                )
            except Exception as e:
                tqdm.write(f"[warn] Failed to load data for ep {ep_idx}: {e}")
                error_counts["load episode failed"] += 1
                continue

            main_frames = frames_dict[args.camera]
            if args.wrist_camera:
                wrist_frames = frames_dict[args.wrist_camera]
                main_h = main_frames[0].shape[0]
                all_frames = []
                for main_f, wrist_f in zip(main_frames, wrist_frames):
                    wh, ww = wrist_f.shape[:2]
                    if wh != main_h:
                        scale = main_h / wh
                        wrist_resized = cv2.resize(
                            wrist_f, (int(ww * scale), main_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        wrist_resized = wrist_f
                    all_frames.append(np.hstack([main_f, wrist_resized]))
            else:
                all_frames = main_frames

            # Triple segmentation (Emma-X): spatial + orient + gripper
            spatial_state = obs_state[:, spatial_indices]
            orient_state = obs_state[:, orient_indices]
            gripper_qpos = obs_state[:, gripper_indices]
            overall_segment = triple_segment(
                spatial_state, orient_state, gripper_qpos, fps=dataset_fps,
            )

            # Get key frames per segment (one list of images per segment)
            key_frames_per_segment, frame_ranges = get_key_frames_per_segment(
                all_frames, overall_segment
            )
            segment_count = len(key_frames_per_segment)
            use_dual_view = bool(args.wrist_camera)

            # VLM annotation: one call per segment
            if args.dry_run:
                segment_labels = {i + 1: ["dry_run", "dry_run"]
                                 for i in range(segment_count)}
                frame_annotations = [["dry_run", "dry_run"]] * len(all_frames)
                model_output = "dry_run"
            else:
                segment_labels = {}
                prev_subtask = None
                for seg_num, seg_key_frames in enumerate(key_frames_per_segment, 1):
                    frame_range = frame_ranges[seg_num - 1]
                    label = segment_to_subtask_single(
                        instruction, subtask_list, seg_key_frames,
                        seg_num, segment_count, frame_range, len(all_frames), model,
                        prev_subtask=prev_subtask,
                        use_dual_view=use_dual_view,
                    )
                    if isinstance(label, list):
                        segment_labels[seg_num] = label
                        if label[0] in subtask_list:
                            prev_subtask = label[0]
                    else:
                        segment_labels[seg_num] = ["error", str(label)]
                        error_counts[label] = error_counts.get(label, 0) + 1
                frame_annotations = build_frame_annotations(
                    overall_segment, segment_labels
                )
                model_output = "per_segment"

            results[key] = {
                "instruction": instruction,
                "subtask_list": subtask_list,
                "subtask_count": subtask_count,
                "segments": overall_segment,
                "segment_count": segment_count,
                "segment_labels": {str(k): v for k, v in segment_labels.items()},
                "model_output": model_output,
                "frame_annotations": frame_annotations,
            }

            # Render debug video (frames are still in memory)
            if viz_dir is not None and not viz_done_for_task:
                dataset_tag = "__".join(dataset_path.parts[-5:-1])  # split/category/task/date
                vid_path = viz_dir / f"{dataset_tag}_ep{ep_idx:04d}_{args.camera}.mp4"
                try:
                    render_episode_video(
                        all_frames, overall_segment, segment_count,
                        frame_annotations, instruction, vid_path, fps=dataset_fps,
                    )
                    tqdm.write(f"  [viz] {vid_path}")
                except Exception as e:
                    tqdm.write(f"  [viz-error] ep {ep_idx}: {e}")
                viz_done_for_task = True

            # Periodic save
            if (ep_idx + 1) % 20 == 0 or ep_idx == ep_indices[-1]:
                with open(result_path, "w") as f:
                    json.dump(results, f, indent=2)
                tqdm.write(f"  [checkpoint] {len(results)} episodes saved")

        # Final save for this dataset
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[done] {len(results)} subtask results saved to {result_path}")
        total_saved += len(results)

    print(f"\n[done] All datasets finished. Total saved entries: {total_saved}")
    if error_counts:
        print(f"  Errors: {dict(error_counts)}")


if __name__ == "__main__":
    main()
