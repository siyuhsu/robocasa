"""
LIBERO-specific helpers for CoT labeling pipeline.

Reuses robocasa_labeling/utils.py for QwenVLLM, triple_segment, describe_move,
get_key_frames_per_segment. This file adds LIBERO-specific I/O:
- LeRobot parquet/video reading (no robocasa.utils.lerobot_utils dependency)
- modality.json field-slicing for state/action layout
- Suite/episode enumeration

LIBERO LeRobot layout (verified on libero_goal_no_noops_1.0.0_lerobot, fps=20):
  observation.state  (8): [x, y, z, roll, pitch, yaw, pad, gripper]
  action             (7): [x, y, z, roll, pitch, yaw, gripper]
  observation.images.image       (256x256, primary third-person)
  observation.images.wrist_image (256x256)

Episode count per suite:
  libero_goal: 428, libero_object: 454, libero_spatial: 432, libero_10: 379
  Total: 1693 episodes, 10 unique tasks per suite (40 distinct instructions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

# Reuse robocasa_labeling utilities (sibling directory).
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "robocasa_labeling"))
from utils import (  # noqa: E402,F401
    QwenVLLM,
    describe_move,
    get_key_frames,
    get_key_frames_per_segment,
    triple_segment,
)


# ───── LIBERO suite registry ────────────────────────────────────────────────

LIBERO_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]


def suite_dir(data_root: Path, suite: str) -> Path:
    """Return the LeRobot dataset directory for a LIBERO suite."""
    return Path(data_root) / f"{suite}_no_noops_1.0.0_lerobot"


def discover_libero_suites(data_root: Path, suites: Iterable[str] | None = None) -> list[Path]:
    """Resolve suite names → existing LeRobot directories under data_root."""
    suites = list(suites) if suites else LIBERO_SUITES
    out = []
    for s in suites:
        d = suite_dir(data_root, s)
        if (d / "meta" / "info.json").exists():
            out.append(d)
        else:
            print(f"[warn] suite '{s}' not found at {d}, skipping")
    return out


# ───── modality / state slicing ─────────────────────────────────────────────


def load_modality(dataset_path: Path) -> dict:
    """Read meta/modality.json (LIBERO-style: per-field {start, end} ranges)."""
    with open(dataset_path / "meta" / "modality.json") as f:
        return json.load(f)


def get_segmentation_indices(modality_dict: dict) -> tuple[list[int], list[int], list[int]]:
    """
    Build column indices for triple_segment from LIBERO modality.

    Returns:
        spatial_indices: state[x,y,z]   → 3 dims
        orient_indices:  state[r,p,y]   → 3 dims (axis-angle, NOT quaternion)
        gripper_indices: state[gripper] → 1 dim
    """
    state = modality_dict["state"]

    def _range(key):
        return list(range(state[key]["start"], state[key]["end"]))

    spatial = _range("x") + _range("y") + _range("z")
    orient = _range("roll") + _range("pitch") + _range("yaw")
    gripper = _range("gripper")
    return spatial, orient, gripper


def load_dataset_meta(dataset_path: Path) -> tuple[dict, int]:
    """Return (modality_dict, fps)."""
    modality_dict = load_modality(dataset_path)
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)
    return modality_dict, int(info.get("fps", 20))


# ───── tasks.jsonl ──────────────────────────────────────────────────────────


def load_task_index_to_instruction(dataset_path: Path) -> dict[int, str]:
    """Read meta/tasks.jsonl: {task_index: instruction_string}."""
    out = {}
    with open(dataset_path / "meta" / "tasks.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj["task_index"])] = obj["task"]
    return out


def load_episode_lengths(dataset_path: Path) -> list[tuple[int, int, str]]:
    """Read meta/episodes.jsonl: list of (episode_index, length, first_task)."""
    out = []
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            ep_idx = int(obj["episode_index"])
            length = int(obj["length"])
            tasks = obj.get("tasks", [])
            inst = tasks[0] if tasks else ""
            out.append((ep_idx, length, inst))
    return out


# ───── parquet reader ───────────────────────────────────────────────────────


def parquet_path(dataset_path: Path, ep_idx: int, chunk_size: int = 1000) -> Path:
    """Build path to a single episode's parquet file."""
    chunk_id = ep_idx // chunk_size
    return dataset_path / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"


def read_episode_states_actions(dataset_path: Path, ep_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read one episode's state + action from parquet.

    Returns:
        states:    (N, state_dim)
        actions:   (N, action_dim)  — LIBERO action_dim=7
        task_idx:  task_index (single int per episode)
    """
    p = parquet_path(dataset_path, ep_idx)
    if not p.exists():
        raise FileNotFoundError(f"parquet not found: {p}")
    t = pq.read_table(p)
    states = np.stack([np.asarray(s, dtype=np.float32) for s in t.column("observation.state").to_pylist()])
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in t.column("action").to_pylist()])
    task_idx = int(t.column("task_index")[0].as_py())
    return states, actions, task_idx


# ───── video reader (primary camera only, motion-only labeling) ─────────────


def video_path(dataset_path: Path, ep_idx: int, video_key: str = "observation.images.image",
               chunk_size: int = 1000) -> Path:
    """Build path to a single episode's MP4 for one camera."""
    chunk_id = ep_idx // chunk_size
    return dataset_path / "videos" / f"chunk-{chunk_id:03d}" / video_key / f"episode_{ep_idx:06d}.mp4"


def read_video_frames(video_file: Path) -> list[np.ndarray]:
    """
    Decode all frames from an MP4. Tries imageio (pyav) first, falls back to cv2.
    Returns list of HxWx3 uint8 RGB arrays.
    """
    try:
        import imageio.v3 as iio
        return [np.ascontiguousarray(f) for f in iio.imiter(str(video_file))]
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames


class HFQwenVLClient:
    """
    In-process Qwen3-VL VLM client with the same .generate_content() API as QwenVLLM.

    Use this when no vLLM server is available. Loads the model once with
    transformers 4.57.0 + flash_attention_2 (Pawsey-aligned env) on the configured
    device, then runs Qwen3-VL inference for labeling calls.

    Mirrors QwenVLLM's interface so robocasa_labeling/generate_subtasks.py's
    segment_to_subtask_single can use either client interchangeably.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda:0",
                 dtype: str = "bfloat16",
                 attn_implementation: str = "sdpa",  # safer default on 4090 vs flash_attention_2
                 max_new_tokens: int = 512,
                 ):
        import torch
        from transformers import AutoProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens
        self._dtype = getattr(torch, dtype)

        print(f"[HFQwenVLClient] Loading Qwen3-VL from {model_path} on {device} ({dtype}, {attn_implementation}) ...")

        # Try Qwen3VLForConditionalGeneration first; fall back to AutoModelForCausalLM
        try:
            from transformers import Qwen3VLForConditionalGeneration as _Cls
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForCausalLM as _Cls

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = _Cls.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        print(f"[HFQwenVLClient] Ready. param mem ≈ {sum(p.numel() for p in self.model.parameters()) * 2 / 1e9:.2f} GB")

    def generate_content(self, content_list, max_tokens: int = None, timeout=None):
        """
        Args:
            content_list: alternating text str and PIL.Image instances
            max_tokens:   max new tokens to generate (overrides ctor default)
        Returns:
            object with `.text` attribute (mirrors requests-based clients)
        """
        import torch

        max_tokens = max_tokens or self.max_new_tokens

        message_content = []
        for item in content_list:
            if isinstance(item, str):
                message_content.append({"type": "text", "text": item})
            else:  # assume PIL.Image.Image
                message_content.append({"type": "image", "image": item})

        messages = [{"role": "user", "content": message_content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Collect images in order they appear
        images = [item for item in content_list if not isinstance(item, str)]

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Cast image tensors to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        # Strip prompt prefix
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = generated[:, prompt_len:]
        text_out = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0]

        class _Resp:
            def __init__(self, t):
                self.text = t

        return _Resp(text_out)


def sample_video_frames_by_indices(video_file: Path, indices: list[int]) -> list[np.ndarray]:
    """
    Decode only the specified frame indices (faster than loading whole video).
    Returns list of HxWx3 uint8 RGB arrays in input order.
    """
    if not indices:
        return []
    sorted_indices = sorted(set(indices))
    try:
        import imageio.v3 as iio
        # imageio supports random access via index= for reasonable performance
        out_by_idx: dict[int, np.ndarray] = {}
        with iio.imopen(str(video_file), "r", plugin="pyav") as src:
            for i in sorted_indices:
                try:
                    f = src.read(index=i)
                    out_by_idx[i] = np.ascontiguousarray(f)
                except Exception:
                    out_by_idx[i] = None  # type: ignore
        return [out_by_idx.get(i) for i in indices]
    except Exception:
        # Fallback: decode all and pick (slow but safe)
        all_frames = read_video_frames(video_file)
        return [all_frames[i] if i < len(all_frames) else None for i in indices]
