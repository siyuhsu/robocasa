"""
Utility functions for RoboCasa labeling pipeline.

Includes:
- QwenVLLM: vLLM API client for Qwen3-VL
- segment_traj: HDBSCAN trajectory segmentation
- segment_gripper: Gripper state-change segmentation
- triple_segment: Spatial + orient + gripper segmentation (Emma-X)
- describe_move: Convert 7D movement vector to natural language
- get_key_frames: Sample key frames from segments for VLM
"""

import base64
import math
import re
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.cluster import HDBSCAN


# ─── VLM client ──────────────────────────────────────────────────────────────

class QwenVLLM:
    """Qwen3-VL model via vLLM OpenAI-compatible API."""

    def __init__(self,
                 api_url="http://localhost:8100/v1/chat/completions",
                 model_name="/data/app/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct"
                            "/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
                 chat_template_kwargs: dict | None = None):
        """
        chat_template_kwargs: extra arguments passed to the chat template at
            request time. For Qwen3.5 (which has built-in 'thinking' that consumes
            tokens before the actual answer) pass {"enable_thinking": False}.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        print(f"VLM API: {api_url}")
        print(f"Model:   {model_name}")

        # Quick health check
        try:
            health_url = api_url.replace("/v1/chat/completions", "/health")
            resp = self.session.get(health_url, timeout=5)
            if resp.status_code == 200:
                print("✓ vLLM service connected")
            else:
                print(f"⚠ vLLM health check returned {resp.status_code}")
        except Exception as e:
            print(f"⚠ Cannot reach vLLM service: {e}")

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str | None:
        if not isinstance(image, Image.Image):
            return None
        buf = BytesIO()
        image.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def generate_content(self, content_list, max_tokens=4096, timeout=300):
        """Send multi-modal prompt (text + images) and return response text."""
        message_content = []
        for item in content_list:
            if isinstance(item, str):
                message_content.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                url = self._image_to_base64(item)
                if url:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                    })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        resp = self.session.post(self.api_url, json=payload, timeout=timeout)
        resp.raise_for_status()

        class _Response:
            def __init__(self, text):
                self.text = text

        return _Response(resp.json()["choices"][0]["message"]["content"])


# ─── trajectory segmentation ─────────────────────────────────────────────────

def segment_traj(full_state, time_weight=1, expected_segments=None,
                 min_cluster_size=None, fps=20):
    """
    HDBSCAN clustering on spatio-temporal features.

    Args:
        full_state: (N, D) array of robot state per timestep.
        time_weight: scaling factor for temporal dimension.
        expected_segments: if provided, adapt min_cluster_size so that
            the number of clusters is close to this target.
        min_cluster_size: if provided, use this value directly as
            HDBSCAN min_cluster_size (overrides expected_segments).
        fps: frame rate of the data (default 20 for RoboCasa).

    Returns:
        (processed_segments, raw_segments) — both length-N lists.
    """

    def _fill_noise(segments):
        """Replace noise labels (-1) with the nearest valid cluster label."""
        raw_list = segments.tolist() if hasattr(segments, 'tolist') else list(segments)
        out = list(raw_list)

        # Forward pass: propagate last valid label forward
        prev = None
        for i, s in enumerate(raw_list):
            if s != -1:
                prev = s
            elif prev is not None:
                out[i] = prev

        # Backward pass: fill any leading noise that forward pass missed
        nxt = None
        for i in range(len(raw_list) - 1, -1, -1):
            if raw_list[i] != -1:
                nxt = raw_list[i]
            elif out[i] == -1 and nxt is not None:
                out[i] = nxt

        return out

    def _distance(p1, p2):
        spatial = euclidean(p1[:-1], p2[:-1])
        temporal = time_weight * abs(p1[-1] - p2[-1])
        return spatial + temporal

    N = len(full_state)
    X = [np.append(o, i / fps) for i, o in enumerate(full_state)]

    if min_cluster_size is not None and min_cluster_size >= 2:
        clustering = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=_distance,
            copy=False,
        )
        raw = clustering.fit_predict(X)
        return _fill_noise(raw), raw.tolist()
    elif expected_segments is not None and expected_segments > 0:
        best_segs, best_raw, best_diff = None, None, float("inf")
        for mcs in range(max(N // expected_segments, 3), 2, -1):
            clustering = HDBSCAN(
                min_cluster_size=mcs,
                metric=_distance,
                copy=False,
            )
            raw = clustering.fit_predict(X)
            filled = _fill_noise(raw)
            n_clusters = len(set(filled))
            diff = abs(n_clusters - expected_segments)
            if diff < best_diff:
                best_diff = diff
                best_segs = filled
                best_raw = raw.tolist()
            if n_clusters == expected_segments:
                break
            if n_clusters > expected_segments * 2:
                continue
        return best_segs, best_raw
    else:
        clustering = HDBSCAN(
            min_cluster_size=3,
            metric=_distance,
            copy=False,
        )
        raw = clustering.fit_predict(X)
        return _fill_noise(raw), raw.tolist()


# ─── gripper state segmentation ───────────────────────────────────────────────

def segment_gripper(gripper_qpos):
    """
    Segment by gripper state changes (open ↔ close transitions).

    Follows Emma-X: discretise the gripper position to binary open/close,
    then mark a new segment whenever a transition occurs.  Each frame is
    assigned the *frame index* of the most recent transition.

    Args:
        gripper_qpos: (N,) or (N, D) gripper joint positions.
            For RoboCasa D=2 (two finger joints); the mean is used.

    Returns:
        Length-N list of segment IDs.
    """
    g = np.asarray(gripper_qpos, dtype=float)
    if g.ndim == 2:
        g = g.mean(axis=1)

    g_min, g_max = g.min(), g.max()
    if g_max - g_min > 1e-6:
        g_norm = (g - g_min) / (g_max - g_min)
    else:
        g_norm = np.zeros_like(g)

    previous_index = 0
    segments = []
    for i in range(len(g_norm)):
        if round(g_norm[i]) != round(g_norm[previous_index]):
            previous_index = i
        segments.append(previous_index)
    return segments


# ─── triple segmentation (Emma-X) ─────────────────────────────────────────────

def get_delta(states):
    """Compute frame-to-frame deltas (velocity).

    Prepends the first delta so the output has the same length as input.
    """
    states = np.asarray(states)
    deltas = np.diff(states, axis=0)
    return np.concatenate([deltas[:1], deltas], axis=0)


def triple_segment(spatial_state, orient_state, gripper_qpos,
                   fps=20, time_weight=1):
    """
    Triple segmentation: spatial + orientation + gripper.

    Reference: Emma-X ``get_soft_plus_gripper_segment`` —
    independently segment three aspects of the trajectory, then combine::

        overall = spatial_segment * 1e4 + orient_segment * 1e2
                  + gripper_segment

    A new segment boundary appears whenever *any* of the three channels
    detects a change: end-effector position delta cluster, rotation delta
    cluster, or gripper open/close transition.

    Args:
        spatial_state: (N, 3) eef position — raw values; deltas are
            computed internally.
        orient_state: (N, D) eef rotation — raw values; deltas are
            computed internally.
        gripper_qpos: (N, D) or (N,) gripper joint positions.
        fps: dataset frame rate for temporal scaling.
        time_weight: weight for temporal dimension in HDBSCAN.

    Returns:
        overall_segment: length-N list of combined segment IDs.
    """
    # spatial_delta = get_delta(spatial_state)
    # orient_delta = get_delta(orient_state)

    spatial_segs, _ = segment_traj(spatial_state, time_weight=time_weight, fps=fps)
    orient_segs, _ = segment_traj(orient_state, time_weight=time_weight, fps=fps)

    spatial_segment = np.array(spatial_segs)
    orient_segment = np.array(orient_segs)
    gripper_seg = np.array(segment_gripper(gripper_qpos))

    overall = (spatial_segment * 1e4 + orient_segment * 1e2 + gripper_seg).tolist()
    return overall


# ─── key-frame sampling ──────────────────────────────────────────────────────

def get_key_frames(images, overall_segment, max_frames_per_segment=3):
    """
    Sample up to *max_frames_per_segment* key frames (first / mid / last)
    from each segment.  Returns ``(_images, count)`` where ``_images`` is
    a flat list suitable for VLM input (``["Segment 1:", img, img, ...]``).

    *images* can be a list of PIL Images **or** numpy arrays.
    """
    # group frame indices by segment label
    seg_frames: dict[int, list[int]] = {}
    for i, seg in enumerate(overall_segment):
        seg_frames.setdefault(seg, []).append(i)

    sorted_segs = sorted(seg_frames.items(), key=lambda kv: kv[1][0])

    out: list = []
    count = 0
    for _seg_id, indices in sorted_segs:
        count += 1
        out.append(f"Segment {count}:")

        n = len(indices)
        if n <= max_frames_per_segment:
            sampled = indices
        else:
            sampled = [indices[0], indices[n // 2], indices[-1]]

        for idx in sampled:
            img = images[idx]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, "RGB")
            out.append(img)

    return out, count


def get_key_frames_per_segment(images, overall_segment, max_frames_per_segment=5):
    """
    Sample key frames from each segment, returning a list of lists:
    one list of images per segment. Suitable for per-segment VLM calls.

    Returns:
        (out_per_seg, frame_ranges): out_per_seg is list of image lists per segment;
        frame_ranges is list of (start_idx, end_idx) for each segment (0-based frame indices).

    *images* can be a list of PIL Images **or** numpy arrays.
    """
    seg_frames: dict[int, list[int]] = {}
    for i, seg in enumerate(overall_segment):
        seg_frames.setdefault(seg, []).append(i)

    sorted_segs = sorted(seg_frames.items(), key=lambda kv: kv[1][0])
    out_per_seg = []
    frame_ranges = []
    for _seg_id, indices in sorted_segs:
        frame_ranges.append((min(indices), max(indices)))
        n = len(indices)
        if n <= max_frames_per_segment:
            sampled = indices
        else:
            pos = np.linspace(0, n - 1, num=max_frames_per_segment, dtype=int)
            sampled = [indices[p] for p in pos]
        imgs = []
        for idx in sampled:
            img = images[idx]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, "RGB")
            imgs.append(img)
        out_per_seg.append(imgs)
    return out_per_seg, frame_ranges


# ─── movement description ────────────────────────────────────────────────────

def describe_move(move_vec):
    """
    Convert a 7-D movement vector to a natural-language string.

    Input format (same as DROID):
        [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
    - xyz: position delta in **metres** → displayed as mm
    - roll/pitch/yaw: rotation delta in **radians** → displayed as degrees
    - gripper: > 0.5 → "open gripper", else → "close gripper"
    """
    assert len(move_vec) == 7

    names = [
        {False: "move backward", True: "move forward"},
        {False: "move right",    True: "move left"},
        {False: "move downward", True: "move upward"},
        {False: "roll downward", True: "roll upward"},
        {False: "pitch downward",True: "pitch upward"},
        {False: "yaw clockwise", True: "yaw counterclockwise"},
        {False: "close gripper", True: "open gripper"},
    ]

    desc = ""
    for i, mv in enumerate(move_vec):
        if i < 3:
            mm = abs(round(float(mv) * 1000))
            desc += f"{names[i][mv > 0]} {mm} mm; "
        elif i < 6:
            deg = abs(round(float(mv) * 180 / math.pi))
            desc += f"{names[i][mv > 0]} {deg} degrees; "
        else:
            desc += f"{names[i][mv > 0.5]};"
    return desc
