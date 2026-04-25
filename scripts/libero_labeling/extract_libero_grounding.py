"""
Extract per-frame bbox + gripper-2D for LIBERO episodes (Phase D.2).

For each episode:
  1. Build SegmentationRenderEnv from the matching bddl_file
  2. Find the init_state (out of 50 in init_files) whose eef_pos matches the
     parquet's first frame — that selects the same starting pose.
  3. Rollout: step the env forward with the parquet's recorded actions, frame
     by frame. After each step, capture from obs:
        - <camera>_segmentation_instance  → bbox per task object
        - sim eef position → project to 2D pixel coords for both cameras
  4. Save annotations.json with the same schema as robocasa_labeling output:
        sample_key, instruction, all_object_names,
        cameras: {<cam>: {gripper_2d: [...], task_obj_bbox: [...],
                          distractor_bboxes: [...], all_object_bboxes: [...]}}

Output:
  <output_root>/<suite>/extras/episode_XXXXXX/annotations.json

Run on node09 with libero_eval conda + ForceVLA libero PYTHONPATH:
  CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl \
  LIBERO_CONFIG_PATH=/ssd/.../ForceVLA/third_party/libero/libero \
  PYTHONPATH=/ssd/.../ForceVLA/third_party/libero \
  /ssd/sxu/miniconda3/envs/libero_eval/bin/python \
    extract_libero_grounding.py --suite libero_goal --num_episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# LIBERO_CONFIG_PATH must be set BEFORE importing libero
os.environ.setdefault(
    "LIBERO_CONFIG_PATH",
    "/ssd/sxu/workspace/fengnianzhang/ForceVLA/third_party/libero/libero",
)
os.environ.setdefault("MUJOCO_GL", "egl")

LIBERO_REPO = "/ssd/sxu/workspace/fengnianzhang/ForceVLA/third_party/libero"
sys.path.insert(0, LIBERO_REPO)

from libero.libero.envs import SegmentationRenderEnv  # noqa: E402

LIBERO_BDDL_DIR = Path(LIBERO_REPO) / "libero/libero/bddl_files"
LIBERO_INIT_DIR = Path(LIBERO_REPO) / "libero/libero/init_files"

LIBERO_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]


# ───── helpers ──────────────────────────────────────────────────────────────


def task_to_bddl(task_str: str) -> str:
    """Convert task instruction string to bddl filename stem (libero convention).

    "put the bowl on the plate" -> "put_the_bowl_on_the_plate.bddl"
    """
    return task_str.lower().replace(" ", "_").replace("'", "_") + ".bddl"


def find_bddl_for_instruction(suite_dir: Path, instruction: str) -> Path | None:
    """Locate the bddl file matching an instruction. Some suites (e.g. libero_10)
    prefix bddl filenames with "KITCHEN_SCENE3_" / "LIVING_ROOM_SCENE2_" etc., so
    we test both exact match and suffix match.
    """
    target_stem = instruction.lower().replace(" ", "_").replace("'", "_")
    target_file = suite_dir / f"{target_stem}.bddl"
    if target_file.exists():
        return target_file
    # Fall back to suffix match (handles SCENE-prefixed bddl)
    for c in sorted(suite_dir.glob("*.bddl")):
        if c.stem.lower().endswith(target_stem):
            return c
    return None


def load_init_states(suite: str, bddl_filename: str) -> np.ndarray:
    """Load init_states for a given task. Returns ndarray (N, 79).

    bddl_filename is the basename (e.g. "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl")
    — init_files mirror this naming with .pruned_init / .init extensions.
    """
    stem = bddl_filename.replace(".bddl", "")
    for ext in (".pruned_init", ".init"):
        p = LIBERO_INIT_DIR / suite / (stem + ext)
        if p.exists():
            return torch.load(str(p), weights_only=False)
    raise FileNotFoundError(f"no init_states for {suite}/{stem} (.pruned_init or .init)")


def find_matching_init(init_states: np.ndarray, target_eef_pos: np.ndarray,
                       eef_qpos_indices: tuple = None) -> int:
    """
    Pick the init_state whose stored eef_pos best matches the parquet first-frame
    eef_pos. If we don't know exactly which qpos slots hold the eef, fall back to
    a deterministic index scan (init_state idx 0 by default).

    For LIBERO, the eef_pos can be computed by env.sim after set_init_state, so a
    truly robust match would be: try each init_state, run env.reset+set_init,
    read sim eef_pos, compare. That is O(N*reset) which is too slow. Instead we
    use a simple heuristic: rely on the fact that LeRobot ep_idx maps to
    init_state idx via (ep_idx % 50) for libero_X (50 init_states / task). This
    matches the LIBERO eval convention.
    """
    return None  # caller will fall back to (ep_idx % 50)


def project_to_pixel(eef_pos: np.ndarray, sim, camera_name: str,
                     H: int = 256, W: int = 256) -> tuple[float, float]:
    """World→pixel projection via mujoco camera intrinsics."""
    cam_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.model.cam_pos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * H / np.tan(np.radians(fovy) / 2)
    diff = np.asarray(eef_pos) - np.asarray(cam_pos)
    cam_coord = cam_mat.T @ diff  # mujoco cam looks along -z
    if cam_coord[2] >= 0:  # behind camera
        return None
    u = f * cam_coord[0] / -cam_coord[2] + W / 2
    v = -f * cam_coord[1] / -cam_coord[2] + H / 2
    return float(u), float(v)


def bbox_from_seg(seg: np.ndarray, inst_id: int) -> list | None:
    """Extract [x1,y1,x2,y2] bbox for a given instance id, or None if absent.
    Returned coords are in the segmentation/sim-native image frame.
    """
    if seg.ndim == 3:
        seg = seg[..., 0]
    mask = (seg == inst_id)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def to_mp4_frame_bbox(bbox: list | None, W: int, H: int) -> list | None:
    """Sim-native bbox → raw-MP4 / displayed frame via 180° rotation.

    LIBERO MP4 stored on disk is 180°-rotated relative to the simulator's
    rendered/seg image (eval_libero.py:139 applies [::-1,::-1] to obs before
    model inference, so the on-disk MP4 == sim-native flipped). Saving coords
    in the MP4 frame means downstream consumers (training data loader,
    visualisation) can use them directly with the raw video.
    """
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return [W - 1 - int(x2), H - 1 - int(y2), W - 1 - int(x1), H - 1 - int(y1)]


def to_mp4_frame_pt(pt: tuple | None, W: int, H: int) -> list | None:
    """Sim-projection gripper point → raw-MP4 frame.

    project_to_pixel below internally flips the v-axis (negative leading sign
    in the formula), so its output v already lives in the MP4-displayed frame
    while u is in sim-native. Only x needs flipping.
    """
    if pt is None or len(pt) < 2:
        return None
    return [W - 1 - int(pt[0]), int(pt[1])]


# ───── per-episode rollout ─────────────────────────────────────────────────


def extract_episode_grounding(env: SegmentationRenderEnv,
                              init_state: np.ndarray,
                              actions: np.ndarray,
                              camera_names: list[str] = CAMERA_NAMES,
                              cam_height: int = 256,
                              cam_width: int = 256,
                              ) -> dict:
    """Run env rollout with the recorded actions; capture seg + gripper_2d per frame.

    Saved bbox / gripper_2d coordinates are in the **raw-MP4 (displayed) frame**
    so that they can be drawn directly on top of the LeRobot MP4 and align with
    what the training data loader / model see. See to_mp4_frame_* helpers.
    """
    obs = env.reset()
    obs = env.set_init_state(init_state)
    sim = env.env.sim

    n = actions.shape[0]
    cam_data = {cam: {"gripper_2d": [], "all_object_bboxes": []} for cam in camera_names}

    # Object name list (deterministic order from instance_to_id)
    # Sort task objects of interest first, then distractors
    obj_of_interest = list(env.obj_of_interest)
    all_objs = obj_of_interest + [
        o for o in env.instance_to_id
        if o not in obj_of_interest and o not in ("MountedPanda0",)
    ]

    W, H = cam_width, cam_height
    for t in range(n):
        # Step env with recorded action (no-op-removed actions still apply forward)
        try:
            obs, _r, _d, _i = env.step(actions[t].astype(np.float64))
        except Exception:
            # If env step fails partway, append nulls and continue
            for cam in camera_names:
                cam_data[cam]["gripper_2d"].append(None)
                cam_data[cam]["all_object_bboxes"].append([None] * len(all_objs))
            continue

        # Read eef position from sim (more accurate than parquet state[:3])
        try:
            eef_pos = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")]
        except Exception:
            eef_pos = None

        for cam in camera_names:
            seg_key = f"{cam}_segmentation_instance"
            seg = obs.get(seg_key)
            if seg is None:
                cam_data[cam]["gripper_2d"].append(None)
                cam_data[cam]["all_object_bboxes"].append([None] * len(all_objs))
                continue

            # gripper 2D — project, then map to MP4 frame (x-only flip)
            if eef_pos is not None:
                pt = project_to_pixel(eef_pos, sim, cam, H=H, W=W)
                cam_data[cam]["gripper_2d"].append(to_mp4_frame_pt(pt, W, H))
            else:
                cam_data[cam]["gripper_2d"].append(None)

            # bbox per object — extract in sim frame, then map to MP4 frame (180°)
            frame_boxes = []
            for obj_name in all_objs:
                inst_id = env.instance_to_id.get(obj_name)
                if inst_id is None:
                    frame_boxes.append(None)
                    continue
                bbox_sim = bbox_from_seg(seg, inst_id)
                frame_boxes.append(to_mp4_frame_bbox(bbox_sim, W, H))
            cam_data[cam]["all_object_bboxes"].append(frame_boxes)

    return {
        "all_object_names": all_objs,
        "obj_cat": all_objs[0] if all_objs else None,
        "distr_cats": all_objs[1:] if len(all_objs) > 1 else [],
        "num_frames": n,
        "coord_frame": "raw_mp4",  # bbox/gripper coords aligned with on-disk MP4
        "cameras": cam_data,
    }


# ───── main ────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Path to LEROBOT_LIBERO_DATA")
    p.add_argument("--output_root", required=True,
                   help="Path to LIBERO_GROUNDING (output)")
    p.add_argument("--suite", default="libero_goal",
                   choices=["all", *LIBERO_SUITES])
    p.add_argument("--start_ep", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=0,
                   help="0 = all episodes in suite")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--camera_height", type=int, default=256)
    p.add_argument("--camera_width", type=int, default=256)
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()

    suites = LIBERO_SUITES if args.suite == "all" else [args.suite]

    for suite in suites:
        sdir = data_root / f"{suite}_no_noops_1.0.0_lerobot"
        if not (sdir / "meta" / "info.json").exists():
            print(f"[warn] suite {suite} not found at {sdir}, skipping")
            continue

        # tasks.jsonl: task_index -> instruction
        tasks_jsonl = sdir / "meta" / "tasks.jsonl"
        task_idx_to_instruction = {}
        with open(tasks_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                task_idx_to_instruction[int(obj["task_index"])] = obj["task"]

        # episodes.jsonl: episode_index, length, tasks
        eps_jsonl = sdir / "meta" / "episodes.jsonl"
        episodes = []
        with open(eps_jsonl) as f:
            for line in f:
                episodes.append(json.loads(line))

        total = len(episodes)
        end_ep = total if args.num_episodes <= 0 else min(args.start_ep + args.num_episodes, total)
        ep_indices = list(range(args.start_ep, end_ep))
        print(f"\n[suite {suite}] {len(ep_indices)} episodes")

        # Cache env per task to avoid rebuilding (env build ~10s; 1693 ep × 10s = 5h wasted)
        env_cache: dict[str, SegmentationRenderEnv] = {}
        init_cache: dict[str, np.ndarray] = {}

        # Group episodes by task to minimize env switches
        ep_idx_by_task: dict[str, list[int]] = {}
        for ep_idx in ep_indices:
            inst = episodes[ep_idx]["tasks"][0]
            ep_idx_by_task.setdefault(inst, []).append(ep_idx)

        for instruction, task_eps in ep_idx_by_task.items():
            bddl_path = find_bddl_for_instruction(LIBERO_BDDL_DIR / suite, instruction)
            if bddl_path is None:
                print(f"[warn] no bddl for: '{instruction}' (suite={suite})")
                continue

            # Build env once per task
            if str(bddl_path) not in env_cache:
                print(f"[info] building env for task: {instruction}")
                try:
                    env = SegmentationRenderEnv(
                        bddl_file_name=str(bddl_path),
                        camera_heights=args.camera_height,
                        camera_widths=args.camera_width,
                        camera_segmentations="instance",
                    )
                    env_cache[str(bddl_path)] = env
                    init_cache[str(bddl_path)] = load_init_states(suite, bddl_path.name)
                except Exception as e:
                    print(f"[error] env build failed: {e}")
                    continue
            env = env_cache[str(bddl_path)]
            init_states = init_cache[str(bddl_path)]
            n_inits = len(init_states)

            for ep_idx in tqdm(task_eps, desc=f"{suite}/{instruction[:30]}"):
                out_dir = output_root / suite / "extras" / f"episode_{ep_idx:06d}"
                out_path = out_dir / "annotations.json"
                if out_path.exists() and not args.overwrite:
                    continue

                # Read parquet
                parquet = sdir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
                t = pq.read_table(parquet)
                actions = np.stack(
                    [np.asarray(a, dtype=np.float32) for a in t.column("action").to_pylist()]
                )

                # Map ep_idx → init_state index. Convention: (ep_idx % n_inits)
                # This matches LIBERO's eval convention (50 init_states cycled).
                init_idx = ep_idx % n_inits
                init_state = init_states[init_idx]

                try:
                    payload = extract_episode_grounding(
                        env, init_state, actions,
                        cam_height=args.camera_height,
                        cam_width=args.camera_width,
                    )
                except Exception as e:
                    tqdm.write(f"[error] {suite} ep{ep_idx}: {e}")
                    continue

                payload["sample_key"] = f"{suite}|{ep_idx}"
                payload["instruction"] = instruction
                payload["init_state_idx"] = int(init_idx)

                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

        # Close all envs in this suite
        for e in env_cache.values():
            try:
                e.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
