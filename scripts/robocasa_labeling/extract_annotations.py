"""
Extract per-frame annotations (gripper 2D, object bboxes) from RoboCasa
episodes using the MuJoCo simulator.

Output: one JSON file per episode, saved under an output root while keeping
the dataset-relative folder structure, e.g.
tmp/robocasa_subtasks/<dataset-relative>/extras/episode_000000/annotations.json

Usage:
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=0  # optional, for GPU-accelerated rendering
    conda run -n siyu_robocasa python scripts/robocasa_labeling/extract_annotations.py \
        --datasets_root datasets/v1.0/ \
        --output tmp/robocasa_annotations/ \
        --visualize \
        --num_shards 4 --shard_id 0
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import robocasa          # noqa: F401 — registers assets
import robosuite
import robocasa.utils.lerobot_utils as LU


# ─── constants ────────────────────────────────────────────────────────────────
CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
RENDER_H, RENDER_W = 256, 256


# ─── env helpers (from visualize_episode.py) ──────────────────────────────────

def build_env(dataset_path: Path):
    env_meta = LU.get_env_metadata(dataset_path)
    kw = env_meta["env_kwargs"].copy()
    kw["env_name"] = env_meta["env_name"]
    kw["has_renderer"] = False
    kw["renderer"] = "mjviewer"
    kw["has_offscreen_renderer"] = True
    kw["use_camera_obs"] = False
    kw["camera_names"] = CAMERA_NAMES
    kw["camera_heights"] = RENDER_H
    kw["camera_widths"] = RENDER_W
    return robosuite.make(**kw)


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


def reset_to(env, state: dict):
    if "model" in state:
        ep_meta = json.loads(state.get("ep_meta", "{}")) if state.get("ep_meta") else {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


# ─── geometry / projection helpers ────────────────────────────────────────────

def get_subtree_geom_ids(sim, root_body_name: str) -> list[int]:
    try:
        root_id = sim.model.body_name2id(root_body_name)
    except Exception:
        return []
    body_ids: set[int] = set()
    queue = [root_id]
    while queue:
        bid = queue.pop()
        body_ids.add(bid)
        for child in range(sim.model.nbody):
            if sim.model.body_parentid[child] == bid:
                queue.append(child)
    return [gid for gid in range(sim.model.ngeom) if sim.model.geom_bodyid[gid] in body_ids]


def _find_root_body_name(sim, query_name: str) -> str | None:
    """Find one representative root body by exact/prefix name match."""
    matches = []
    for bid in range(sim.model.nbody):
        bname = sim.model.body_id2name(bid)
        if not bname:
            continue
        if bname == query_name or bname.startswith(query_name + "_"):
            matches.append(bname)
    if not matches:
        return None
    matches.sort(key=len)
    return matches[0]


def _split_camel_case(text: str) -> str:
    """Convert CamelCase text to space-separated lowercase-ish words."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text)


def _normalize_text(text: str) -> str:
    """Normalize text for robust token matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _normalize_token(token: str) -> str:
    """Light token normalization for matching language/object words."""
    t = token.strip().lower()
    # Basic singularization: drawers -> drawer, cabinets -> cabinet, etc.
    if len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


def collect_scene_targets(ep_meta: dict) -> list[dict[str, str]]:
    """
    Build all annotation targets from object_cfgs + fixture_refs.

    Returns a list of dicts with fields:
      query_name: body-name prefix used to find sim bodies
      label: human-readable label used in output / visualization
    """
    targets: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _append_target(query_name: str, label: str):
        key = (str(query_name), str(label))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        targets.append({"query_name": key[0], "label": key[1]})

    # All episode objects defined in object_cfgs.
    for cfg in ep_meta.get("object_cfgs", []):
        cfg_name = cfg.get("name", "")
        if not cfg_name:
            continue
        label = cfg.get("info", {}).get("cat", cfg_name)
        _append_target(cfg_name, label.replace("_", " "))

    # All task-referenced fixtures (e.g. drawer, cabinet, microwave).
    for ref_key, fixture_name in ep_meta.get("fixture_refs", {}).items():
        if not fixture_name:
            continue
        fixture_meta = ep_meta.get("fixtures", {}).get(str(fixture_name), {})
        fixture_cls = str(fixture_meta.get("cls", ref_key)).lower()
        _append_target(str(fixture_name), fixture_cls)

    # Fallback: some episodes have empty fixture_refs; infer fixture targets from lang.
    if not ep_meta.get("fixture_refs"):
        lang = _normalize_text(ep_meta.get("lang", ""))
        lang_words = {_normalize_token(tok) for tok in lang.split() if tok}
        for fixture_name, fixture_meta in ep_meta.get("fixtures", {}).items():
            cls_raw = str(fixture_meta.get("cls", "")).strip()
            if not cls_raw:
                continue
            cls_phrase = _normalize_text(_split_camel_case(cls_raw))
            cls_tokens = [_normalize_token(tok) for tok in cls_phrase.split() if tok]
            if not cls_tokens:
                continue
            if all(tok in lang_words for tok in cls_tokens):
                _append_target(str(fixture_name), cls_raw.lower())

    return targets


def resolve_target_geom_lists(sim, ep_meta: dict) -> tuple[list[str], list[list[int]]]:
    """Resolve each target to one geom-id list for segmentation bboxes."""
    labels: list[str] = []
    geom_lists: list[list[int]] = []
    seen_roots: set[str] = set()

    for target in collect_scene_targets(ep_meta):
        root = _find_root_body_name(sim, target["query_name"])
        if not root or root in seen_roots:
            continue
        geoms = get_subtree_geom_ids(sim, root)
        if not geoms:
            continue
        seen_roots.add(root)
        labels.append(target["label"])
        geom_lists.append(geoms)

    return labels, geom_lists


def project_to_pixel(pos_world, sim, cam_name, img_h, img_w):
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = np.array(sim.data.cam_xpos[cam_id])
    cam_mat = np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    p_cam = cam_mat.T @ (pos_world - cam_pos)
    x_c, y_c, z_c = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z_c >= 0:
        return None
    fovy = float(sim.model.cam_fovy[cam_id])
    f = img_h / (2.0 * np.tan(np.radians(fovy / 2.0)))
    u = f * x_c / (-z_c) + img_w / 2.0
    v = img_h / 2.0 - f * y_c / (-z_c)
    return [int(round(u)), int(round(v))]


def segmentation_bbox(seg_img, geom_ids):
    if not geom_ids:
        return None
    type_ch = seg_img[:, :, 0]
    id_ch = seg_img[:, :, 1]
    geom_mask = (type_ch == 5) | (type_ch == 1)
    mask = np.zeros(type_ch.shape, dtype=bool)
    for gid in geom_ids:
        mask |= geom_mask & (id_ch == gid)
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]


# ─── per-episode extraction ──────────────────────────────────────────────────


def extract_episode(env, dataset_path: Path, ep_idx: int):
    """
    Extract annotations for one episode.

        Returns a dict with:
            instruction, all_object_names, num_frames,
            cameras: {cam_name: {gripper_2d, all_object_bboxes}}
    """
    states = LU.get_episode_states(dataset_path, ep_idx)
    xml_str = LU.get_episode_model_xml(dataset_path, ep_idx)
    ep_meta = LU.get_episode_meta(dataset_path, ep_idx)

    n_frames = states.shape[0]
    instruction = ep_meta.get("lang", "")

    # Reset to initial state (loads model XML)
    initial = {"states": states[0], "model": xml_str, "ep_meta": json.dumps(ep_meta)}
    reset_to(env, initial)

    # Resolve all object/fixture geom IDs in this scene.
    all_object_names, all_geom_lists = resolve_target_geom_lists(env.sim, ep_meta)

    # Get gripper EEF site id
    eef_site_id = None
    try:
        eef_site_id = env.robots[0].eef_site_id
        if isinstance(eef_site_id, dict):
            eef_site_id = list(eef_site_id.values())[0]
    except Exception:
        pass

    # Init camera data containers
    cam_data = {}
    for cam in CAMERA_NAMES:
        cam_data[cam] = {
            "gripper_2d": [],
            "all_object_bboxes": [],
        }
    
    # Iterate through frames
    for t in range(n_frames):
        reset_to(env, {"states": states[t]})

        # EEF world position
        eef_pos = np.array(env.sim.data.site_xpos[eef_site_id]) if eef_site_id is not None else None

        for cam in CAMERA_NAMES:
            # Segmentation render only (no RGB needed)
            seg = env.sim.render(height=RENDER_H, width=RENDER_W,
                                 camera_name=cam, segmentation=True)[::-1]

            # Gripper 2D
            if eef_pos is not None:
                pt = project_to_pixel(eef_pos, env.sim, cam, RENDER_H, RENDER_W)
                cam_data[cam]["gripper_2d"].append(pt)  # may be None
            else:
                cam_data[cam]["gripper_2d"].append(None)

            frame_bboxes = []
            for geom_ids in all_geom_lists:
                frame_bboxes.append(segmentation_bbox(seg, geom_ids))
            cam_data[cam]["all_object_bboxes"].append(frame_bboxes)

    return {
        "instruction": instruction,
        "all_object_names": all_object_names,
        "num_frames": n_frames,
        "cameras": cam_data,
    }


def extract_episode_with_viz_frames(
    env,
    dataset_path: Path,
    ep_idx: int,
):
    """
    Same as extract_episode, but also collects RGB frames from all cameras
    for stitched debug video rendering.
    """
    states = LU.get_episode_states(dataset_path, ep_idx)
    xml_str = LU.get_episode_model_xml(dataset_path, ep_idx)
    ep_meta = LU.get_episode_meta(dataset_path, ep_idx)

    n_frames = states.shape[0]
    instruction = ep_meta.get("lang", "")

    initial = {"states": states[0], "model": xml_str, "ep_meta": json.dumps(ep_meta)}
    reset_to(env, initial)

    all_object_names, all_geom_lists = resolve_target_geom_lists(env.sim, ep_meta)

    eef_site_id = None
    try:
        eef_site_id = env.robots[0].eef_site_id
        if isinstance(eef_site_id, dict):
            eef_site_id = list(eef_site_id.values())[0]
    except Exception:
        pass

    cam_data = {}
    for cam in CAMERA_NAMES:
        cam_data[cam] = {
            "gripper_2d": [],
            "all_object_bboxes": [],
        }

    viz_frames_by_cam: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}
    for t in range(n_frames):
        reset_to(env, {"states": states[t]})

        for cam in CAMERA_NAMES:
            rgb = env.sim.render(
                height=RENDER_H, width=RENDER_W, camera_name=cam
            )[::-1]
            viz_frames_by_cam[cam].append(rgb)

        eef_pos = np.array(env.sim.data.site_xpos[eef_site_id]) if eef_site_id is not None else None

        for cam in CAMERA_NAMES:
            seg = env.sim.render(height=RENDER_H, width=RENDER_W,
                                 camera_name=cam, segmentation=True)[::-1]

            if eef_pos is not None:
                pt = project_to_pixel(eef_pos, env.sim, cam, RENDER_H, RENDER_W)
                cam_data[cam]["gripper_2d"].append(pt)
            else:
                cam_data[cam]["gripper_2d"].append(None)

            frame_bboxes = []
            for geom_ids in all_geom_lists:
                frame_bboxes.append(segmentation_bbox(seg, geom_ids))
            cam_data[cam]["all_object_bboxes"].append(frame_bboxes)

    ep_data = {
        "instruction": instruction,
        "all_object_names": all_object_names,
        "num_frames": n_frames,
        "cameras": cam_data,
    }
    return ep_data, viz_frames_by_cam


def _draw_bbox(frame_bgr: np.ndarray, bbox, color: tuple[int, int, int], label: str):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame_bgr, label, (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def render_episode_video(
    frames_rgb_by_cam: dict[str, list[np.ndarray]],
    ep_data: dict,
    out_path: Path,
    fps: int = 20,
):
    """
    Render one debug video with bboxes + gripper point overlays.
    """
    if not frames_rgb_by_cam:
        return

    active_cameras = [cam for cam in CAMERA_NAMES if cam in frames_rgb_by_cam and frames_rgb_by_cam[cam]]
    if not active_cameras:
        return

    h, w = frames_rgb_by_cam[active_cameras[0]][0].shape[:2]
    out_w = w * len(active_cameras)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{h}",
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

    n = min(len(frames_rgb_by_cam[cam]) for cam in active_cameras)
    all_names = ep_data.get("all_object_names", [])

    for i in range(n):
        stitched_parts: list[np.ndarray] = []
        for cam in active_cameras:
            frame = cv2.cvtColor(frames_rgb_by_cam[cam][i], cv2.COLOR_RGB2BGR)
            cam_ann = ep_data.get("cameras", {}).get(cam, {})

            grippers = cam_ann.get("gripper_2d", [])
            all_bboxes = cam_ann.get("all_object_bboxes", [])

            if i < len(all_bboxes):
                frame_boxes = all_bboxes[i]
                for j, bbox in enumerate(frame_boxes):
                    label = all_names[j] if j < len(all_names) else f"obj_{j}"
                    color = (0, 255, 255) if j == 0 else (0, 180, 0)
                    _draw_bbox(frame, bbox, color, label)

            pt = grippers[i] if i < len(grippers) else None
            if pt is not None:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            cv2.putText(
                frame, cam, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Frame {i + 1}/{n}", (8, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            stitched_parts.append(frame)

        stitched = cv2.hconcat(stitched_parts)
        proc.stdin.write(stitched.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract RoboCasa annotations")
    parser.add_argument("--dataset", type=str, default="",
                        help="Optional: single lerobot dataset dir. If set, overrides --datasets_root/--subsets.")
    parser.add_argument("--datasets_root", type=str, default="datasets/v1.0",
                        help="Root dir containing pretrain/ and target/")
    parser.add_argument("--output", type=str, default="tmp/robocasa_grounding_debug",
                        help="Output directory. Keeps dataset-relative folder structure.")
    parser.add_argument("--visualize", default=True, action="store_true",
                        help="Render one debug video per dataset")
    parser.add_argument("--viz_output", type=str, default="tmp/grounding_viz",
                        help="Output directory for debug videos")
    parser.add_argument("--viz_camera", type=str, default="robot0_agentview_left",
                        help="Camera used for debug video rendering")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process (0 = all)")
    parser.add_argument("--start_ep", type=int, default=0,
                        help="Starting episode index")
    parser.add_argument(
        "--subsets", nargs="+",
        default=["target_no_nav", "pretrain_human300_no_navigation"],
        help="DATASET_SOUP_REGISTRY subset names to process. "
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
        print(f"[info] Visualisation enabled -> {viz_dir}")
        if args.viz_camera not in CAMERA_NAMES:
            print(f"[warn] Unknown viz camera '{args.viz_camera}', fallback to {CAMERA_NAMES[0]}")
            args.viz_camera = CAMERA_NAMES[0]

    if args.dataset:
        dataset_paths = [Path(args.dataset)]
        print(f"[info] Single-dataset mode: {dataset_paths[0]}")
    else:
        allowed_paths: set[str] | None = None
        if args.subsets and args.subsets != [""]:
            allowed_paths = resolve_subset_paths(args.subsets)
            print(f"[info] Filtering to subsets: {args.subsets} ({len(allowed_paths)} dataset paths)")
        dataset_paths = discover_lerobot_datasets(datasets_root, allowed_paths=allowed_paths)
        print(f"[info] Found {len(dataset_paths)} datasets to process")

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

    total_saved = 0
    total_skipped = 0
    total_failed = 0

    for dataset_i, dataset_path in enumerate(dataset_paths, 1):
        dataset_path = Path(dataset_path)
        print(f"\n[dataset {dataset_i}/{len(dataset_paths)}] {dataset_path}")

        if args.dataset:
            try:
                rel = dataset_path.relative_to(datasets_root)
                result_dataset_dir = output_dir / rel
            except ValueError:
                result_dataset_dir = output_dir / dataset_path.name
        else:
            rel = dataset_path.relative_to(datasets_root)
            result_dataset_dir = output_dir / rel
        result_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Discover episodes
        episodes = LU.get_episodes(dataset_path)
        total = len(episodes)
        print(f"[info] Found {total} episodes")

        end_ep = total if args.max_episodes <= 0 else min(args.start_ep + args.max_episodes, total)
        ep_indices = list(range(args.start_ep, end_ep))
        if ep_indices:
            print(f"[info] Processing episodes {ep_indices[0]} → {ep_indices[-1]}")
        else:
            print("[info] No episodes selected by current start/max settings, skipping")
            continue

        # Build env once, reuse across episodes
        print("[info] Building environment ...")
        t0 = time.time()
        env = build_env(dataset_path)
        print(f"[info] Env built in {time.time() - t0:.1f}s")

        saved_count = 0
        skipped_count = 0
        failed_count = 0
        viz_done_for_dataset = False

        for ep_idx in tqdm(ep_indices, desc="Extracting"):
            # Output path:
            # <output>/<dataset-relative>/extras/episode_XXXXXX/annotations.json
            ep_dir = result_dataset_dir / "extras" / f"episode_{ep_idx:06d}"
            ann_path = ep_dir / "annotations.json"
            if ann_path.exists():
                skipped_count += 1
                continue
            try:
                if viz_dir is not None and not viz_done_for_dataset:
                    ep_data, viz_frames = extract_episode_with_viz_frames(
                        env, dataset_path, ep_idx
                    )
                else:
                    ep_data = extract_episode(env, dataset_path, ep_idx)
                    viz_frames = None

                ep_dir.mkdir(parents=True, exist_ok=True)
                with open(ann_path, "w") as f:
                    json.dump(ep_data, f)
                saved_count += 1

                if viz_dir is not None and (viz_frames is not None) and not viz_done_for_dataset:
                    dataset_tag = "__".join(dataset_path.parts[-5:-1])
                    vid_path = viz_dir / f"{dataset_tag}_ep{ep_idx:04d}_3cams.mp4"
                    try:
                        render_episode_video(
                            viz_frames, ep_data, vid_path, fps=20
                        )
                        tqdm.write(f"  [viz] {vid_path}")
                    except Exception as e:
                        tqdm.write(f"  [viz-error] ep {ep_idx}: {e}")
                    viz_done_for_dataset = True
            except Exception as e:
                failed_count += 1
                print(f"\n[warn] Episode {ep_idx} failed: {e}")
                continue

        env.close()
        print(f"[done] Saved {saved_count}, skipped {skipped_count}, failed {failed_count}")
        print(f"  Output: {result_dataset_dir / 'extras'}/episode_*/annotations.json")

        total_saved += saved_count
        total_skipped += skipped_count
        total_failed += failed_count

    print(f"\n[done] All datasets finished. Saved={total_saved}, skipped={total_skipped}, failed={total_failed}")


if __name__ == "__main__":
    main()
