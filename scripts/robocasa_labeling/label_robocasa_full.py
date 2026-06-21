#!/usr/bin/env python3
"""Full-scale RoboCasa COT labeling (libero256). Stage-1 instruction->subtask
map computed ONCE per unique lang (consistency across episodes of a task), then
Stage-2 (segment + per-segment VLM subtask + LNDS) over ALL episodes in parallel,
round-robin across multiple vLLM endpoints. Output per-task cot dirs."""
import sys, json, argparse, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCR = "/ssd/sxu/workspace/code/robocasa/scripts"
for p in [f"{SCR}/libero_labeling", f"{SCR}/robocasa_labeling", f"{SCR}/vla_arena_labeling"]:
    sys.path.insert(0, p)
from utils import QwenVLLM
from generate_instruction_subtasks import decompose_instruction_vlm, extract_keyframes
from label_vla_arena_episodes import label_one_episode

SIDX, OIDX, GIDX = [0, 1, 2], [3, 4, 5], [6, 7]   # libero256 state-8
VIDEO_KEY = "robot0_agentview_left_image"
FPS = 20


def ep_lang_map(taskdir):
    out = {}
    for l in open(Path(taskdir) / "meta" / "episodes.jsonl"):
        r = json.loads(l); out[r["episode_index"]] = (r.get("tasks") or [""])[0]
    return out


def video_file(taskdir, ep):
    return Path(taskdir) / "videos" / f"chunk-{ep // 1000:03d}" / VIDEO_KEY / f"episode_{ep:06d}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_v01_libero256_lerobot")
    ap.add_argument("--output", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_cot_lib256_full")
    ap.add_argument("--api_urls", required=True, help="comma-separated vLLM endpoints")
    ap.add_argument("--model_name", default="Qwen3.5-9B")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    urls = [u.strip() for u in a.api_urls.split(",") if u.strip()]
    clients = [QwenVLLM(api_url=u, model_name=a.model_name, chat_template_kwargs={"enable_thinking": False}) for u in urls]
    print(f"[info] {len(clients)} vLLM endpoints, {a.num_workers} workers", flush=True)
    tasks = a.tasks or sorted(d.name for d in Path(a.data_root).iterdir() if d.is_dir())

    # ---- collect (task, ep, lang) + unique langs ----
    jobs = []; lang_kf_src = {}
    for t in tasks:
        td = Path(a.data_root) / t
        if not (td / "meta" / "episodes.jsonl").exists():
            continue
        elm = ep_lang_map(td)
        for ep, lang in elm.items():
            jobs.append((t, ep, lang))
            if lang not in lang_kf_src:
                lang_kf_src[lang] = (t, ep)
    print(f"[info] {len(jobs)} episodes, {len(lang_kf_src)} unique langs", flush=True)

    # ---- Stage 1: subtask map per unique lang (parallel) ----
    smap = {}
    def s1(item):
        lang, (t, ep) = item
        kf = extract_keyframes(video_file(Path(a.data_root) / t, ep), 1.0)
        m = clients[hash(lang) % len(clients)]
        subs = decompose_instruction_vlm(lang, kf, m)
        return lang, (subs or ["approach", "interact", "complete"])
    with ThreadPoolExecutor(max_workers=a.num_workers) as ex:
        for fut in as_completed([ex.submit(s1, it) for it in lang_kf_src.items()]):
            lang, subs = fut.result(); smap[lang] = subs
    json.dump(smap, open("/tmp/robocasa_full_map.json", "w"), ensure_ascii=False, indent=2)
    print(f"[info] Stage1 map done ({len(smap)} langs)", flush=True)

    # ---- Stage 2: label all episodes (parallel, round-robin) ----
    done = {"ok": 0, "skip": 0, "err": 0}
    def s2(idx_item):
        idx, (t, ep, lang) = idx_item
        td = Path(a.data_root) / t
        ed = Path(a.output) / t / "extras" / f"episode_{ep:06d}"
        outp = ed / "cot_annotations.json"
        if outp.exists() and not a.overwrite:
            return "skip"
        try:
            payload = label_one_episode(suite_dir=td, suite_name=t, ep_idx=ep, instruction=lang,
                                        subtask_list=smap.get(lang, ["approach", "interact", "complete"]),
                                        fps=FPS, spatial_indices=SIDX, orient_indices=OIDX, gripper_indices=GIDX,
                                        model=clients[idx % len(clients)], video_key=VIDEO_KEY)
            if payload is None:
                return "err"
            ed.mkdir(parents=True, exist_ok=True)
            json.dump(payload, open(outp, "w"), ensure_ascii=False, indent=2)
            return "ok"
        except Exception as e:
            print(f"  [err] {t} ep{ep}: {repr(e)[:80]}", flush=True)
            return "err"
    n = 0
    with ThreadPoolExecutor(max_workers=a.num_workers) as ex:
        futs = [ex.submit(s2, (i, j)) for i, j in enumerate(jobs)]
        for fut in as_completed(futs):
            done[fut.result()] += 1; n += 1
            if n % 200 == 0:
                print(f"  [{n}/{len(jobs)}] {done}", flush=True)
    print(f"[DONE] {done}", flush=True)


if __name__ == "__main__":
    main()
