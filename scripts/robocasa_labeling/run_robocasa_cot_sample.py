#!/usr/bin/env python3
"""RoboCasa CoT SAMPLE: 1 episode per task. Same flow as libero/VLA-Arena
(Stage1 instruction->subtask plan + Stage2 triple-segment + per-segment VLM
subtask labeling + weighted-LNDS order correction), adapted for the cosmos
lerobot format (state-9: gripper_qpos[0:2], eef_pos[2:5], eef_quat[5:9];
video key robot0_agentview_left_image). Renders a full-COT review video.
gripper_2d/bbox grounding is a follow-up after subtask quality is confirmed.
"""
import sys, json, argparse, glob
from pathlib import Path
import numpy as np, cv2
from PIL import Image

SCR = "/ssd/sxu/workspace/code/robocasa/scripts"
for p in [f"{SCR}/libero_labeling", f"{SCR}/robocasa_labeling", f"{SCR}/vla_arena_labeling"]:
    sys.path.insert(0, p)

from utils import QwenVLLM                                     # robocasa_labeling
from generate_instruction_subtasks import decompose_instruction_vlm, extract_keyframes  # Stage1
from label_vla_arena_episodes import label_one_episode        # Stage2 (video_key param)

# libero256 state-8 layout: eef_pos[0:3], axisangle[3:6], gripper[6:8]
# (cosmos state-9 was [2,3,4]/[5,6,7,8]/[0,1])
SIDX, OIDX, GIDX = [0, 1, 2], [3, 4, 5], [6, 7]
VIDEO_KEY = "robot0_agentview_left_image"
FPS = 20


def video_file(taskdir, ep, key):
    return Path(taskdir) / "videos" / f"chunk-{ep // 1000:03d}" / key / f"episode_{ep:06d}.mp4"


def ep0_lang(taskdir, ep):
    for l in open(Path(taskdir) / "meta" / "episodes.jsonl"):
        r = json.loads(l)
        if r["episode_index"] == ep:
            return (r.get("tasks") or [""])[0]
    return ""


def read_frames(vp):
    cap = cv2.VideoCapture(str(vp)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def wrap(text, width, font, scale, thick):
    words = text.split(); lines = []; cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if cv2.getTextSize(t, font, scale, thick)[0][0] > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    if cur: lines.append(cur)
    return lines


def render_cot(cot, frames, out_path, scale=3):
    H = W = frames[0].shape[0] * scale
    panelW = 360
    font = cv2.FONT_HERSHEY_SIMPLEX
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (W + panelW, H))
    fr = cot["frames"]
    long_plan = fr[len(fr) // 2].get("assistant_plan_level", "")
    for i, frame in enumerate(frames):
        if i >= len(fr): break
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((H, panelW, 3), np.uint8)
        y = 24
        def put(txt, color=(255, 255, 255), sc=0.5, th=1, dy=20):
            nonlocal y
            for ln in wrap(txt, panelW - 16, font, sc, th):
                cv2.putText(panel, ln, (8, y), font, sc, color, th, cv2.LINE_AA); y += dy
        put(f"frame {i} seg={fr[i].get('segment_id')}", (160, 160, 160), 0.45)
        put("INSTRUCTION:", (120, 200, 255), 0.45); put(cot.get("instruction", "")[:80], (200, 230, 255), 0.45)
        y += 6; put("SUBTASK:", (120, 255, 160), 0.5); put(fr[i].get("subtask", ""), (180, 255, 200), 0.5)
        y += 4; put("REASON:", (200, 200, 120), 0.42); put(fr[i].get("reason", ""), (220, 220, 180), 0.42)
        y += 8; put("LONG PLAN:", (255, 180, 120), 0.42)
        for ln in long_plan.replace("LONG PLAN:", "").strip().split("\n"):
            put(ln.strip(), (230, 200, 170), 0.4, dy=16)
        vw.write(cv2.cvtColor(np.hstack([img.astype(np.uint8), panel]), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_v01_cosmos_lerobot")
    ap.add_argument("--output", default="/ssd/sxu/workspace/sizhe/datasets/robocasa_cot_sample")
    ap.add_argument("--api_url", default="http://localhost:8110/v1/chat/completions")
    ap.add_argument("--model_name", default="Qwen3.5-9B")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--ep", type=int, default=0)
    a = ap.parse_args()
    model = QwenVLLM(api_url=a.api_url, model_name=a.model_name,
                     chat_template_kwargs={"enable_thinking": False})
    tasks = a.tasks or sorted(d.name for d in Path(a.data_root).iterdir() if d.is_dir())
    out = Path(a.output); (out / "viz").mkdir(parents=True, exist_ok=True)
    for task in tasks:
        taskdir = Path(a.data_root) / task
        if not (taskdir / "meta" / "episodes.jsonl").exists():
            print(f"[skip] {task}"); continue
        lang = ep0_lang(taskdir, a.ep)
        vp = video_file(taskdir, a.ep, VIDEO_KEY)
        kf = extract_keyframes(vp, 1.0)
        subtasks = decompose_instruction_vlm(lang, kf, model) or ["approach", "interact", "complete"]
        print(f"[{task}] lang={lang!r} subtasks={subtasks}", flush=True)
        payload = label_one_episode(
            suite_dir=taskdir, suite_name=task, ep_idx=a.ep, instruction=lang,
            subtask_list=subtasks, fps=FPS, spatial_indices=SIDX, orient_indices=OIDX,
            gripper_indices=GIDX, model=model, video_key=VIDEO_KEY)
        if payload is None:
            print(f"[{task}] EMPTY"); continue
        ed = out / task / "extras" / f"episode_{a.ep:06d}"; ed.mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(ed / "cot_annotations.json", "w"), ensure_ascii=False, indent=2)
        frames = read_frames(vp)
        render_cot(payload, frames, out / "viz" / f"{task}_ep{a.ep:06d}_cot.mp4")
        runs = []
        for f in payload["frames"]:
            if not runs or runs[-1] != f["subtask"]: runs.append(f["subtask"])
        print(f"[{task}] DONE segs={payload.get('segment_count')} subtask_runs={len(runs)} -> viz", flush=True)


if __name__ == "__main__":
    main()
