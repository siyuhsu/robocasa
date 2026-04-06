# Plan: RoboCasa CoT 标注流水线

## TL;DR
参照 DROID 标注流水线（`scripts/droid_labeling/`），为 RoboCasa 数据集实现完整的 Chain-of-Thought 标注系统。利用 MuJoCo 仿真器直接提取 bbox 和 gripper 2D 位置（已在 `visualize_episode.py` 中实现），再通过 HDBSCAN 分段 + Qwen3-VL-4B VLM 标注子任务，最终生成可训练的 CoT JSON 数据集。先在 PickPlaceCounterToCabinet(502 episodes) 上可视化验证，确认无误后大规模标注。

---

## Phase 1: 基础设施 — VLM 服务 & 工具函数

### Step 1: 创建 VLM 启动脚本
- 文件: `scripts/robocasa_labeling/start_vlm.sh`
- 使用 vLLM 在 GPU 6 上启动 Qwen3-VL-4B
- 模型路径: `/data/app/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17`
- GPU 6 已有 46GB 占用（A100 80GB），VLM 4B 约需 ~10GB，可以共存
- 命令: `CUDA_VISIBLE_DEVICES=6 vllm serve <model_path> --port 8100 --max-model-len 4096 --limit-mm-per-prompt image=20`

### Step 2: 创建工具函数模块
- 文件: `scripts/robocasa_labeling/utils.py`
- 内容:
  - `QwenVLLM` 类 (复用 DROID 的 vLLM client，改端口为 8100)
  - `segment_traj()` 分段函数 (复用 HDBSCAN)
  - `describe_move()` 动作描述函数 (**需适配 RoboCasa 12D action!** DROID 是 7D)
  - `get_key_frames()` 关键帧采样 (复用)

**RoboCasa vs DROID 状态差异**:
- DROID state: 6D cartesian (x,y,z,rx,ry,rz radians) + 1D gripper
- RoboCasa state: 16D = base_pos(3) + base_quat(4) + eef_pos_rel(3) + eef_quat_rel(4) + gripper_qpos(2)
- RoboCasa action: 12D = base_motion(4) + control_mode(1) + eef_pos(3) + eef_rot(3) + gripper_close(1)
- **分段时使用**: eef_pos_relative(3) + eef_quat_relative(4) = 7D，类似 DROID 的 cartesian position
- **describe_move 适配**: 使用 action 中的 eef_position(3) + eef_rotation(3) + gripper_close(1) = 7D

## Phase 2: 数据提取 — bbox + gripper 2D + state

### Step 3: 创建数据提取脚本
- 文件: `scripts/robocasa_labeling/extract_annotations.py`
- 功能: 遍历 episodes，通过 MuJoCo sim 提取每帧的:
  - **gripper 2D** (3个相机各一个): 复用 `visualize_episode.py` 的 `project_to_pixel()`
  - **task object bbox** (3个相机各一个): 复用 `segmentation_bbox()`
  - **distractor object bbox** (3个相机各一个): 复用 `segmentation_bbox()`
  - **observation.state**: 从 parquet 读取 16D state
  - **action**: 从 parquet 读取 12D action (通过 `LU.get_episode_actions()`)
  - **ep_meta**: 从 extras 读取 (lang, object_cfgs, etc.)
- 输出: `annotations/PickPlaceCounterToCabinet/extracted_episodes.json`
  ```json
  {
    "task_name|ep_idx": {
      "instruction": "Pick the honey bottle...",
      "ep_meta": {...},
      "num_frames": 246,
      "cameras": {
        "robot0_agentview_left": {
          "gripper_2d": [[x,y], ...],
          "task_obj_bbox": [[x1,y1,x2,y2], ...],
          "distractor_bboxes": [[[x1,y1,x2,y2], ...], ...]
        },
        "robot0_agentview_right": {...},
        "robot0_eye_in_hand": {...}
      },
      "states": [...],
      "actions": [...]
    }
  }
  ```
- 依赖: 需要构建 env、reset_to 每个 episode 的初始状态、逐帧 reset_to + render segmentation
- **优化**: 不需要渲染RGB图像，只渲染 segmentation map (更快)

### Step 4: 可视化验证提取结果
- 文件: `scripts/robocasa_labeling/visualize_annotations.py`
- 功能: 读取提取的 annotations，将 bbox/gripper 画到视频帧上
- 从 LeRobot 的 MP4 视频中解码帧（避免再次 MuJoCo 渲染）
- 输出: 少量验证视频（3-5个episodes）到 `tmp/annotation_viz/`

## Phase 3: 子任务标注 — HDBSCAN + VLM

### Step 5: 创建子任务生成脚本
- 文件: `scripts/robocasa_labeling/generate_subtasks.py`
- 流程:
  1. 读取 parquet 中的 states (eef_pos_relative)
  2. HDBSCAN 分段 (复用 segment_traj)
  3. 从 LeRobot MP4 视频中提取关键帧图片
  4. 调用 Qwen3-VL VLM 生成子任务标注
  5. 输出: `annotations/subtask_results/subtasks_PickPlaceCounterToCabinet.json`
- checkpoint resume 支持
- **RoboCasa 特殊处理**: 分段用 state[7:14] (eef_pos_rel + eef_quat_rel)

### Step 6: 可视化子任务分割结果
- 在 visualize_annotations.py 中增加子任务分段标注的可视化
- 不同子任务用不同颜色标记

## Phase 4: 数据集生成 — CoT 训练数据

### Step 7: 创建数据集生成脚本
- 文件: `scripts/robocasa_labeling/create_dataset.py`
- 参考: `create_dataset_droid.py`
- 类结构:
  - `RawSample`: 适配 RoboCasa 的 16D state / 12D action
  - `Sample`: 输出格式与 DROID 一致
- 三种策略: single_policy / multiple_policy / aug_multiple_policy
- **关键适配点**:
  - `get_position_change()`: 使用 action[5:8](eef_pos) + action[8:11](eef_rot) + action[11](gripper) = 7D
  - `describe_move()`: 同 DROID 的 7D 格式
  - `gripper_2d`: 从 extracted annotations 读取（不再需要单独文件）
  - `bboxes`: 从 extracted annotations 读取
  - 三个相机的 gripper 和 bbox 都包含

### Step 8: 可视化最终训练样本
- 在 visualize_annotations.py 中增加完整 CoT 样本的可视化
- 显示: LONG PLAN, SHORT PLAN, gripper trajectory, bbox, movement text

## Phase 5: 验证 & 大规模标注

### Step 9: 小规模验证 (5 episodes)
- 运行 extract → subtask → dataset 全流程
- 检查输出 JSON 格式、数值合理性
- 生成验证视频

### Step 10: 大规模标注 (502 episodes)
- PickPlaceCounterToCabinet 全量标注
- 统计标注质量

---

## 关键文件

| 文件 | 用途 | 参考 |
|------|------|------|
| `scripts/robocasa_labeling/start_vlm.sh` | VLM 启动脚本 | — |
| `scripts/robocasa_labeling/utils.py` | 分段/VLM/描述工具 | `scripts/droid_labeling/utils.py` |
| `scripts/robocasa_labeling/extract_annotations.py` | MuJoCo bbox/gripper 提取 | `scripts/visualize_episode.py` |
| `scripts/robocasa_labeling/generate_subtasks.py` | VLM 子任务标注 | `scripts/droid_labeling/generate_subtasks_droid.py` |
| `scripts/robocasa_labeling/create_dataset.py` | CoT JSON 数据集生成 | `scripts/droid_labeling/create_dataset_droid.py` |
| `scripts/robocasa_labeling/visualize_annotations.py` | 各阶段可视化验证 | — |

## 参考代码

- `scripts/visualize_episode.py`:
  - `build_env()`, `reset_to()` — 环境构建和状态重置
  - `project_to_pixel()` — 3D→2D 投影
  - `segmentation_bbox()` — 分割图 bbox 提取
  - `find_task_body_names()`, `get_subtree_geom_ids()` — 目标体识别
  - `render_frame()` — 3相机渲染 + 标注叠加
- `scripts/droid_labeling/create_dataset_droid.py`:
  - `RawSample` — 样本数据结构
  - `Sample` — 输出格式
  - template strings — CoT 模板
  - `create_dataset_droid()` — 数据集创建流程
  - visualization functions — 可视化辅助
- `scripts/droid_labeling/utils.py`:
  - `segment_traj()` — HDBSCAN 分段
  - `describe_move()` — 动作描述
  - `get_key_frames()` — 关键帧提取
- `robocasa/utils/lerobot_utils.py`:
  - `get_episode_states()`, `get_episode_actions()`, `get_episode_meta()` — 数据读取
  - `get_env_metadata()`, `get_episodes()` — 环境元数据

## 验证步骤

1. **VLM 服务验证**: 启动后用 curl 测试 `/health` 和 `/v1/models` endpoint
2. **提取验证**: 运行 extract_annotations.py（5 episodes），检查 gripper 2D 坐标是否在 [0,255] 范围内，bbox 是否合理
3. **可视化验证**: 生成验证视频，人眼检查 bbox 和 gripper 位置是否准确
4. **子任务验证**: 检查 VLM 输出格式是否正确，子任务描述是否合理
5. **数据集验证**: 检查最终 JSON 样本的 delta_full_state 数值分布、模板填充是否完整
6. **端到端验证**: 选 2-3 个 episode 运行完整流程并可视化

## 决策记录

- **bbox/gripper 提取**: 使用 MuJoCo segmentation（已在 visualize_episode.py 实现），不使用外部检测器
- **相机覆盖**: 3 个相机全部标注（agentview_left, agentview_right, eye_in_hand）
- **分段方法**: HDBSCAN on eef_pos_relative(3) + eef_quat_relative(4)
- **VLM**: Qwen3-VL-4B on GPU 6, vLLM serve
- **action → describe_move**: 使用 action[5:11] + action[11] = 7D (eef_pos + eef_rot + gripper)
- **输出格式**: 与 DROID 标注 JSON 格式一致，便于下游训练代码复用

## 进一步考虑

1. **eye_in_hand 相机的 bbox**: 手内相机可能看不到 task object（尤其抓取前），bbox 可能频繁为 null → 这是正常的，不影响标注
2. **多 distractor**: RoboCasa ep_meta 可能有多个 distractor (distr_counter, distr_sink 等)，全部标注
3. **base_motion**: RoboCasa 有移动底座，action 中 base_motion(4D) 在 describe_move 中暂不描述，后续可扩展
