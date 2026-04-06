"""
Create dataset for DROID - adapted from create_dataset.py for Bridge

Clarification: 
(current_image, current_state) go through current_action -> (next_image, next_state)

Usage:
python create_dataset_droid.py main --tag single_policy --split_id 0 --output_dir dataset_droid
python create_dataset_droid.py main --tag multiple_policy --split_id 0 --output_dir dataset_droid
python create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --output_dir dataset_droid
python GCOT/create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --visualize
python GCOT/create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --visualize --max_viz_episodes 10
python create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --output_dir dataset_droid

"""

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import re
import time
import textwrap
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Union, Optional

# 设置 TensorFlow 使用 CPU，避免占用 GPU 显存（GPU 用于 vLLM）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow_datasets as tfds
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils import describe_move

# 延迟导入可视化依赖（cv2, PIL），仅在可视化时需要
cv2 = None
Image = None
ImageDraw = None
ImageFont = None

def _import_viz_deps():
    """延迟导入可视化依赖"""
    global cv2, Image, ImageDraw, ImageFont
    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if Image is None:
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        Image = _Image
        ImageDraw = _ImageDraw
        ImageFont = _ImageFont


# === DROID-specific configurations ===
DROID_DATA_PATH = "/scratch/sh89/sx0401/data/"
DROID_VERSION = "1.0.1"
SUBTASK_RESULTS_DIR = "GCOT/subtask_results_droid"
EMBODIED_FEATURES_DIR = "data"  # Contains embodied features splits
EMBODIED_FEATURES_JSON = "data/droid_embodied_features.json"  # Fallback full file
BBOX_DATA_FILE = "bounding_boxes/data/merged_all_bboxes.json"  # BBox annotations


input_template = (
    "What action should the robot take to achieve the instruction\n"
    "INSTRUCTION: \n{instruction}\n"
    "CURRENT GRIPPER: {gripper_2d}\n"
)

# LONG PLAN template (fixed at beginning, doesn't change across steps)
long_plan_template = "LONG PLAN:\n{long_plan}\n"

position_level_template = (
    "NEXT GRIPPER: {gripper_2d_next}\n"
)

object_level_template = "OBJECT:\n{objects}\n"

# SHORT PLAN template (updated every step)
short_plan_template = """SHORT PLAN:
Current Positions: {current_positions}
Subtask: {subtask}, move from {subtask_trajectory}
Subtask Reasoning: {reasoning}
Current Movement: {current_movement}
Subtask Movement: {subtask_movement}
"""


class RawSample(BaseModel, extra="allow", arbitrary_types_allowed=True):
    sample_dir: str  # file_path|episode_id
    instruction: str
    highlevel_plan: Union[str, dict]  # highlevel plan key starts from 1
    segments: list[int]  # segment starts from 1 [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    gripper_2d: list[list[int]]
    bboxes: list[list] = []  # BBox annotations: [[[conf, name, [x1,y1,x2,y2]], ...], ...]
    segment_index_to_image_index: dict = dict()
    full_state: np.ndarray  # DROID: cartesian_position (6D: x, y, z, rx, ry, rz in radians)
    gripper_state: np.ndarray  # DROID: gripper_position (1D: gripper opening, 1.0=closed, 0.0=open)
    action_policy: np.ndarray  # DROID: action (7D: [6x joint velocities, 1x gripper position])
    valid: bool = False

    def prepare_segments(self):
        """
        segment index starts from 1 {1: 0, 2: 6, 3: 11, 4: 20, 5: 28, 6: 33, -1: 38}
        the -1 index indicates the last frame index
        """
        count = 0
        init = None
        _segments = []
        for i, oseg in enumerate(self.segments):
            if oseg != init:
                init = oseg
                count += 1
                self.segment_index_to_image_index[count] = i
            _segments.append(count)

        self.segment_index_to_image_index[-1] = len(self.segments) - 1
        self.segments = _segments

    def check_valid(self):
        if self.highlevel_plan == "NA":
            return "no response"

        search_result = re.search(r"\{[\s\S]*\}", self.highlevel_plan)
        if search_result == None:
            return "no dict"
        try:
            match = ast.literal_eval(search_result.group(0))
        except Exception:
            return "no valid dict"

        # Check Format
        for k, v in match.items():
            if len(v) != 2:
                return "wrong format"

        self.highlevel_plan = match

        self.prepare_segments()

        if len(match) != max(self.segments):
            return "wrong segment number"

        self.valid = True

    def format_plan(self):
        """Format the full episode plan as a readable string, removing consecutive duplicates"""
        highlevel_plan = OrderedDict(self.highlevel_plan)
        plan_lines = []
        prev_goal = None
        renumbered_idx = 0
        
        for seg_idx, (goal, reason) in highlevel_plan.items():
            # Skip consecutive duplicate goals
            if goal == prev_goal:
                continue
            
            renumbered_idx += 1
            plan_lines.append(f"  {renumbered_idx}. {goal}")
            prev_goal = goal
        
        return "\n".join(plan_lines)

    def get_initial_positions(self):
        """
        获取初始帧（frame 0）的 gripper 位置和所有物体的 bbox
        
        Returns:
            str: "Gripper [x, y], object1 [x1, y1, x2, y2], object2 [x1, y1, x2, y2], ..."
        """
        parts = []
        
        # Gripper position at frame 0
        gripper_pos = self.get_gripper_position(0)
        parts.append(f"Gripper [{gripper_pos[0]}, {gripper_pos[1]}]")
        
        # Objects bbox at frame 0
        if self.bboxes and len(self.bboxes) > 0:
            frame_bboxes = self.bboxes[0]
            for bbox_item in frame_bboxes:
                if len(bbox_item) >= 3:
                    conf, name, bbox = bbox_item[0], bbox_item[1], bbox_item[2]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        parts.append(f"{name} [{x1}, {y1}, {x2}, {y2}]")
        
        return ", ".join(parts)

    def get_current_positions(self, frame_index):
        """
        获取当前帧的 gripper 位置和所有物体的 bbox 中心位置
        
        Args:
            frame_index: 帧索引
        
        Returns:
            str: "Gripper [x, y], object1 [cx, cy], object2 [cx, cy], ..."
        """
        parts = []
        
        # Gripper position at current frame
        gripper_pos = self.get_gripper_position(frame_index)
        parts.append(f"Gripper [{gripper_pos[0]}, {gripper_pos[1]}]")
        
        # Objects bbox center at current frame
        if self.bboxes and frame_index < len(self.bboxes):
            frame_bboxes = self.bboxes[frame_index]
            for bbox_item in frame_bboxes:
                if len(bbox_item) >= 3:
                    conf, name, bbox = bbox_item[0], bbox_item[1], bbox_item[2]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        # 计算 bbox 中心点
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        parts.append(f"{name} [{cx}, {cy}]")
        
        return ", ".join(parts)

    def get_dynamic_subtask_trajectory(self, frame_index, segment_index_next):
        """
        获取动态更新的子任务轨迹（起点和中点随当前帧更新，终点固定）
        
        Args:
            frame_index: 当前帧索引
            segment_index_next: 下一个子任务的 segment index (或 -1 表示结束)
        
        Returns:
            str: "[x1, y1], [x2, y2], [x3, y3]" (当前位置, 中间位置, 终点)
        """
        # 起点：当前帧的 gripper 位置
        start_pos = self.get_gripper_position(frame_index)
        
        # 终点：子任务结束位置（固定）
        end_frame = self.segment_index_to_image_index.get(segment_index_next, len(self.gripper_2d) - 1)
        end_frame = min(end_frame, len(self.gripper_2d) - 1)
        end_pos = self.get_gripper_position(end_frame)
        
        # 中点：当前位置到终点的中间帧位置
        mid_frame = (frame_index + end_frame) // 2
        mid_pos = self.get_gripper_position(mid_frame)
        
        return f"[{start_pos[0]}, {start_pos[1]}], [{mid_pos[0]}, {mid_pos[1]}], [{end_pos[0]}, {end_pos[1]}]"

    def format_short_plan(self, frame_index, goal, reason, segment_index, segment_index_next, current_movement_text):
        """
        Format SHORT PLAN for current step
        
        Args:
            frame_index: 当前帧索引
            goal: 当前子任务目标
            reason: 当前子任务推理
            segment_index: 当前子任务 segment index
            segment_index_next: 下一个子任务 segment index
            current_movement_text: 当前 step 到下一个 step 的移动描述
        
        Returns:
            str: SHORT PLAN 格式化文本
        """
        # Current Positions
        current_positions = self.get_current_positions(frame_index)
        
        # Subtask trajectory (起点和中点动态更新，终点固定)
        subtask_trajectory = self.get_dynamic_subtask_trajectory(frame_index, segment_index_next)
        
        # Subtask Movement: 从当前帧到子阶段终点的移动（动态更新）
        end_frame = self.segment_index_to_image_index.get(segment_index_next, len(self.gripper_2d) - 1)
        end_frame = min(end_frame, len(self.gripper_2d) - 1)
        delta_to_end = self.get_position_change(frame_index, end_frame)
        subtask_movement_text = describe_move(delta_to_end)
        
        short_plan = short_plan_template.format(
            current_positions=current_positions,
            subtask=goal,
            subtask_trajectory=subtask_trajectory,
            reasoning=reason,
            current_movement=current_movement_text,
            subtask_movement=subtask_movement_text,
        )
        
        return short_plan

    def get_subtask_trajectory(self, segment_index, segment_index_next):
        """
        获取子任务的 gripper 轨迹（起点、中点、终点）
        
        Args:
            segment_index: 当前子任务的 segment index
            segment_index_next: 下一个子任务的 segment index (或 -1 表示结束)
        
        Returns:
            str: "[x1, y1], [x2, y2], [x3, y3]" (起点, 中点, 终点)
        """
        start_frame = self.segment_index_to_image_index.get(segment_index, 0)
        end_frame = self.segment_index_to_image_index.get(segment_index_next, len(self.gripper_2d) - 1)
        
        # 确保 end_frame 有效
        end_frame = min(end_frame, len(self.gripper_2d) - 1)
        
        # 起点
        start_pos = self.get_gripper_position(start_frame)
        
        # 终点
        end_pos = self.get_gripper_position(end_frame)
        
        # 中点（取中间帧的位置）
        mid_frame = (start_frame + end_frame) // 2
        mid_pos = self.get_gripper_position(mid_frame)
        
        return f"[{start_pos[0]}, {start_pos[1]}], [{mid_pos[0]}, {mid_pos[1]}], [{end_pos[0]}, {end_pos[1]}]"

    def format_long_plan(self):
        """
        Format LONG PLAN with Initial Positions and Task Plans
        
        Format:
        Initial Positions: Gripper [x, y], object1 [x1, y1, x2, y2], ...
        Task Plans: 1. subtask, move from [x1, y1], [x2, y2], [x3, y3]; 2. ...
        """
        # Initial Positions
        initial_positions = self.get_initial_positions()
        
        # Task Plans with trajectories
        highlevel_plan = OrderedDict(self.highlevel_plan)
        plan_items = list(highlevel_plan.items())
        task_plans = []
        prev_goal = None
        renumbered_idx = 0
        
        for plan_index, (seg_idx, (goal, reason)) in enumerate(plan_items):
            # Skip consecutive duplicate goals
            if goal == prev_goal:
                continue
            
            # Get segment index
            if type(seg_idx) == str:
                segment_index = int(re.findall(r"\d+", seg_idx)[0])
            else:
                segment_index = seg_idx
            
            # Get next segment index
            if segment_index == len(highlevel_plan):
                segment_index_next = -1
            elif plan_index < len(plan_items) - 1:
                next_seg_idx = plan_items[plan_index + 1][0]
                if type(next_seg_idx) == str:
                    segment_index_next = int(re.findall(r"\d+", next_seg_idx)[0])
                else:
                    segment_index_next = next_seg_idx
            else:
                segment_index_next = -1
            
            # Get trajectory for this subtask
            trajectory = self.get_subtask_trajectory(segment_index, segment_index_next)
            
            renumbered_idx += 1
            task_plans.append(f"{renumbered_idx}. {goal}, move from {trajectory}")
            prev_goal = goal
        
        task_plans_text = "; ".join(task_plans)
        
        # Combine
        long_plan = f"Initial Positions: {initial_positions}\nTask Plans: {task_plans_text}"
        
        return long_plan

    def get_samples_multiple_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        # LONG PLAN is fixed at beginning, computed once
        long_plan_text = self.format_long_plan()
        
        for plan_index in range(len(highlevel_plan)):
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            # the last segment index should use the -1, check self.prepare_segments
            # note: goal_next, reason_next is actually not needed
            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )
            elif plan_index < len(highlevel_plan) - 1:
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])
            image_index_next = self.segment_index_to_image_index[segment_index_next]

            # user
            image_index = self.segment_index_to_image_index[segment_index]
            image_path = os.path.join(self.sample_dir, f"frame_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            plan_level = long_plan_template.format(long_plan=long_plan_text)
            
            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(image_index_next)
            )

            # delta_full_state 用于记录到下一帧的状态变化
            delta_full_state = self.get_position_change(image_index, image_index_next)
            current_movement_text = describe_move(delta_full_state)
            
            # SHORT PLAN (包含 current movement 和 subtask movement)
            short_plan = self.format_short_plan(
                frame_index=image_index,
                goal=goal,
                reason=reason,
                segment_index=segment_index,
                segment_index_next=segment_index_next,
                current_movement_text=current_movement_text,
            )
            
            # OBJECT level with bbox info
            objects_text = self.get_objects_text(image_index)
            object_level = object_level_template.format(objects=objects_text) if objects_text else ""

            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_plan_level=plan_level,
                assistant_short_plan=short_plan,
                assistant_position_level=position_level,
                assistant_object_level=object_level,
                delta_full_state=delta_full_state.tolist(),
            )
            samples.append(sample.dict())
        return samples

    def get_samples_single_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        # LONG PLAN is fixed at beginning, computed once
        long_plan_text = self.format_long_plan()
        
        for index in range(len(self.segments) - 1):
            plan_index = self.segments[index] - 1
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )
            elif segment_index < len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])

            image_index = index
            image_index_next: int = image_index + 1
            image_index_next_segment = self.segment_index_to_image_index[
                segment_index_next
            ]

            if image_index_next >= len(self.segments):
                continue

            # user
            image_path = os.path.join(self.sample_dir, f"frame_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            plan_level = long_plan_template.format(long_plan=long_plan_text)
            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(image_index_next)
            )
            # delta_full_state 用于记录到下一帧的状态变化
            delta_full_state = self.get_position_change(
                image_index,
                image_index_next,
            )
            current_movement_text = describe_move(delta_full_state)
            
            # SHORT PLAN (包含 current movement 和 subtask movement)
            short_plan = self.format_short_plan(
                frame_index=image_index,
                goal=goal,
                reason=reason,
                segment_index=segment_index,
                segment_index_next=segment_index_next,
                current_movement_text=current_movement_text,
            )
            
            # OBJECT level with bbox info
            objects_text = self.get_objects_text(image_index)
            object_level = object_level_template.format(objects=objects_text) if objects_text else ""
            
            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_plan_level=plan_level,
                assistant_short_plan=short_plan,
                assistant_position_level=position_level,
                assistant_object_level=object_level,
                delta_full_state=delta_full_state.tolist(),
            )

            samples.append(sample.dict())

        return samples

    def get_samples_aug_multiple_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        # LONG PLAN is fixed at beginning, computed once
        long_plan_text = self.format_long_plan()
        
        for frame_index in range(len(self.segments) - 1):
            plan_index = self.segments[frame_index] - 1
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )

            elif segment_index < len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])

            image_index_next = self.segment_index_to_image_index[segment_index_next]

            # user
            image_index = frame_index
            image_path = os.path.join(self.sample_dir, f"frame_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            plan_level = long_plan_template.format(long_plan=long_plan_text)
            
            # NEXT GRIPPER 使用下一帧的位置（frame_index + 1），而不是 segment 结束位置
            # 这样 gripper 位置预测更连贯，每帧都预测下一帧的位置
            next_frame_index = frame_index + 1
            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(next_frame_index)
            )
            
            # delta_full_state 用于记录到下一帧的状态变化
            delta_full_state = self.get_position_change(image_index, next_frame_index)
            current_movement_text = describe_move(delta_full_state)
            
            # SHORT PLAN (包含 current movement 和 subtask movement)
            short_plan = self.format_short_plan(
                frame_index=image_index,
                goal=goal,
                reason=reason,
                segment_index=segment_index,
                segment_index_next=segment_index_next,
                current_movement_text=current_movement_text,
            )
            
            # OBJECT level with bbox info
            objects_text = self.get_objects_text(image_index)
            object_level = object_level_template.format(objects=objects_text) if objects_text else ""
            
            assert image_index < next_frame_index
            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_plan_level=plan_level,
                assistant_short_plan=short_plan,
                assistant_position_level=position_level,
                assistant_object_level=object_level,
                delta_full_state=delta_full_state.tolist(),
            )

            samples.append(sample.dict())

        return samples

    def get_position_change(self, image_index, image_index_next):
        """
        计算状态变化，用于 describe_move 生成自然语言描述
        
        DROID 格式: 
        - cartesian_position (6D): [x, y, z, rx, ry, rz] (欧拉角，单位：弧度)
        - gripper_position (1D): [gripper] (1.0=closed, 0.0=open)
        
        输出格式: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_next]
        
        注意：
        1. 位置使用差分 (delta)，单位为米 (m)
        2. 旋转：DROID 直接使用欧拉角 (rx, ry, rz)，单位为弧度
        3. DROID gripper: 1.0=closed, 0.0=open
        4. describe_move 期望: gripper > 0.5 为 "open gripper"，需要反转
        """
        # 1. 位置差分（米）
        delta_xyz = (
            self.full_state[image_index_next][:3] - self.full_state[image_index][:3]
        )
        
        # 2. 旋转差分（弧度）
        # DROID 的 cartesian_position[3:6] 直接是欧拉角 [rx, ry, rz]
        delta_rotation = (
            self.full_state[image_index_next][3:6] - self.full_state[image_index][3:6]
        )
        
        # 3. Gripper 状态
        # DROID gripper_position: 1.0 = closed (闭合), 0.0 = open (张开)
        # describe_move 期望: gripper > 0.5 为 "open gripper"
        # 因此需要反转: gripper_for_describe = 1.0 - gripper_state
        gripper_state_next = float(self.gripper_state[image_index_next][0])
        gripper_for_describe = 1.0 - gripper_state_next
        
        # 拼接为 [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        delta_full_state = np.concatenate(
            (delta_xyz, delta_rotation, [gripper_for_describe])
        )
        
        return delta_full_state

    def get_gripper_position(self, index):
        gripper2d = self.gripper_2d[index]
        # Gripper坐标已经在256x256坐标系下（来自merged_all_gripper_positions.json）
        # 直接返回，无需缩放
        return [int(gripper2d[0]), int(gripper2d[1])]

    def get_objects_text(self, index):
        """
        生成 OBJECT 字段文本
        格式: object_name: [x1,y1], [x2,y2]
        
        Args:
            index: 帧索引
        Returns:
            str: 格式化的物体文本
        """
        if not self.bboxes or index >= len(self.bboxes):
            return ""
        
        frame_bboxes = self.bboxes[index]
        if not frame_bboxes:
            return ""
        
        object_lines = []
        for bbox_item in frame_bboxes:
            if len(bbox_item) >= 3:
                conf, name, bbox = bbox_item[0], bbox_item[1], bbox_item[2]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # 格式: object_name: [x1,y1], [x2,y2]
                    object_lines.append(f"  {name}: [{x1},{y1}], [{x2},{y2}]")
        
        return "\n".join(object_lines) if object_lines else ""


class Sample(BaseModel):  # one trajectory
    current_image_path: str
    user: str
    assistant_plan_level: str  # LONG PLAN (fixed at beginning)
    assistant_short_plan: str  # SHORT PLAN (updated every step)
    assistant_position_level: str
    assistant_object_level: str = ""  # OBJECT field with bbox info
    delta_full_state: list
    delta_full_state_norm: list = []


def normalize_movement(tag, samples: Sample, overwrite=False, output_dir="dataset_droid"):
    all_movements = []
    for sample in samples:
        all_movements.append(sample["delta_full_state"])
    all_movements = np.array(all_movements)
    mean = np.mean(all_movements, axis=0)
    std = np.std(all_movements, axis=0)
    low = np.percentile(all_movements, 1, axis=0)
    high = np.percentile(all_movements, 99, axis=0)

    percentiles = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "Q1": low.tolist(),
        "Q99": high.tolist(),
    }
    os.makedirs(f"{output_dir}/{tag}", exist_ok=True)
    with open(f"{output_dir}/{tag}/dataset_statistics.json", "w") as f:
        json.dump(percentiles, f, indent=4)

    all_movements_norm = 2 * (all_movements - low) / (high - low + 1e-8) - 1
    all_movements_norm = np.clip(all_movements_norm, -1, 1)

    for i, sample in enumerate(samples):
        sample["delta_full_state_norm"] = all_movements_norm[i].tolist()
        # Note: movement is now part of SHORT PLAN, no longer a separate field
    return samples


def load_gripper_positions(split_id=None, num_splits=10):
    """
    Load gripper position data from merged file (256x256 coordinate system)
    
    Args:
        split_id: Split ID (0-based), optional - used for filtering by RLDS split
        num_splits: Total number of splits (default: 10)
    
    Returns:
        dict: Gripper position data {file_path: [[x1, y1], [x2, y2], ...]}
    """
    merged_file = "gripper_positions/data/merged_all_gripper_positions.json"
    
    if not os.path.exists(merged_file):
        raise FileNotFoundError(
            f"Merged gripper positions file not found: {merged_file}\n"
            f"Please run merge_camera_and_endtoend_results.py first."
        )
    
    print(f"Loading gripper positions from {merged_file}...")
    start = time.time()
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)
    print(f"✓ Loaded {len(merged_data)} episodes in {time.time() - start:.2f} seconds")
    
    # Extract gripper positions
    # merged_data structure: {file_path: {"cameras": {"exterior_image_1_left": {"gripper_positions": [...], ...}}}}
    gripper_positions = {}
    skipped_no_camera = 0
    skipped_no_gripper = 0
    
    for file_path, episode_data in merged_data.items():
        # Use file_path directly as key (gripper results don't use episode_id)
        # Check camera data
        if "cameras" not in episode_data or "exterior_image_1_left" not in episode_data["cameras"]:
            skipped_no_camera += 1
            continue
        
        camera_data = episode_data["cameras"]["exterior_image_1_left"]
        if "gripper_positions" not in camera_data:
            skipped_no_gripper += 1
            continue
        
        # Convert gripper positions to integers (from 256x256 float coordinates)
        raw_positions = camera_data["gripper_positions"]
        gripper_positions[file_path] = [
            [int(round(pos[0])), int(round(pos[1]))] if pos is not None and len(pos) == 2 else pos
            for pos in raw_positions
        ]
    
    if split_id is not None:
        print(f"  Note: Split filtering will be done by RLDS train[{split_id*100//num_splits}%:{(split_id+1)*100//num_splits}%]")
    
    print(f"✓ Extracted {len(gripper_positions)} gripper position sequences")
    if skipped_no_camera > 0:
        print(f"  Skipped {skipped_no_camera} episodes (no camera data)")
    if skipped_no_gripper > 0:
        print(f"  Skipped {skipped_no_gripper} episodes (no gripper positions)")
    
    return gripper_positions


def load_bbox_data(bbox_file=None):
    """
    Load bounding box data from merged file (256x256 coordinate system)
    
    Args:
        bbox_file: Path to bbox JSON file (default: BBOX_DATA_FILE)
    
    Returns:
        dict: BBox data {file_path: {"bboxes": [[[conf, name, [x1,y1,x2,y2]], ...], ...], ...}}
    """
    if bbox_file is None:
        bbox_file = BBOX_DATA_FILE
    
    if not os.path.exists(bbox_file):
        print(f"Warning: BBox file not found: {bbox_file}")
        print("  BBox annotations will not be available.")
        return {}
    
    print(f"Loading bbox data from {bbox_file}...")
    start = time.time()
    
    # 检查是否有 pickle 缓存文件
    pickle_file = bbox_file.replace('.json', '.pkl')
    if os.path.exists(pickle_file):
        print(f"  Using pickle cache: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            bbox_data = pickle.load(f)
    else:
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
        # 保存 pickle 缓存
        try:
            print(f"  Saving pickle cache for future use...")
            with open(pickle_file, 'wb') as f:
                pickle.dump(bbox_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Warning: Failed to save pickle cache: {e}")
    
    print(f"✓ Loaded {len(bbox_data)} episodes in {time.time() - start:.2f} seconds")
    
    # 提取 bboxes，格式：{file_path: [[frame_bboxes], ...]}
    bbox_positions = {}
    for file_path, episode_data in bbox_data.items():
        if "bboxes" in episode_data:
            bbox_positions[file_path] = episode_data["bboxes"]
    
    print(f"✓ Extracted {len(bbox_positions)} bbox sequences")
    return bbox_positions


def load_droid_subtask_results():
    """Load all DROID subtask results from split files"""
    all_results = {}
    
    # Load all split files
    split_files = [f for f in os.listdir(SUBTASK_RESULTS_DIR) if f.startswith("subtasks_") and f.endswith(".json")]
    
    print(f"Loading DROID subtask results from {len(split_files)} files...")
    for split_file in sorted(split_files):
        file_path = os.path.join(SUBTASK_RESULTS_DIR, split_file)
        try:
            with open(file_path, "r") as f:
                split_data = json.load(f)
                all_results.update(split_data)
        except Exception as e:
            print(f"Warning: Failed to load {split_file}: {e}")
    
    print(f"Loaded {len(all_results)} DROID episodes with subtask annotations")
    return all_results


def create_dataset_droid(gripper_position, split_percent, tag="multiple_policy", 
                         output_dir="dataset_droid", bbox_data=None):
    """
    Create dataset for DROID
    
    Args:
        gripper_position: dict of {file_path|episode_id: gripper_2d_list}
        split_percent: tuple (start, end) for dataset split percentage
        tag: policy type (single_policy, multiple_policy, aug_multiple_policy)
        output_dir: output directory for dataset files
        bbox_data: dict of {file_path: bboxes_list} for bounding box annotations
    """
    os.makedirs(f"{output_dir}/{tag}", exist_ok=True)
    
    # Load DROID subtask results
    droid_plans = load_droid_subtask_results()
    print(f"You choose the tag: {tag}")

    # Load DROID dataset
    start, end = split_percent
    split = f"train[{start}%:{end}%]"
    print(f"Loading DROID dataset split: {split}")
    
    # Set TF to use CPU to avoid GPU memory conflicts
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    dataset = tfds.load(
        "droid",
        data_dir=DROID_DATA_PATH,
        split=split
    )

    sample_list = []
    num_traj = 0
    skipped_no_plan = 0
    skipped_no_gripper = 0
    skipped_no_bbox = 0
    skipped_invalid = 0
    
    # Process episodes without pre-counting (avoid double iteration)
    pbar = tqdm(dataset, desc="Processing episodes", unit="ep")
    for episode in pbar:
        # Extract episode metadata
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
        key = f"{file_path}|{episode_id}"
        
        # Filter: only process success episodes
        if "success" not in file_path.lower():
            continue
        
        # Check if we have subtask annotations (uses file_path|episode_id)
        if key not in droid_plans:
            skipped_no_plan += 1
            continue
        
        plan_data = droid_plans[key]
        instruction = plan_data['instruction']
        segments = plan_data['segments']
        model_output = plan_data['model_output']
        
        # Check if we have gripper 2D positions (uses file_path only)
        gripper_2d = gripper_position.get(file_path, None)
        if gripper_2d is None:
            skipped_no_gripper += 1
            continue
        
        # Get bbox data (uses file_path only)
        bboxes = []
        if bbox_data:
            bboxes = bbox_data.get(file_path, [])
            if not bboxes:
                skipped_no_bbox += 1
        
        # Extract states, gripper states, and actions from episode
        states = []
        gripper_states = []
        actions = []
        for step in episode["steps"]:
            # DROID cartesian_position (6D: x, y, z, rx, ry, rz in radians)
            cart_pos = step["observation"]["cartesian_position"].numpy()
            states.append(cart_pos)
            
            # DROID gripper_position (1D: gripper opening, 1.0=closed, 0.0=open)
            gripper_pos = step["observation"]["gripper_position"].numpy()
            gripper_states.append(gripper_pos)
            
            # DROID action (7D: [6x joint velocities, 1x gripper position])
            action = step["action"].numpy()
            actions.append(action)
        
        full_state = np.array(states)
        gripper_state = np.array(gripper_states)
        action_policy = np.array(actions)
        
        # Validate dimensions
        if len(segments) != len(full_state) or len(full_state) != len(gripper_2d):
            print(f"Warning: Dimension mismatch for {key}")
            print(f"  segments: {len(segments)}, states: {len(full_state)}, gripper: {len(gripper_2d)}")
            skipped_invalid += 1
            continue
        
        # Validate bbox dimensions if available
        if bboxes and len(bboxes) != len(full_state):
            # Truncate to minimum length
            min_len = min(len(bboxes), len(full_state))
            bboxes = bboxes[:min_len]
        
        raw_sample = RawSample(
            sample_dir=key,
            instruction=instruction,
            highlevel_plan=model_output,
            segments=segments,
            gripper_2d=gripper_2d,
            bboxes=bboxes,
            full_state=full_state,
            gripper_state=gripper_state,
            action_policy=action_policy,
        )

        raw_sample.check_valid()
        if not raw_sample.valid:
            skipped_invalid += 1
            continue

        tag_clean = tag.replace("_", "")
        # This condition order is important, do not change
        if "singlepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_single_policy())
        elif "augmultiplepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_aug_multiple_policy())
        elif "multiplepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_multiple_policy())
        else:
            raise AssertionError(f"Unknown tag: {tag}")
        num_traj += 1
        
        # Update progress bar with statistics
        pbar.set_postfix({
            'valid': num_traj,
            'no_plan': skipped_no_plan,
            'no_gripper': skipped_no_gripper,
            'invalid': skipped_invalid
        })
    
    pbar.close()
    
    print(f"\nDataset creation summary:")
    print(f"  Valid trajectories: {num_traj}")
    print(f"  Generated samples: {len(sample_list)}")
    print(f"  Skipped (no plan): {skipped_no_plan}")
    print(f"  Skipped (no gripper): {skipped_no_gripper}")
    print(f"  Skipped (no bbox): {skipped_no_bbox}")
    print(f"  Skipped (invalid): {skipped_invalid}")
    
    return sample_list


def create_dataset_droid_debug(gripper_position, split_percent, tag="multiple_policy", 
                               output_dir="dataset_droid", bbox_data=None, max_episodes=5):
    """
    Create dataset for DROID - DEBUG MODE (limited episodes)
    
    Same as create_dataset_droid but stops after processing max_episodes valid episodes.
    """
    os.makedirs(f"{output_dir}/{tag}", exist_ok=True)
    
    # Load DROID subtask results
    droid_plans = load_droid_subtask_results()
    print(f"Policy tag: {tag}")
    print(f"Debug mode: processing max {max_episodes} episodes")

    # Load DROID dataset
    start, end = split_percent
    split = f"train[{start}%:{end}%]"
    print(f"Loading DROID dataset split: {split}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    dataset = tfds.load(
        "droid",
        data_dir=DROID_DATA_PATH,
        split=split
    )

    sample_list = []
    num_traj = 0
    skipped_no_plan = 0
    skipped_no_gripper = 0
    skipped_no_bbox = 0
    skipped_invalid = 0
    
    pbar = tqdm(dataset, desc="Processing episodes (debug)", unit="ep")
    for episode in pbar:
        # Check if we've reached the limit
        if num_traj >= max_episodes:
            print(f"\nReached max_episodes limit ({max_episodes}), stopping...")
            break
        
        # Extract episode metadata
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
        key = f"{file_path}|{episode_id}"
        
        # Filter: only process success episodes
        if "success" not in file_path.lower():
            continue
        
        # Check if we have subtask annotations
        if key not in droid_plans:
            skipped_no_plan += 1
            continue
        
        plan_data = droid_plans[key]
        instruction = plan_data['instruction']
        segments = plan_data['segments']
        model_output = plan_data['model_output']
        
        # Check gripper 2D positions
        gripper_2d = gripper_position.get(file_path, None)
        if gripper_2d is None:
            skipped_no_gripper += 1
            continue
        
        # Get bbox data
        bboxes = []
        if bbox_data:
            bboxes = bbox_data.get(file_path, [])
            if not bboxes:
                skipped_no_bbox += 1
        
        # Extract states
        states = []
        gripper_states = []
        actions = []
        for step in episode["steps"]:
            cart_pos = step["observation"]["cartesian_position"].numpy()
            states.append(cart_pos)
            gripper_pos = step["observation"]["gripper_position"].numpy()
            gripper_states.append(gripper_pos)
            action = step["action"].numpy()
            actions.append(action)
        
        full_state = np.array(states)
        gripper_state = np.array(gripper_states)
        action_policy = np.array(actions)
        
        # Validate dimensions
        if len(segments) != len(full_state) or len(full_state) != len(gripper_2d):
            skipped_invalid += 1
            continue
        
        # Validate bbox dimensions
        if bboxes and len(bboxes) != len(full_state):
            min_len = min(len(bboxes), len(full_state))
            bboxes = bboxes[:min_len]
        
        raw_sample = RawSample(
            sample_dir=key,
            instruction=instruction,
            highlevel_plan=model_output,
            segments=segments,
            gripper_2d=gripper_2d,
            bboxes=bboxes,
            full_state=full_state,
            gripper_state=gripper_state,
            action_policy=action_policy,
        )

        raw_sample.check_valid()
        if not raw_sample.valid:
            skipped_invalid += 1
            continue

        tag_clean = tag.replace("_", "")
        if "singlepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_single_policy())
        elif "augmultiplepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_aug_multiple_policy())
        elif "multiplepolicy" in tag_clean:
            sample_list.extend(raw_sample.get_samples_multiple_policy())
        else:
            raise AssertionError(f"Unknown tag: {tag}")
        num_traj += 1
        
        pbar.set_postfix({
            'valid': num_traj,
            'target': max_episodes,
        })
    
    pbar.close()
    
    print(f"\nDebug dataset creation summary:")
    print(f"  Valid trajectories: {num_traj}/{max_episodes}")
    print(f"  Generated samples: {len(sample_list)}")
    print(f"  Skipped (no plan): {skipped_no_plan}")
    print(f"  Skipped (no gripper): {skipped_no_gripper}")
    print(f"  Skipped (no bbox): {skipped_no_bbox}")
    print(f"  Skipped (invalid): {skipped_invalid}")
    
    return sample_list


def main(tag="aug_multiple_policy", split_id=0, num_splits=10, 
         output_dir="dataset_droid", bbox_file=None, 
         visualize=False, max_viz_episodes=5, viz_output_dir=None):
    """
    Create DROID dataset
    
    Args:
        tag: Policy type (single_policy, multiple_policy, aug_multiple_policy)
        split_id: Split ID to process (0-based), REQUIRED
        num_splits: Total number of splits (default: 10)
        output_dir: Output directory for dataset files (default: dataset_droid)
        bbox_file: Path to bbox JSON file (default: auto-detect)
        visualize: Debug mode - only process a few episodes, generate videos, then exit
        max_viz_episodes: Number of episodes to process in debug/visualize mode (default: 5)
        viz_output_dir: Output directory for visualization videos (default: output_dir/visualization)
    
    Usage:
        # Full dataset creation (no visualization)
        python create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --num_splits 10
        
        # Debug mode: process only 5 episodes, generate videos, then exit
        python create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --visualize
        
        # Debug mode with more episodes
        python create_dataset_droid.py main --tag aug_multiple_policy --split_id 0 --visualize --max_viz_episodes 10
        
        # Note: split_id is required. Full dataset loading is not supported.
    """
    # Debug/Visualize mode
    if visualize:
        print("=" * 80)
        print("DROID Dataset Creation - DEBUG/VISUALIZE MODE")
        print("=" * 80)
        print(f"Tag: {tag}")
        print(f"Split: {split_id}/{num_splits}")
        print(f"Output directory: {output_dir}")
        print(f"Debug mode: processing only {max_viz_episodes} episodes")
        print()
        
        # Set viz output dir
        if viz_output_dir is None:
            viz_output_dir = f"{output_dir}/{tag}/visualization_debug"
        
        # Run debug mode
        _run_debug_mode(
            tag=tag, split_id=split_id, num_splits=num_splits,
            output_dir=output_dir, bbox_file=bbox_file,
            max_episodes=max_viz_episodes, viz_output_dir=viz_output_dir
        )
        return
    
    # Full dataset creation mode
    print("=" * 80)
    print("DROID Dataset Creation - FULL MODE")
    print("=" * 80)
    print(f"Tag: {tag}")
    print(f"Split: {split_id}/{num_splits}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load gripper positions from splits
    gripper_position = load_gripper_positions(split_id=split_id, num_splits=num_splits)
    
    if len(gripper_position) == 0:
        print("ERROR: No gripper positions loaded!")
        print("Please ensure gripper position extraction has been completed.")
        return
    
    print()
    
    # Load bbox data
    bbox_data = load_bbox_data(bbox_file)
    print()
    
    # Calculate split percentages
    split_size = 100 // num_splits
    split_start = split_id * split_size
    split_end = (split_id + 1) * split_size if split_id < num_splits - 1 else 100
    
    print(f"RLDS split: train[{split_start}%:{split_end}%]")
    print()

    # Create dataset for specified split
    sample_list = create_dataset_droid(
        gripper_position=gripper_position,
        split_percent=(split_start, split_end),
        tag=tag,
        output_dir=output_dir,
        bbox_data=bbox_data,
    )
    
    if len(sample_list) == 0:
        print("WARNING: No samples generated!")
        return
    
    # Normalize movements
    save_file_path = f"{output_dir}/{tag}/droid_dataset_split_{split_id:02d}_of_{num_splits:02d}.json"
    sample_list = normalize_movement(tag, sample_list, overwrite=False, output_dir=output_dir)
    
    # Save dataset
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, "w") as f:
        json.dump(sample_list, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Dataset saved to: {save_file_path}")
    print(f"Total samples: {len(sample_list)}")
    print(f"{'='*80}\n")
    
    show_data(save_file_path)


def _run_debug_mode(tag, split_id, num_splits, output_dir, bbox_file, 
                    max_episodes, viz_output_dir):
    """
    Debug mode: process only a few episodes, generate annotations and videos, then exit.
    
    This is useful for verifying data correctness before full-scale generation.
    """
    # Load gripper positions
    gripper_position = load_gripper_positions(split_id=split_id, num_splits=num_splits)
    
    if len(gripper_position) == 0:
        print("ERROR: No gripper positions loaded!")
        return
    
    print()
    
    # Load bbox data
    bbox_data = load_bbox_data(bbox_file)
    print()
    
    # Calculate split percentages
    split_size = 100 // num_splits
    split_start = split_id * split_size
    split_end = (split_id + 1) * split_size if split_id < num_splits - 1 else 100
    
    print(f"RLDS split: train[{split_start}%:{split_end}%]")
    print()
    
    # Create dataset (limited to max_episodes)
    sample_list = create_dataset_droid_debug(
        gripper_position=gripper_position,
        split_percent=(split_start, split_end),
        tag=tag,
        output_dir=output_dir,
        bbox_data=bbox_data,
        max_episodes=max_episodes,
    )
    
    if len(sample_list) == 0:
        print("WARNING: No samples generated!")
        return
    
    # Save debug samples
    debug_save_path = f"{output_dir}/{tag}/debug_samples_{max_episodes}_episodes.json"
    os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
    with open(debug_save_path, "w") as f:
        json.dump(sample_list, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"Debug samples saved to: {debug_save_path}")
    print(f"Total samples: {len(sample_list)}")
    print(f"{'='*80}")
    
    # Show sample example
    if len(sample_list) > 0:
        print(f"\nSample example (index 0):")
        print(json.dumps(sample_list[0], indent=2))
    
    # Generate visualization videos
    print(f"\n{'='*80}")
    print(f"Generating visualization videos...")
    print(f"{'='*80}")
    
    visualize_dataset(
        input_file=debug_save_path,
        output_dir=viz_output_dir,
        max_episodes=max_episodes,
        droid_dir=DROID_DATA_PATH
    )
    
    print(f"\n{'='*80}")
    print(f"DEBUG MODE COMPLETE")
    print(f"{'='*80}")
    print(f"  Annotations: {debug_save_path}")
    print(f"  Videos: {viz_output_dir}")
    print(f"  Episodes processed: {max_episodes}")
    print(f"\nReview the videos, then run without --visualize for full dataset generation.")
    print(f"{'='*80}")


def show_data(filepath):
    """Show dataset statistics"""
    data = json.load(open(filepath))
    print(f"\n{'='*80}")
    print(f"Dataset Statistics")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print(f"Total samples: {len(data)}")
    
    if len(data) > 0:
        print(f"\nSample example (index 0):")
        print(json.dumps(data[0], indent=2))
        
        num_frame_per_segment = []
        for sample in data:
            num_frame_per_segment.append(len(sample["assistant_action_policy"]))
        print(f"\nAverage frames per segment: {np.mean(num_frame_per_segment):.2f}")
        print(f"Min frames: {np.min(num_frame_per_segment)}")
        print(f"Max frames: {np.max(num_frame_per_segment)}")
    print(f"{'='*80}\n")


def merge_splits(tag="aug_multiple_policy", num_splits=10, output_dir="dataset_droid"):
    """
    Merge multiple split files into one
    
    Usage:
        python create_dataset_droid.py merge_splits --tag aug_multiple_policy --num_splits 10
    """
    all_samples = []
    
    for i in range(num_splits):
        split_file = f"{output_dir}/{tag}/droid_dataset_split_{i:02d}_of_{num_splits:02d}.json"
        if os.path.exists(split_file):
            print(f"Loading {split_file}...")
            with open(split_file) as f:
                samples = json.load(f)
                all_samples.extend(samples)
        else:
            print(f"Warning: {split_file} not found, skipping...")
    
    if len(all_samples) == 0:
        print("No samples to merge!")
        return
    
    # Re-normalize after merging
    all_samples = normalize_movement(tag, all_samples, overwrite=False, output_dir=output_dir)
    
    output_file = f"{output_dir}/{tag}/droid_dataset_full.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=4)
    
    print(f"\nMerged dataset saved to: {output_file}")
    print(f"Total samples: {len(all_samples)}")
    show_data(output_file)


# ============================================================================
# Visualization Functions
# ============================================================================

def parse_sample_path(image_path):
    """
    解析 sample 的 image_path，提取 file_path, episode_id, frame_index
    
    Args:
        image_path: "gs://xembodiment_data/.../trajectory.h5|episode_id/frame_X.jpg"
    
    Returns:
        (file_path, episode_id, frame_index)
    """
    parts = image_path.rsplit('/', 1)
    frame_str = parts[1]
    frame_index = int(re.search(r'frame_(\d+)\.jpg', frame_str).group(1))
    
    path_with_episode = parts[0]
    file_path, episode_id = path_with_episode.rsplit('|', 1)
    
    return file_path, episode_id, frame_index


def parse_gripper_from_text(text):
    """从文本中解析 gripper 坐标"""
    match = re.search(r'\[(\d+),\s*(\d+)\]', text)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    return None


def parse_objects_from_text(text):
    """
    从 OBJECT 字段文本中解析物体和 bbox
    
    Returns:
        list of (name, [x1, y1, x2, y2])
    """
    objects = []
    if not text:
        return objects
    
    # 格式: object_name: [x1,y1], [x2,y2]
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue
        
        # 解析 name: [x1,y1], [x2,y2]
        name_part, coords_part = line.split(':', 1)
        name = name_part.strip()
        
        # 解析两个坐标
        coords_match = re.findall(r'\[(\d+),(\d+)\]', coords_part)
        if len(coords_match) >= 2:
            x1, y1 = int(coords_match[0][0]), int(coords_match[0][1])
            x2, y2 = int(coords_match[1][0]), int(coords_match[1][1])
            objects.append((name, [x1, y1, x2, y2]))
    
    return objects


def resize_pos(pos, img_size, orig_size=(256, 256)):
    """调整坐标到目标图像尺寸"""
    if pos is None or len(pos) != 2:
        return [0, 0]
    return [
        int(pos[0] * img_size[0] / orig_size[0]),
        int(pos[1] * img_size[1] / orig_size[1])
    ]


def name_to_random_color(name):
    """根据名称生成一致的随机颜色"""
    return tuple([(hash(name) // (256**i)) % 256 for i in range(3)])


def draw_gripper_viz(img, pos, img_size, color=(0, 255, 255), label=None):
    """绘制 gripper 位置"""
    _import_viz_deps()
    pos_resized = resize_pos(pos, img_size, orig_size=(256, 256))
    cv2.circle(img, tuple(pos_resized), 10, (0, 0, 0), -1)
    cv2.circle(img, tuple(pos_resized), 8, color, -1)
    cv2.circle(img, tuple(pos_resized), 3, (255, 255, 255), -1)
    
    if label:
        cv2.putText(img, label, (pos_resized[0] + 12, pos_resized[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_bboxes_viz(img, objects, img_size):
    """
    绘制 bounding boxes
    
    Args:
        objects: List of (name, [x1, y1, x2, y2])
        img_size: (width, height)
    """
    _import_viz_deps()
    for name, bbox in objects:
        x1, y1, x2, y2 = bbox
        pt1 = resize_pos([x1, y1], img_size, orig_size=(256, 256))
        pt2 = resize_pos([x2, y2], img_size, orig_size=(256, 256))
        
        color = name_to_random_color(name)
        cv2.rectangle(img, tuple(pt1), tuple(pt2), color, 2)
        
        label = name
        cv2.putText(img, label, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def parse_long_plan_trajectories(long_plan_text):
    """
    从 LONG PLAN 文本中解析每个子任务的轨迹点
    
    Args:
        long_plan_text: LONG PLAN 字段内容
        
    实际格式示例:
        Task Plans: 1. Locating the blue ring, move from [19, 205], [26, 199], [74, 154]; 2. Picking up...
    
    Returns:
        list of dict: [{"index": 1, "subtask": "...", "trajectory": [[x1,y1], [x2,y2], [x3,y3]]}, ...]
    """
    trajectories = []
    
    # 提取 Task Plans 部分
    if "Task Plans:" in long_plan_text:
        task_plans_part = long_plan_text.split("Task Plans:")[-1].strip()
        
        # 解析每个子任务
        # 格式: 1. subtask content, move from [x1, y1], [x2, y2], [x3, y3]; 2. ...
        # 使用非贪婪匹配来处理 subtask 名称
        pattern = r'(\d+)\.\s*(.+?),\s*move from\s*\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, task_plans_part)
        
        for match in matches:
            idx, subtask, x1, y1, x2, y2, x3, y3 = match
            trajectories.append({
                "index": int(idx),
                "subtask": subtask.strip(),
                "trajectory": [
                    [int(x1), int(y1)],  # 起点
                    [int(x2), int(y2)],  # 中点
                    [int(x3), int(y3)]   # 终点
                ]
            })
    
    return trajectories


def parse_short_plan_trajectory(short_plan_text):
    """
    从 SHORT PLAN 文本中解析当前子任务的轨迹点（起点、中点、终点）
    
    Args:
        short_plan_text: SHORT PLAN 字段内容
        
    实际格式示例:
        Subtask: Locating the blue ring, move from [19, 205], [26, 199], [74, 154]
    
    Returns:
        dict: {"subtask": "...", "trajectory": [[x1,y1], [x2,y2], [x3,y3]]} or None
    """
    # 解析 Subtask 行
    # 格式: Subtask: subtask content, move from [x1, y1], [x2, y2], [x3, y3]
    pattern = r'Subtask:\s*(.+?),\s*move from\s*\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]'
    match = re.search(pattern, short_plan_text)
    
    if match:
        subtask = match.group(1).strip()
        trajectory = [
            [int(match.group(2)), int(match.group(3))],  # 起点
            [int(match.group(4)), int(match.group(5))],  # 中点
            [int(match.group(6)), int(match.group(7))]   # 终点
        ]
        return {
            "subtask": subtask,
            "trajectory": trajectory
        }
    
    return None


def parse_initial_positions(long_plan_text):
    """
    从 LONG PLAN 文本中解析初始位置信息
    
    Args:
        long_plan_text: LONG PLAN 字段内容
        
    实际格式示例:
        Initial Positions: Gripper [19, 205], table [86, 79, 255, 253], wooden tray [87, 94, 157, 172], blue ring [149, 135, 169, 159]
    
    Returns:
        dict: {"gripper": [x, y], "objects": [(name, [x1, y1, x2, y2]), ...]}
    """
    result = {"gripper": None, "objects": []}
    
    if "Initial Positions:" not in long_plan_text:
        return result
    
    # 提取 Initial Positions 部分（到 Task Plans 或换行为止）
    initial_part = long_plan_text.split("Initial Positions:")[-1]
    if "Task Plans:" in initial_part:
        initial_part = initial_part.split("Task Plans:")[0]
    if "\n" in initial_part:
        initial_part = initial_part.split("\n")[0]
    initial_part = initial_part.strip()
    
    # 解析 Gripper 位置 (格式: Gripper [x, y])
    gripper_match = re.search(r'Gripper\s*\[(\d+),\s*(\d+)\]', initial_part)
    if gripper_match:
        result["gripper"] = [int(gripper_match.group(1)), int(gripper_match.group(2))]
    
    # 解析物体 bbox (格式: object name [x1, y1, x2, y2])
    # 使用正则表达式直接匹配所有 "name [x1, y1, x2, y2]" 模式
    # 物体名不以数字开头，且后面紧跟 4 个数字的 bbox
    bbox_pattern = r'([a-zA-Z][a-zA-Z0-9\s]*?)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    bbox_matches = re.findall(bbox_pattern, initial_part)
    
    for match in bbox_matches:
        name = match[0].strip()
        if name.lower() != "gripper":
            bbox = [int(match[1]), int(match[2]), int(match[3]), int(match[4])]
            result["objects"].append((name, bbox))
    
    return result


def extract_subtask_from_sample(sample):
    """从样本中提取当前子任务名称"""
    # 从 SHORT PLAN 中提取 Subtask (格式: Subtask: subtask content, move from ...)
    short_plan = sample.get('assistant_short_plan', '')
    match = re.search(r'Subtask:\s*(.+?),\s*move from', short_plan)
    if match:
        return match.group(1).strip()
    return None


def segment_gripper_trajectory_by_subtask(frame_samples):
    """
    根据子任务分段 gripper 轨迹
    
    Args:
        frame_samples: [(frame_index, sample), ...]
    
    Returns:
        list of dict: [{"subtask": "...", "gripper_positions": [[x,y], ...], "frame_range": (start, end)}, ...]
    """
    if not frame_samples:
        return []
    
    segments = []
    current_subtask = None
    current_positions = []
    current_start_frame = None
    
    for frame_idx, sample in frame_samples:
        subtask = extract_subtask_from_sample(sample)
        gripper = parse_gripper_from_text(sample.get('user', ''))
        
        if subtask != current_subtask:
            # 保存上一个子任务的轨迹
            if current_subtask is not None and len(current_positions) > 0:
                segments.append({
                    "subtask": current_subtask,
                    "gripper_positions": current_positions,
                    "frame_range": (current_start_frame, frame_idx - 1)
                })
            # 开始新的子任务
            current_subtask = subtask
            current_positions = []
            current_start_frame = frame_idx
        
        if gripper:
            current_positions.append(gripper)
    
    # 保存最后一个子任务
    if current_subtask is not None and len(current_positions) > 0:
        last_frame_idx = frame_samples[-1][0] if frame_samples else 0
        segments.append({
            "subtask": current_subtask,
            "gripper_positions": current_positions,
            "frame_range": (current_start_frame, last_frame_idx)
        })
    
    return segments


def generate_long_plan_image(frame, sample, img_size, output_path, frame_samples=None):
    """
    生成 LONG PLAN 可视化图像
    
    在初始帧上绘制:
    - 所有物体的 bbox
    - gripper 初始位置
    - 每个子任务的完整轨迹（用实际 gripper 位置连接）
    
    Args:
        frame: 初始帧图像 (numpy array)
        sample: 包含 LONG PLAN 的样本
        img_size: (width, height)
        output_path: 输出图像路径
        frame_samples: [(frame_index, sample), ...] 所有帧的样本（可选，用于绘制完整轨迹）
    """
    _import_viz_deps()
    
    # 复制图像
    img = frame.copy()
    
    # 获取 LONG PLAN 文本
    long_plan_text = sample.get('assistant_plan_level', '')
    long_plan_text = long_plan_text.replace("LONG PLAN:\n", "").replace("PLAN:\n", "")
    
    # 解析初始位置
    initial_pos = parse_initial_positions(long_plan_text)
    
    # 绘制所有物体 bbox
    for name, bbox in initial_pos["objects"]:
        x1, y1, x2, y2 = bbox
        pt1 = resize_pos([x1, y1], img_size, orig_size=(256, 256))
        pt2 = resize_pos([x2, y2], img_size, orig_size=(256, 256))
        
        color = name_to_random_color(name)
        cv2.rectangle(img, tuple(pt1), tuple(pt2), color, 2)
        
        # 添加标签（带背景）
        label = name
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 5), (pt1[0] + text_w, pt1[1]), color, -1)
        cv2.putText(img, label, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 绘制 gripper 初始位置
    if initial_pos["gripper"]:
        gripper_pos = resize_pos(initial_pos["gripper"], img_size, orig_size=(256, 256))
        cv2.circle(img, tuple(gripper_pos), 12, (0, 0, 0), -1)
        cv2.circle(img, tuple(gripper_pos), 10, (0, 255, 255), -1)
        cv2.circle(img, tuple(gripper_pos), 4, (255, 0, 0), -1)
        cv2.putText(img, "Gripper", (gripper_pos[0] + 15, gripper_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    
    # 定义轨迹颜色（每个子任务不同颜色）
    trajectory_colors = [
        (255, 0, 0),    # 红
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 蓝
        (255, 255, 0),  # 青
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄
        (128, 0, 255),  # 紫
        (255, 128, 0),  # 橙
    ]
    
    subtask_list = []
    
    # 如果有完整的帧数据，使用实际轨迹
    if frame_samples and len(frame_samples) > 1:
        # 根据子任务分段轨迹
        segments = segment_gripper_trajectory_by_subtask(frame_samples)
        
        # 绘制每个子任务的完整轨迹
        for i, seg in enumerate(segments):
            color = trajectory_colors[i % len(trajectory_colors)]
            positions = seg["gripper_positions"]
            subtask = seg["subtask"]
            subtask_list.append(subtask)
            
            if len(positions) < 2:
                continue
            
            # 转换所有坐标
            points = [resize_pos(pt, img_size, orig_size=(256, 256)) for pt in positions]
            
            # 绘制完整轨迹线
            for j in range(len(points) - 1):
                pt1 = tuple(points[j])
                pt2 = tuple(points[j + 1])
                cv2.line(img, pt1, pt2, color, 2)
            
            # 在最后一段绘制箭头表示方向
            if len(points) >= 2:
                cv2.arrowedLine(img, tuple(points[-2]), tuple(points[-1]), color, 2, tipLength=0.3)
            
            # 标记起点、中点、终点
            key_indices = [0, len(points) // 2, len(points) - 1]
            point_labels = ["S", "M", "E"]
            
            for j, (idx, label) in enumerate(zip(key_indices, point_labels)):
                if idx < len(points):
                    pt = points[idx]
                    # 绘制点
                    cv2.circle(img, tuple(pt), 8, (0, 0, 0), -1)
                    cv2.circle(img, tuple(pt), 6, color, -1)
                    # 标签
                    cv2.putText(img, f"{i+1}{label}", (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    else:
        # 如果没有完整帧数据，使用 LONG PLAN 中的三个点（回退方案）
        trajectories = parse_long_plan_trajectories(long_plan_text)
        
        for i, traj_info in enumerate(trajectories):
            color = trajectory_colors[i % len(trajectory_colors)]
            trajectory = traj_info["trajectory"]
            subtask = traj_info["subtask"]
            idx = traj_info["index"]
            subtask_list.append(subtask)
            
            # 转换坐标
            points = [resize_pos(pt, img_size, orig_size=(256, 256)) for pt in trajectory]
            
            # 绘制轨迹线（起点->中点->终点）
            for j in range(len(points) - 1):
                pt1 = tuple(points[j])
                pt2 = tuple(points[j + 1])
                cv2.line(img, pt1, pt2, color, 3)
                cv2.arrowedLine(img, pt1, pt2, color, 3, tipLength=0.15)
            
            # 绘制轨迹点
            point_labels = ["S", "M", "E"]
            for j, (pt, label) in enumerate(zip(points, point_labels)):
                cv2.circle(img, tuple(pt), 8, (0, 0, 0), -1)
                cv2.circle(img, tuple(pt), 6, color, -1)
                cv2.putText(img, f"{idx}{label}", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    
    # 创建图例面板
    legend_height = 150
    legend = np.ones((legend_height, img_size[0], 3), dtype=np.uint8) * 255
    
    pil_legend = Image.fromarray(legend)
    draw = ImageDraw.Draw(pil_legend)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_bold = font
    
    # 标题
    draw.text((10, 5), "LONG PLAN Visualization", fill=(0, 0, 0), font=font_bold)
    
    y_offset = 25
    # 图例说明
    draw.text((10, y_offset), "Trajectory: S=Start, M=Middle, E=End", fill=(100, 100, 100), font=font)
    y_offset += 18
    
    # 显示每个子任务
    for i, subtask in enumerate(subtask_list):
        color = trajectory_colors[i % len(trajectory_colors)]
        # PIL 使用 RGB，cv2 使用 BGR
        color_rgb = (color[2], color[1], color[0])
        subtask_text = f"{i+1}. {subtask[:50]}..." if len(subtask) > 50 else f"{i+1}. {subtask}"
        draw.text((10, y_offset), subtask_text, fill=color_rgb, font=font)
        y_offset += 16
        if y_offset > legend_height - 20:
            break
    
    legend = np.array(pil_legend)
    
    # 合并图像和图例
    combined = np.vstack([img, legend])
    
    # 保存图像
    cv2.imwrite(output_path, combined)
    print(f"  LONG PLAN image saved: {output_path}")
    
    return True


def create_text_panel_viz(sample, frame_index, total_frames, img_width=640):
    """创建文本面板显示 CoT 信息"""
    _import_viz_deps()
    panel_height = 500
    panel = np.ones((panel_height, img_width, 3), dtype=np.uint8) * 255
    
    pil_img = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_bold = font
    
    y_offset = 10
    line_height = 14
    max_width = 75
    
    # Frame info
    draw.text((10, y_offset), f"Frame {frame_index}/{total_frames}", fill=(0, 0, 255), font=font_bold)
    y_offset += line_height + 5
    
    # Current Gripper
    current_gripper = parse_gripper_from_text(sample['user'])
    if current_gripper:
        draw.text((10, y_offset), f"CURRENT GRIPPER: {current_gripper}", fill=(0, 200, 200), font=font)
        y_offset += line_height + 3
    
    # Next Gripper
    next_gripper = parse_gripper_from_text(sample['assistant_position_level'])
    if next_gripper:
        draw.text((10, y_offset), f"NEXT GRIPPER: {next_gripper}", fill=(255, 0, 255), font=font)
        y_offset += line_height + 8
    
    # LONG PLAN
    draw.text((10, y_offset), "LONG PLAN:", fill=(200, 0, 0), font=font_bold)
    y_offset += line_height
    
    plan_text = sample['assistant_plan_level'].replace("LONG PLAN:\n", "").replace("PLAN:\n", "").strip()
    for line in plan_text.split('\n')[:8]:
        # Truncate long lines
        line_text = line.strip()[:max_width]
        draw.text((15, y_offset), line_text, fill=(100, 0, 0), font=font_small)
        y_offset += line_height
    y_offset += 5
    
    # SHORT PLAN (contains Current Positions, Subtask, Subtask Reasoning, Movement)
    short_plan = sample.get('assistant_short_plan', '')
    
    # Current Positions
    current_pos_match = re.search(r'Current Positions:\s*(.+?)(?:\n|$)', short_plan)
    if current_pos_match:
        draw.text((10, y_offset), "CURRENT POSITIONS:", fill=(0, 128, 128), font=font_bold)
        y_offset += line_height
        wrapped = textwrap.wrap(current_pos_match.group(1).strip(), width=max_width)
        for line in wrapped[:2]:
            draw.text((15, y_offset), line, fill=(0, 100, 100), font=font_small)
            y_offset += line_height
        y_offset += 3
    
    # Subtask with trajectory
    subtask_match = re.search(r'Subtask:\s*(.+?)(?:\nSubtask Reasoning:|$)', short_plan, re.DOTALL)
    if subtask_match:
        draw.text((10, y_offset), "SUBTASK:", fill=(0, 0, 180), font=font_bold)
        y_offset += line_height
        wrapped = textwrap.wrap(subtask_match.group(1).strip(), width=max_width)
        for line in wrapped[:2]:
            draw.text((15, y_offset), line, fill=(0, 0, 140), font=font_small)
            y_offset += line_height
        y_offset += 3
    
    # Subtask Reasoning
    reasoning_match = re.search(r'Subtask Reasoning:\s*(.+?)(?:\nCurrent Movement:|$)', short_plan, re.DOTALL)
    if reasoning_match:
        draw.text((10, y_offset), "SUBTASK REASONING:", fill=(0, 128, 0), font=font_bold)
        y_offset += line_height
        wrapped = textwrap.wrap(reasoning_match.group(1).strip(), width=max_width)
        for line in wrapped[:2]:
            draw.text((15, y_offset), line, fill=(0, 100, 0), font=font_small)
            y_offset += line_height
        y_offset += 3
    
    # Current Movement (当前 step 到下一个 step)
    current_movement_match = re.search(r'Current Movement:\s*(.+?)(?:\nSubtask Movement:|$)', short_plan, re.DOTALL)
    if current_movement_match:
        draw.text((10, y_offset), "CURRENT MOVEMENT:", fill=(128, 0, 128), font=font_bold)
        y_offset += line_height
        wrapped = textwrap.wrap(current_movement_match.group(1).strip(), width=max_width)
        for line in wrapped[:2]:
            draw.text((15, y_offset), line, fill=(100, 0, 100), font=font_small)
            y_offset += line_height
        y_offset += 3
    
    # Subtask Movement (当前 step 到子阶段终点)
    subtask_movement_match = re.search(r'Subtask Movement:\s*(.+?)(?:\n|$)', short_plan, re.DOTALL)
    if subtask_movement_match:
        draw.text((10, y_offset), "SUBTASK MOVEMENT:", fill=(180, 100, 0), font=font_bold)
        y_offset += line_height
        wrapped = textwrap.wrap(subtask_movement_match.group(1).strip(), width=max_width)
        for line in wrapped[:2]:
            draw.text((15, y_offset), line, fill=(140, 80, 0), font=font_small)
            y_offset += line_height
    y_offset += 5
    
    # OBJECT
    if sample.get('assistant_object_level'):
        draw.text((10, y_offset), "OBJECT:", fill=(180, 100, 0), font=font_bold)
        y_offset += line_height
        
        object_text = sample['assistant_object_level'].replace("OBJECT:\n", "").strip()
        for line in object_text.split('\n')[:5]:
            draw.text((15, y_offset), line.strip()[:max_width], fill=(140, 80, 0), font=font_small)
            y_offset += line_height
    
    return np.array(pil_img)


def load_frames_from_rlds_viz(droid_dir, file_path, frame_indices):
    """
    从 RLDS 数据集加载指定帧
    
    Args:
        droid_dir: DROID 数据集目录
        file_path: episode 的 gs:// 文件路径
        frame_indices: 需要加载的帧索引列表
    
    Returns:
        dict: {frame_index: frame_image}
    """
    frames = {}
    max_frame = max(frame_indices) if frame_indices else 0
    
    try:
        dataset = tfds.load(
            "droid",
            data_dir=droid_dir,
            split="train"
        )
        
        for episode in dataset:
            ep_file_path = episode["episode_metadata"]["file_path"].numpy().decode()
            
            if ep_file_path == file_path:
                for frame_idx, step in enumerate(episode["steps"]):
                    if frame_idx in frame_indices:
                        obs = step["observation"]
                        if "exterior_image_1_left" in obs:
                            frame = obs["exterior_image_1_left"].numpy()
                            frames[frame_idx] = frame
                    
                    if frame_idx >= max_frame:
                        break
                break
    
    except Exception as e:
        print(f"Error loading frames: {e}")
        import traceback
        traceback.print_exc()
    
    return frames


def group_samples_by_episode(samples):
    """
    将样本按 episode 分组
    
    Returns:
        dict: {(file_path, episode_id): [samples]}
    """
    episodes = defaultdict(list)
    
    for sample in samples:
        file_path, episode_id, frame_index = parse_sample_path(sample['current_image_path'])
        key = (file_path, episode_id)
        episodes[key].append((frame_index, sample))
    
    for key in episodes:
        episodes[key].sort(key=lambda x: x[0])
    
    return episodes


def visualize_episode(file_path, episode_id, frame_samples, output_video_path, droid_dir):
    """
    可视化一个 episode
    
    Args:
        file_path: gs:// 文件路径
        episode_id: episode ID
        frame_samples: [(frame_index, sample), ...]
        output_video_path: 输出视频路径
        droid_dir: DROID 数据集目录
    
    生成:
    - 视频文件: output_video_path
    - LONG PLAN 可视化图像: output_video_path.replace('.mp4', '_long_plan.png')
    """
    _import_viz_deps()
    frame_indices = [fs[0] for fs in frame_samples]
    total_frames = max(frame_indices) + 1
    
    # 确保加载第 0 帧用于 LONG PLAN 可视化
    if 0 not in frame_indices:
        frame_indices_with_0 = [0] + frame_indices
    else:
        frame_indices_with_0 = frame_indices
    
    print(f"  Loading {len(frame_indices_with_0)} frames from RLDS...")
    frames = load_frames_from_rlds_viz(droid_dir, file_path, frame_indices_with_0)
    
    if len(frames) == 0:
        print("  No frames loaded!")
        return False
    
    print(f"  Loaded {len(frames)} frames")
    
    sample_frame = list(frames.values())[0]
    original_h, original_w = sample_frame.shape[:2]
    img_width = 640
    img_height = int(img_width * original_h / original_w)
    
    # 生成 LONG PLAN 可视化图像（使用第 0 帧，绘制完整轨迹）
    if 0 in frames and len(frame_samples) > 0:
        first_sample = frame_samples[0][1]
        frame_0 = frames[0]
        frame_0_resized = cv2.resize(frame_0, (img_width, img_height))
        frame_0_bgr = cv2.cvtColor(frame_0_resized, cv2.COLOR_RGB2BGR)
        
        long_plan_image_path = output_video_path.replace('.mp4', '_long_plan.png')
        generate_long_plan_image(
            frame_0_bgr, first_sample, (img_width, img_height), 
            long_plan_image_path, frame_samples=frame_samples
        )
    
    fps = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height + 500))
    
    for frame_idx, sample in tqdm(frame_samples, desc="  Rendering", leave=False):
        if frame_idx not in frames:
            continue
        
        frame = frames[frame_idx]
        
        frame_resized = cv2.resize(frame, (img_width, img_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        
        # 绘制当前 gripper 位置
        current_gripper = parse_gripper_from_text(sample['user'])
        if current_gripper:
            draw_gripper_viz(frame_rgb, current_gripper, (img_width, img_height), 
                        color=(0, 255, 255), label="curr")
        
        # 绘制目标 gripper 位置
        next_gripper = parse_gripper_from_text(sample['assistant_position_level'])
        if next_gripper:
            draw_gripper_viz(frame_rgb, next_gripper, (img_width, img_height), 
                        color=(255, 0, 255), label="next")
        
        # 绘制 gripper 连接线
        if current_gripper and next_gripper:
            curr_pos = resize_pos(current_gripper, (img_width, img_height))
            next_pos = resize_pos(next_gripper, (img_width, img_height))
            cv2.arrowedLine(frame_rgb, tuple(curr_pos), tuple(next_pos),
                           (0, 255, 0), 2, tipLength=0.3)
        
        # 绘制 bounding boxes
        if sample.get('assistant_object_level'):
            objects = parse_objects_from_text(sample['assistant_object_level'])
            if objects:
                draw_bboxes_viz(frame_rgb, objects, (img_width, img_height))
        
        # 绘制 SHORT PLAN 的子任务轨迹（起点、中点、终点）
        if sample.get('assistant_short_plan'):
            short_traj = parse_short_plan_trajectory(sample['assistant_short_plan'])
            if short_traj and short_traj.get('trajectory'):
                trajectory = short_traj['trajectory']
                # 转换坐标到显示分辨率
                points = [resize_pos(pt, (img_width, img_height), orig_size=(256, 256)) for pt in trajectory]
                
                # 绘制轨迹线（青色虚线风格）
                traj_color = (255, 200, 0)  # 橙黄色
                for j in range(len(points) - 1):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[j + 1])
                    cv2.line(frame_rgb, pt1, pt2, traj_color, 2)
                
                # 在最后一段绘制箭头
                if len(points) >= 2:
                    cv2.arrowedLine(frame_rgb, tuple(points[-2]), tuple(points[-1]), 
                                   traj_color, 2, tipLength=0.2)
                
                # 绘制三个关键点：S(起点), M(中点), E(终点)
                point_labels = ["S", "M", "E"]
                for j, (pt, label) in enumerate(zip(points, point_labels)):
                    # 绘制点
                    cv2.circle(frame_rgb, tuple(pt), 6, (0, 0, 0), -1)
                    cv2.circle(frame_rgb, tuple(pt), 4, traj_color, -1)
                    # 标签
                    cv2.putText(frame_rgb, label, (pt[0] + 6, pt[1] - 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, traj_color, 1, cv2.LINE_AA)
        
        # 创建文本面板
        text_panel = create_text_panel_viz(sample, frame_idx, total_frames, img_width)
        
        # 合并图像和文本
        combined = np.vstack([frame_rgb, text_panel])
        
        out.write(combined)
    
    out.release()
    print(f"  Video saved: {output_video_path}")
    return True


def visualize_dataset(input_file, output_dir="outputs/visualization_dataset", 
                      max_episodes=10, droid_dir=None):
    """
    可视化数据集样本
    
    Args:
        input_file: 输入 JSON 文件路径
        output_dir: 输出视频目录
        max_episodes: 最大可视化 episode 数量
        droid_dir: DROID 数据集目录
    
    Usage:
        python create_dataset_droid.py visualize --input sample_100.json --max-episodes 10
    """
    if droid_dir is None:
        droid_dir = DROID_DATA_PATH
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("DROID Dataset Visualization")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Max episodes: {max_episodes}")
    print()
    
    print("Loading samples...")
    with open(input_file, 'r') as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")
    
    print("Grouping by episode...")
    episodes = group_samples_by_episode(samples)
    print(f"Found {len(episodes)} episodes")
    
    episode_keys = list(episodes.keys())[:max_episodes]
    print(f"Processing {len(episode_keys)} episodes...")
    print("=" * 80)
    
    success_count = 0
    failed_count = 0
    
    for idx, (file_path, episode_id) in enumerate(episode_keys):
        frame_samples = episodes[(file_path, episode_id)]
        
        print(f"\n[{idx+1}/{len(episode_keys)}] Episode: {episode_id}")
        print(f"  File: ...{file_path[-60:]}")
        print(f"  Frames: {len(frame_samples)}")
        
        output_video = Path(output_dir) / f"episode_{idx:03d}_{episode_id[-8:]}.mp4"
        
        success = visualize_episode(
            file_path=file_path,
            episode_id=episode_id,
            frame_samples=frame_samples,
            output_video_path=str(output_video),
            droid_dir=droid_dir
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*80}")
    print(f"Visualization complete!")
    print(f"  Success: {success_count}/{len(episode_keys)}")
    print(f"  Failed:  {failed_count}/{len(episode_keys)}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(description="Create DROID dataset with embodied CoT")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Main command: create dataset
    main_parser = subparsers.add_parser("main", help="Create dataset for a specific split")
    main_parser.add_argument("--tag", type=str, default="aug_multiple_policy",
                            choices=["single_policy", "multiple_policy", "aug_multiple_policy"],
                            help="Policy type (default: aug_multiple_policy)")
    main_parser.add_argument("--split_id", type=int, required=True,
                            help="Split ID to process (0-based, REQUIRED)")
    main_parser.add_argument("--num_splits", type=int, default=10,
                            help="Total number of splits (default: 10)")
    main_parser.add_argument("--output_dir", type=str, default="dataset_droid",
                            help="Output directory for dataset files (default: dataset_droid)")
    main_parser.add_argument("--bbox_file", type=str, default=None,
                            help="Path to bbox JSON file (default: auto-detect)")
    main_parser.add_argument("--visualize", action="store_true",
                            help="Visualize samples after creation")
    main_parser.add_argument("--max_viz_episodes", type=int, default=5,
                            help="Maximum number of episodes to visualize (default: 5)")
    main_parser.add_argument("--viz_output_dir", type=str, default=None,
                            help="Output directory for visualization videos (default: output_dir/tag/visualization)")
    
    # Show data command
    show_parser = subparsers.add_parser("show_data", help="Show dataset statistics")
    show_parser.add_argument("--filepath", type=str, required=True,
                            help="Path to dataset JSON file")
    
    # Merge splits command
    merge_parser = subparsers.add_parser("merge_splits", help="Merge all split files")
    merge_parser.add_argument("--tag", type=str, default="aug_multiple_policy",
                             choices=["single_policy", "multiple_policy", "aug_multiple_policy"],
                             help="Policy type (default: aug_multiple_policy)")
    merge_parser.add_argument("--num_splits", type=int, default=10,
                             help="Total number of splits (default: 10)")
    merge_parser.add_argument("--output_dir", type=str, default="dataset_droid",
                             help="Output directory for dataset files (default: dataset_droid)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize dataset samples as videos")
    viz_parser.add_argument("--input", type=str, required=True,
                           help="Input JSON file with samples")
    viz_parser.add_argument("--output_dir", type=str, default="outputs/visualization_dataset",
                           help="Output directory for videos (default: outputs/visualization_dataset)")
    viz_parser.add_argument("--max_episodes", type=int, default=10,
                           help="Maximum number of episodes to visualize (default: 10)")
    viz_parser.add_argument("--droid_dir", type=str, default=None,
                           help="DROID dataset directory (default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.command == "main":
        main(tag=args.tag, split_id=args.split_id, num_splits=args.num_splits,
             output_dir=args.output_dir, bbox_file=args.bbox_file,
             visualize=args.visualize, max_viz_episodes=args.max_viz_episodes,
             viz_output_dir=args.viz_output_dir)
    elif args.command == "show_data":
        show_data(filepath=args.filepath)
    elif args.command == "merge_splits":
        merge_splits(tag=args.tag, num_splits=args.num_splits, output_dir=args.output_dir)
    elif args.command == "visualize":
        visualize_dataset(input_file=args.input, output_dir=args.output_dir,
                         max_episodes=args.max_episodes, droid_dir=args.droid_dir)
    else:
        parser.print_help()
