#!/bin/bash

export PBS_JOBFS=/jobfs/164757456.gadi-pbs
export PBS_JOBID=164757456.gadi-pbs

# 检查脚本是否被 source 执行
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "Error: This script must be sourced, not executed directly."
    echo "Usage: source scripts/env.sh"
    echo "   or: . scripts/env.sh"
    exit 1
fi

# 获取当前脚本所在的项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME=$(basename "$PROJECT_ROOT")

# export PBS_JOBFS="/jobfs/164712726.gadi-pbs"
# 检查 PBS_JOBFS 环境变量是否存在和非空 
if [ -z "$PBS_JOBFS" ]; then
    echo "Warning: PBS_JOBFS is not set or empty."
    PBS_JOBFS="/tmp/$USER/$PROJECT_NAME"
else
    if [[ "$PBS_JOBFS" != *"/$PROJECT_NAME" ]]; then
        PBS_JOBFS="$PBS_JOBFS/$PROJECT_NAME"
    fi
fi

mkdir -p "$PBS_JOBFS"

echo "Project name: $PROJECT_NAME"
echo "PBS_JOBFS: $PBS_JOBFS"

# ── 解压 conda-pack 打包的 robocasa 环境 ──────────────────────────────────────
CONDA_ENV_DIR="$PBS_JOBFS/robocasa_env"

if [ -d "$CONDA_ENV_DIR" ]; then
    echo "Conda environment already exists in $CONDA_ENV_DIR, skipping extraction."
else
    echo "Extracting conda environment to JOBFS..."
    if [ -f "$PROJECT_ROOT/robocasa_env.tar" ]; then
        mkdir -p "$CONDA_ENV_DIR"
        tar -xf "$PROJECT_ROOT/robocasa_env.tar" -C "$CONDA_ENV_DIR"
        echo "Conda environment extraction completed."

        # 激活环境并运行 conda-unpack 修复硬编码路径
        echo "Fixing conda environment paths (conda-unpack)..."
        source "$CONDA_ENV_DIR/bin/activate"
        conda-unpack
        echo "conda-unpack completed."
    else
        echo "Error: robocasa_env.tar not found at $PROJECT_ROOT/robocasa_env.tar"
        return 1
    fi
fi

# ── 激活 conda-pack 环境 ───────────────────────────────────────────────────────
source "$CONDA_ENV_DIR/bin/activate"

# ── 解压并挂载 assets 到 JOBFS（避免共享盘 inode 爆炸） ─────────────────────────
ASSET_ARCHIVE="${ROBOCASA_ASSET_ARCHIVE:-$PROJECT_ROOT/robocasa/models/assets.tar}"
ASSET_JOBFS_DIR="$PBS_JOBFS/robocasa_assets"
ASSET_EXTRACTED_DIR="$ASSET_JOBFS_DIR/assets"

mkdir -p "$ASSET_JOBFS_DIR"

if [ -d "$ASSET_EXTRACTED_DIR" ]; then
    echo "Assets already extracted in $ASSET_EXTRACTED_DIR, skipping extraction."
else
    echo "Extracting assets archive to JOBFS..."
    if [ -f "$ASSET_ARCHIVE" ]; then
        tar -xf "$ASSET_ARCHIVE" -C "$ASSET_JOBFS_DIR"
        echo "Assets extraction completed."
    else
        echo "Error: assets archive not found at $ASSET_ARCHIVE"
        echo "Tip: create it with scripts/pack_assets.sh first."
        return 1
    fi
fi
echo "Using assets from JOBFS: $ASSET_EXTRACTED_DIR"

# ── 解压数据集（目录结构与源路径保持一致：datasets/v1.0/） ───────────────────
DATASET_V1_DIR="$PBS_JOBFS/datasets/v1.0"
mkdir -p "$DATASET_V1_DIR"

if [ -d "$DATASET_V1_DIR/target" ]; then
    echo "Dataset 'target' already exists in $DATASET_V1_DIR, skipping extraction."
else
    echo "Extracting human_target dataset to JOBFS..."
    if [ -f "$PROJECT_ROOT/datasets/v1.0/human_target.tar" ]; then
        tar -xf "$PROJECT_ROOT/datasets/v1.0/human_target.tar" -C "$DATASET_V1_DIR"
        echo "human_target extraction completed."
    else
        echo "Error: human_target.tar not found at $PROJECT_ROOT/datasets/v1.0/human_target.tar"
        return 1
    fi
fi

if [ -d "$DATASET_V1_DIR/pretrain" ]; then
    echo "Dataset 'pretrain' already exists in $DATASET_V1_DIR, skipping extraction."
else
    echo "Extracting human_pretrain dataset to JOBFS..."
    if [ -f "$PROJECT_ROOT/datasets/v1.0/human_pretrain.tar" ]; then
        tar -xf "$PROJECT_ROOT/datasets/v1.0/human_pretrain.tar" -C "$DATASET_V1_DIR"
        echo "human_pretrain extraction completed."
    else
        echo "Error: human_pretrain.tar not found at $PROJECT_ROOT/datasets/v1.0/human_pretrain.tar"
        return 1
    fi
fi

# ── 设置环境变量 ──────────────────────────────────────────────────────────────
export DATASET_BASE_PATH="$PBS_JOBFS/datasets"
export ROBOCASA_ASSETS_ROOT="$ASSET_EXTRACTED_DIR"
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/third_party/robosuite:$PROJECT_ROOT/policy/openpi/src:$PROJECT_ROOT/policy/openpi/packages/openpi-client/src:$PYTHONPATH

echo "Project root:      $PROJECT_ROOT"
echo "DATASET_BASE_PATH: $DATASET_BASE_PATH"
echo "ROBOCASA_ASSETS_ROOT: $ROBOCASA_ASSETS_ROOT"
echo "Python:            $(which python)"
echo "PYTHONPATH:        $PYTHONPATH"
echo "RoboCasa environment setup completed."
