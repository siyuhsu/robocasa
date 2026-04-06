#!/bin/bash
set -euo pipefail

WORKSPACE=/scratch/sx11/sx0401/robocasa
SIF_FILE=/scratch/sx11/sx0401/workspace/code/cosmos-policy/docker/cosmos-policy_latest.sif

echo "======================================"
echo "RoboCasa Singularity Setup"
echo "======================================"

module purge
module load singularity

if [ ! -f "$SIF_FILE" ]; then
    echo "错误: 找不到 Singularity 镜像: $SIF_FILE"
    exit 1
fi

export HOST_USER_ID=$(id -u)
export HOST_GROUP_ID=$(id -g)

# 启动 Singularity 容器
singularity shell \
  --nv \
  --writable-tmpfs \
  --bind $HOME/.cache:/home/cosmos/.cache \
  --bind $WORKSPACE:/workspace \
  --bind /scratch:/scratch \
  --bind /tmp:/tmp \
  --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  --env HOST_USER_ID=$HOST_USER_ID \
  --env HOST_GROUP_ID=$HOST_GROUP_ID \
  --env MUJOCO_GL=egl \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  --env DISPLAY= \
  --pwd /workspace \
  $SIF_FILE
