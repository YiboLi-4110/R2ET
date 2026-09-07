#!/usr/bin/env bash
# 将 train_char 下指定动物子目录的 FBX 渲染为预览 MP4。
#
# 用法:
#   bash ./render_fbx_videos.sh clouded_leopard_male
#   bash ./render_fbx_videos.sh clouded_leopard_male red_fox_female
#   bash ./render_fbx_videos.sh --all          # 渲染 train_char 下全部子目录
#   LIMIT=2 bash ./render_fbx_videos.sh clouded_leopard_male   # 只渲前 N 个（试跑）
#
# 可选环境变量:
#   BLENDER   Blender 可执行文件路径
#   WORKERS   并行 Blender 进程数（默认 4）
#   GPU_IDS   逗号分隔 GPU 编号（默认 0,1,2,3）
#   ENGINE    cycles|workbench|eevee（默认 cycles）
#   SAMPLES   Cycles 采样数（默认 1）
#   RES       分辨率（默认 640x360）
#   OUT_DIR   输出根目录
#   LIMIT     仅处理前 N 个 FBX（0=全部）
#   OVERWRITE 设为 1 时覆盖已有视频

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER="${BLENDER:-/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender}"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/shepherd/cat_actions/train_char}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/shepherd/cat_actions/train_char_videos}"
WORKERS="${WORKERS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
ENGINE="${ENGINE:-eevee}"
SAMPLES="${SAMPLES:-1}"
RES="${RES:-640x360}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subdir> [subdir2 ...] | --all"
  echo "Example: $0 clouded_leopard_male red_fox_female"
  exit 2
fi

SUBDIR_ARGS=()
if [[ "$1" == "--all" ]]; then
  shift
else
  SUBDIR_ARGS=(--subdirs "$@")
fi

CMD=(
  "$BLENDER" -b -P "${SCRIPT_DIR}/render_fbx_to_video.py" --
  --data_path "$DATA_PATH"
  --output_dir "$OUT_DIR"
  --engine "$ENGINE"
  --resolution "$RES"
  --samples "$SAMPLES"
  --workers "$WORKERS"
  --gpu_ids "$GPU_IDS"
  --threads_per_worker 1
)

if [[ ${#SUBDIR_ARGS[@]} -gt 0 ]]; then
  CMD+=("${SUBDIR_ARGS[@]}")
fi

if [[ "$LIMIT" != "0" ]]; then
  CMD+=(--limit "$LIMIT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running:"
printf '  %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
