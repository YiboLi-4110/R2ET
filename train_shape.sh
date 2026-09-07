#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Single GPU (physical GPU 2 in this example)
# python -u train_shape_aware_smal33.py \
#   --config ./config/train_shape_aware_smal33.yaml \
#   --device 2 \
#   --batch_size 128

# Multi-GPU DDP: auto-launches torchrun when --device lists multiple IDs.
# batch_size is PER GPU; global batch = batch_size * num_gpus.
# Example: 4x GPU 0,1,2,3 with batch 32 per GPU => global batch 128
# python -u train_shape_aware_smal33.py \
#   --config ./config/train_shape_aware_smal33.yaml \
#   --device 0 1 2 3 \
#   --batch_size 32 \
#   --disable_att_loss \
#   --disable_front_rdf \
#   --sdf_grid_size 24 \
#   --geo_frame_stride 2 \
#   --step 20 35 \
#   --epoch 50

# Enable front-leg RDF if needed:
#   --enable_front_rdf

# Use only 2 GPUs (physical 1 and 3):
# python -u train_shape_aware_smal33.py \
#   --config ./config/train_shape_aware_smal33.yaml \
#   --device 1 3 \
#   --batch_size 64

python -u train_shape_aware_smal33.py \
  --config ./config/train_shepherd_shape_aware_smal33.yaml \
  --device 2 3 \
  --batch_size 12 \
  # --disable_att_loss \
  # --disable_front_rdf \
  # --sdf_grid_size 32 \
  # --geo_frame_stride 1 \
  # --step 20 35 \
  # --master_port 29507 \
