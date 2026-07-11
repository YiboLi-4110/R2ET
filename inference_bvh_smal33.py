#!/usr/bin/env python3
"""
Skeleton-aware SMAL33 inference / visualization.

Loads stage-1 retargetor weights and exports:
  - input BVH
  - target rest BVH
  - retargeted BVH
  - optional side-by-side skeleton mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.smal33_motion_io import (
    SMAL33_PARENTS,
    build_model_inputs,
    get_inp_from_bvh,
    load_retnet,
    load_shape_vector,
    load_stats,
    retarget_to_bvh,
    save_skeleton_video,
    setup_cuda_device,
    world_joints_from_motion,
)

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="SMAL33 skeleton-aware BVH inference")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config/inference_bvh_smal33_cfg.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--save_path", type=Path, default=REPO_ROOT / "save/inference_smal33")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_joint", type=int, default=33)
    parser.add_argument("--make-video", dest="make_video", action="store_true", default=False)
    parser.add_argument(
        "--ret_model_args",
        type=dict,
        default={
            "num_joint": 33,
            "token_channels": 64,
            "hidden_channels_p": 256,
            "embed_channels_p": 128,
            "kp": 0.8,
        },
    )
    parser.add_argument("--load_inp_data", type=dict, default={})
    return parser


@torch.no_grad()
def run_inference(model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device):
    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)
    # parents = torch.from_numpy(SMAL33_PARENTS).to(device)

    local_b, global_b, quat_b, _ = model(
        inp_batch["seq"],
        None,
        inp_batch["skel"],
        tgt_batch["skel"],
        inp_batch["shape"],
        tgt_batch["shape"],
        inp_batch["quat"],
        inp_batch["height"],
        tgt_batch["height"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        SMAL33_PARENTS,
    )
    return (
        local_b[0].cpu().numpy(),
        global_b[0].cpu().numpy(),
        quat_b[0].cpu().numpy(),
    )


def make_comparison_video(inp_motion, local_rt, global_rt, stats, out_mp4):
    inp_local = np.reshape(inp_motion["seq"][:, :-8], (-1, 33, 3))
    inp_global = inp_motion["seq"][:, -8:-4]
    inp_total = np.concatenate(
        [inp_local.reshape(len(inp_local), -1), inp_global], axis=-1
    )

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]
    out_local = local_rt * local_std + local_mean
    out_total = np.concatenate(
        [out_local.reshape(len(local_rt), -1), global_rt], axis=-1
    )

    # animation_plot 期望 (1, T, joints*3+4)
    save_skeleton_video([inp_total[None], out_total[None]], SMAL33_PARENTS, out_mp4)


def Animation_positions_global(anim):
    import Animation

    return Animation.positions_global(anim)


def motion_parse_options(load_data, prefix):
    return {
        "axis_transform": load_data.get(
            f"{prefix}_axis_transform",
            load_data.get("axis_transform", "none"),
        ),
        "forward_mode": load_data.get(
            f"{prefix}_forward_mode",
            load_data.get("forward_mode", "across"),
        ),
    }


def main():
    parser = parse_args()
    p = parser.parse_args()
    if p.config is not None and p.config.exists():
        with open(p.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**cfg)
        p = parser.parse_args()

    if not p.weights:
        raise SystemExit("--weights is required (or set in config).")
    load_data = p.load_inp_data
    required = ["inp_bvh_path", "tgt_bvh_path", "inp_shape_path", "tgt_shape_path", "stats_path"]
    for key in required:
        if key not in load_data:
            raise SystemExit(f"Missing load_inp_data.{key}")

    device = setup_cuda_device(p.device)
    stats = load_stats(load_data["stats_path"])
    model = load_retnet(p.weights, p.ret_model_args, device)

    inp_motion = get_inp_from_bvh(
        load_data["inp_bvh_path"],
        **motion_parse_options(load_data, "inp"),
    )
    tgt_motion = get_inp_from_bvh(
        load_data["tgt_bvh_path"],
        **motion_parse_options(load_data, "tgt"),
    )
    if inp_motion is None or tgt_motion is None:
        raise SystemExit("Failed to parse input/target BVH.")

    inp_shape = load_shape_vector(load_data["inp_shape_path"])
    tgt_shape = load_shape_vector(load_data["tgt_shape_path"])

    local_rt, global_rt, quat_rt = run_inference(
        model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device
    )

    inp_name = Path(load_data["inp_bvh_path"]).parent.name
    tgt_name = Path(load_data["tgt_bvh_path"]).parent.name
    bvh_name = Path(load_data["inp_bvh_path"]).name
    pair_tag = f"{inp_name}_to_{tgt_name}_{Path(bvh_name).stem}"

    p.save_path.mkdir(parents=True, exist_ok=True)
    inp_copy, tgt_copy, out_copy = retarget_to_bvh(
        inp_motion,
        tgt_motion,
        local_rt,
        global_rt,
        quat_rt,
        stats,
        p.save_path,
        pair_tag,
        inp_bvh_path=load_data["inp_bvh_path"],
        tgt_bvh_path=load_data["tgt_bvh_path"],
    )

    print("Saved:")
    print(f"  input:    {inp_copy}")
    print(f"  target:   {tgt_copy}")
    print(f"  retarget: {out_copy}")

    if p.make_video:
        mp4_path = p.save_path / f"{pair_tag}_compare.mp4"
        try:
            make_comparison_video(inp_motion, local_rt, global_rt, stats, mp4_path)
            print(f"  video:    {mp4_path}")
        except Exception as exc:
            print(f"Video export failed ({exc}). BVH files are still valid for Blender.")


if __name__ == "__main__":
    main()
