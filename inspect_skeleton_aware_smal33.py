#!/usr/bin/env python3
"""
Inspect stage-1 skeleton-aware SMAL33 outputs without relying on BVH export.

This script helps answer a narrow but critical question:
  "Is the stage-1 model output itself reasonable?"

It loads the same inputs as inference_bvh_smal33.py, runs the retargetor once,
then saves:
  - raw model outputs (.npz)
  - summary metrics (.json)
  - world-space comparison video (.mp4, optional)
  - a few static frame snapshots (.png)

Recommended usage:
  1. Self reconstruction: input == target
  2. Cross retarget: input and target are different characters
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import yaml

from datasets.smal33_motion_io import (
    SMAL33_PARENTS,
    build_motion_world,
    character_label_from_path,
    dump_json,
    get_inp_from_bvh,
    load_retnet,
    load_shape_vector,
    load_stats,
    rest_skel_to_world,
    setup_cuda_device,
    world_joints_from_motion,
)

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect stage-1 SMAL33 outputs in world space."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/inference_bvh_smal33_cfg.yaml",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--save_path",
        type=Path,
        default=REPO_ROOT / "work_dir/train_skeleton_aware_smal33/inspect",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_joint", type=int, default=33)
    parser.add_argument(
        "--self_recon",
        action="store_true",
        help="Use input shape/BVH as target as well.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip mp4 export and only save npz/json/png.",
    )
    parser.add_argument(
        "--view_mode",
        type=str,
        default="root_centered",
        choices=["root_centered", "world"],
        help="Visualization mode. root_centered is recommended for diagnosis.",
    )
    parser.add_argument(
        "--compare_spacing",
        type=float,
        default=2.0,
        help="Horizontal spacing between compared skeletons. Use 0 to overlap them.",
    )
    parser.add_argument(
        "--show_target_rest",
        action="store_true",
        help="Also draw the target rest skeleton in blue.",
    )
    parser.add_argument(
        "--frame_margin",
        type=float,
        default=0.08,
        help="Axis margin ratio used by plots.",
    )
    parser.add_argument(
        "--ret_model_args",
        type=yaml.safe_load,
        default={
            "num_joint": 33,
            "token_channels": 64,
            "hidden_channels_p": 256,
            "embed_channels_p": 128,
            "kp": 0.8,
        },
    )
    parser.add_argument("--load_inp_data", type=yaml.safe_load, default={})
    return parser


@torch.no_grad()
def run_inference(model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device):
    from datasets.smal33_motion_io import build_model_inputs

    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)

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


def build_target_rest_world(tgt_motion):
    return rest_skel_to_world(tgt_motion["skel"][0], SMAL33_PARENTS)


def bone_lengths(world_seq, parents):
    lengths = []
    for j, p in enumerate(parents):
        if p == -1:
            continue
        seg = np.linalg.norm(world_seq[:, j] - world_seq[:, p], axis=-1)
        lengths.append(seg)
    return np.stack(lengths, axis=1)


def bone_length_summary(output_world, target_rest_world, parents):
    lens = bone_lengths(output_world, parents)
    mean_l = lens.mean(axis=0)
    std_l = lens.std(axis=0)
    cv = std_l / np.maximum(mean_l, 1e-8)

    target_l = []
    for j, p in enumerate(parents):
        if p == -1:
            continue
        target_l.append(np.linalg.norm(target_rest_world[j] - target_rest_world[p]))
    target_l = np.array(target_l, dtype=np.float32)

    return {
        "bone_len_cv_mean": float(cv.mean()),
        "bone_len_cv_max": float(cv.max()),
        "target_bone_len_mae": float(np.mean(np.abs(mean_l - target_l))),
        "target_bone_len_maxae": float(np.max(np.abs(mean_l - target_l))),
    }


def world_error_summary(input_world, output_world):
    err = np.linalg.norm(output_world - input_world, axis=-1)
    return {
        "self_world_err_mean": float(err.mean()),
        "self_world_err_p95": float(np.percentile(err, 95)),
        "self_world_err_max": float(err.max()),
    }


def quat_norm_summary(quat_rt):
    norms = np.linalg.norm(quat_rt, axis=-1)
    return {
        "quat_norm_mean": float(norms.mean()),
        "quat_norm_p95_abs_dev": float(np.percentile(np.abs(norms - 1.0), 95)),
        "quat_norm_max_abs_dev": float(np.max(np.abs(norms - 1.0))),
    }


def root_motion_summary(input_world, output_world):
    inp_root = input_world[:, 0]
    out_root = output_world[:, 0]
    return {
        "root_disp_input": float(np.linalg.norm(inp_root[-1] - inp_root[0])),
        "root_disp_output": float(np.linalg.norm(out_root[-1] - out_root[0])),
        "root_height_input_mean": float(inp_root[:, 1].mean()),
        "root_height_output_mean": float(out_root[:, 1].mean()),
    }


def states_from_prediction(local_rt, global_rt, stats):
    local = local_rt * stats["local_std"] + stats["local_mean"]
    return np.concatenate([local.reshape(len(local_rt), -1), global_rt], axis=-1)


def draw_skeleton(ax, joints, parents, color, x_shift=0.0, lw=2.0, alpha=1.0):
    joints = joints.copy()
    joints[:, 0] += x_shift
    for j, p in enumerate(parents):
        if p == -1:
            continue
        ax.plot(
            [joints[j, 0], joints[p, 0]],
            [joints[j, 2], joints[p, 2]],
            [joints[j, 1], joints[p, 1]],
            color=color,
            lw=lw,
            alpha=alpha,
        )


def root_center_world(world_seq):
    return world_seq - world_seq[:, 0:1, :]


def root_center_pose(joints):
    return joints - joints[0:1]


def set_axes_from_points(ax, points, x_margin=0.15, y_margin=0.15, z_margin=0.15):
    pts = np.concatenate(points, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-3)
    mins = mins - spans * np.array([x_margin, z_margin, y_margin], dtype=np.float32)
    maxs = maxs + spans * np.array([x_margin, z_margin, y_margin], dtype=np.float32)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[2], maxs[2])
    ax.set_zlim(mins[1], maxs[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    try:
        ax.set_box_aspect((maxs[0] - mins[0], maxs[2] - mins[2], maxs[1] - mins[1]))
    except AttributeError:
        pass
    ax.view_init(elev=18, azim=-75)


def save_static_frames(
    input_world,
    output_world,
    target_rest_world,
    parents,
    save_dir,
    spacing,
    show_target_rest,
    frame_margin,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = OrderedFrameIds(len(output_world))
    for name, idx in frame_ids.items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, input_world[idx], parents, color="green", x_shift=-spacing)
        draw_skeleton(ax, output_world[idx], parents, color="red", x_shift=0.0)
        points = [
            input_world[idx] + np.array([-spacing, 0.0, 0.0], dtype=np.float32),
            output_world[idx],
        ]
        title = f"{name}: green=input, red=output"
        if show_target_rest:
            draw_skeleton(ax, target_rest_world, parents, color="blue", x_shift=spacing)
            points.append(
                target_rest_world + np.array([spacing, 0.0, 0.0], dtype=np.float32)
            )
            title += ", blue=target_rest"
        set_axes_from_points(
            ax,
            points,
            x_margin=frame_margin,
            y_margin=frame_margin,
            z_margin=frame_margin,
        )
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def infer_spacing(seqs):
    all_pts = np.concatenate([s.reshape(-1, 3) for s in seqs], axis=0)
    span_x = float(all_pts[:, 0].max() - all_pts[:, 0].min())
    return max(span_x * 0.15, 2.0)


def OrderedFrameIds(num_frames):
    mid = max(num_frames // 2, 0)
    return {
        "frame_000": 0,
        "frame_mid": mid,
        "frame_last": max(num_frames - 1, 0),
    }


def save_world_compare_video(
    input_world,
    output_world,
    target_rest_world,
    parents,
    save_path,
    spacing,
    show_target_rest,
    frame_margin,
    fps=30,
):
    target_seq = np.repeat(target_rest_world[None], len(output_world), axis=0)
    shifted_in = input_world.copy()
    shifted_in[:, :, 0] -= spacing
    shifted_out = output_world.copy()
    shifted_tgt = target_seq.copy()
    shifted_tgt[:, :, 0] += spacing

    seqs = [shifted_in, shifted_out]
    colors = ["green", "red"]
    title = "green=input, red=output"
    all_pts = [shifted_in.reshape(-1, 3), shifted_out.reshape(-1, 3)]
    if show_target_rest:
        seqs.append(shifted_tgt)
        colors.append("blue")
        all_pts.append(shifted_tgt.reshape(-1, 3))
        title += ", blue=target_rest"

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_from_points(
        ax,
        all_pts,
        x_margin=frame_margin,
        y_margin=frame_margin,
        z_margin=frame_margin,
    )
    ax.set_title(title, fontsize=10)

    lines = []
    for seq, color in zip(seqs, colors):
        seq_lines = []
        for j, p in enumerate(parents):
            if p == -1:
                seq_lines.append(None)
                continue
            line = ax.plot(
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                color=color,
                lw=2.0,
            )[0]
            seq_lines.append(line)
        lines.append(seq_lines)

    def animate(frame_idx):
        changed = []
        for seq_idx, seq in enumerate(seqs):
            joints = seq[frame_idx]
            for j, p in enumerate(parents):
                if p == -1:
                    continue
                x = np.array([joints[j, 0], joints[p, 0]], dtype=np.float64)
                y = np.array([joints[j, 2], joints[p, 2]], dtype=np.float64)
                z = np.array([joints[j, 1], joints[p, 1]], dtype=np.float64)
                line = lines[seq_idx][j]
                if hasattr(line, "set_data_3d"):
                    line.set_data_3d(x, y, z)
                else:
                    line.set_data(x, y)
                    line.set_3d_properties(z)
                changed.append(line)
        return changed

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(len(output_world)),
        interval=1000.0 / fps,
        blit=False,
    )
    ani.save(str(save_path), writer="ffmpeg", fps=fps)
    plt.close(fig)


def prepare_visualization_data(
    input_world,
    output_world,
    target_rest_world,
    view_mode,
):
    if view_mode == "root_centered":
        vis_input = root_center_world(input_world)
        vis_output = root_center_world(output_world)
        vis_target = root_center_pose(target_rest_world)
    else:
        vis_input = input_world.copy()
        vis_output = output_world.copy()
        vis_target = target_rest_world.copy()
    return vis_input, vis_output, vis_target


def make_summary(
    inp_motion,
    tgt_motion,
    input_world,
    output_world,
    target_rest_world,
    quat_rt,
    local_rt,
    global_rt,
    stats,
    self_recon,
    view_mode,
    compare_spacing,
    show_target_rest,
):
    summary = {
        "num_frames": int(len(output_world)),
        "self_recon": bool(self_recon),
        "view_mode": view_mode,
        "compare_spacing": float(compare_spacing),
        "show_target_rest": bool(show_target_rest),
        "input_character": inp_motion.get("character"),
        "target_character": tgt_motion.get("character"),
        "input_sequence": inp_motion.get("sequence"),
        "target_sequence": tgt_motion.get("sequence"),
        "inp_shape_path": inp_motion.get("_shape_path", ""),
        "tgt_shape_path": tgt_motion.get("_shape_path", ""),
        "inp_bvh_path": inp_motion.get("_bvh_path", ""),
        "tgt_bvh_path": tgt_motion.get("_bvh_path", ""),
    }
    summary.update(quat_norm_summary(quat_rt))
    summary.update(root_motion_summary(input_world, output_world))
    summary.update(bone_length_summary(output_world, target_rest_world, SMAL33_PARENTS))

    local_denorm = local_rt * stats["local_std"] + stats["local_mean"]
    summary["local_abs_mean"] = float(np.mean(np.abs(local_denorm)))
    summary["global_abs_mean"] = float(np.mean(np.abs(global_rt)))

    if self_recon:
        summary.update(world_error_summary(input_world, output_world))
    return summary


def prepare_paths(p):
    load_data = dict(p.load_inp_data)
    required = [
        "inp_bvh_path",
        "tgt_bvh_path",
        "inp_shape_path",
        "tgt_shape_path",
        "stats_path",
    ]
    for key in required:
        if key not in load_data:
            raise SystemExit(f"Missing load_inp_data.{key}")

    if p.self_recon:
        load_data["tgt_bvh_path"] = load_data["inp_bvh_path"]
        load_data["tgt_shape_path"] = load_data["inp_shape_path"]
    return load_data


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

    load_data = prepare_paths(p)
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

    inp_motion["_bvh_path"] = str(load_data["inp_bvh_path"])
    tgt_motion["_bvh_path"] = str(load_data["tgt_bvh_path"])
    inp_motion["_shape_path"] = str(load_data["inp_shape_path"])
    tgt_motion["_shape_path"] = str(load_data["tgt_shape_path"])
    inp_motion["character"] = character_label_from_path(load_data["inp_shape_path"])
    tgt_motion["character"] = character_label_from_path(load_data["tgt_shape_path"])
    inp_motion["sequence"] = Path(load_data["inp_bvh_path"]).stem
    tgt_motion["sequence"] = Path(load_data["tgt_bvh_path"]).stem

    inp_shape = load_shape_vector(load_data["inp_shape_path"])
    tgt_shape = load_shape_vector(load_data["tgt_shape_path"])
    local_rt, global_rt, quat_rt = run_inference(
        model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device
    )

    input_world, _, inp_total = build_motion_world(inp_motion)
    output_world = world_joints_from_motion(local_rt, global_rt, stats)
    target_rest_world = build_target_rest_world(tgt_motion)
    output_states = states_from_prediction(local_rt, global_rt, stats)
    vis_input_world, vis_output_world, vis_target_rest_world = prepare_visualization_data(
        input_world,
        output_world,
        target_rest_world,
        p.view_mode,
    )
    spacing = p.compare_spacing

    inp_name = inp_motion["character"]
    tgt_name = tgt_motion["character"]
    seq_name = inp_motion["sequence"]
    mode_tag = "self" if p.self_recon else "cross"
    out_dir = p.save_path / f"{mode_tag}_{p.view_mode}_{inp_name}_to_{tgt_name}_{seq_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "predictions.npz",
        local_rt=local_rt.astype(np.float32),
        global_rt=global_rt.astype(np.float32),
        quat_rt=quat_rt.astype(np.float32),
        input_states=inp_total.astype(np.float32),
        output_states=output_states.astype(np.float32),
        input_world=input_world.astype(np.float32),
        output_world=output_world.astype(np.float32),
        target_rest_world=target_rest_world.astype(np.float32),
        vis_input_world=vis_input_world.astype(np.float32),
        vis_output_world=vis_output_world.astype(np.float32),
        vis_target_rest_world=vis_target_rest_world.astype(np.float32),
    )

    summary = make_summary(
        inp_motion,
        tgt_motion,
        input_world,
        output_world,
        target_rest_world,
        quat_rt,
        local_rt,
        global_rt,
        stats,
        self_recon=p.self_recon,
        view_mode=p.view_mode,
        compare_spacing=spacing,
        show_target_rest=p.show_target_rest,
    )
    dump_json(out_dir / "summary.json", summary)
    save_static_frames(
        vis_input_world,
        vis_output_world,
        vis_target_rest_world,
        SMAL33_PARENTS,
        out_dir / "frames",
        spacing=spacing,
        show_target_rest=p.show_target_rest,
        frame_margin=p.frame_margin,
    )

    if not p.no_video:
        save_world_compare_video(
            vis_input_world,
            vis_output_world,
            vis_target_rest_world,
            SMAL33_PARENTS,
            out_dir / "compare.mp4",
            spacing=spacing,
            show_target_rest=p.show_target_rest,
            frame_margin=p.frame_margin,
        )

    print("Saved inspection results:")
    print(f"  dir:      {out_dir}")
    print(f"  npz:      {out_dir / 'predictions.npz'}")
    print(f"  summary:  {out_dir / 'summary.json'}")
    print(f"  frames:   {out_dir / 'frames'}")
    if not p.no_video:
        print(f"  video:    {out_dir / 'compare.mp4'}")


if __name__ == "__main__":
    main()
