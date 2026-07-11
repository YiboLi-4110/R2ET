#!/usr/bin/env python3
"""
Inspect stage-2 shape-aware SMAL33 outputs with skeleton and skinned-mesh videos.

Compared to inspect_stage1_outputs_smal33.py:
  - Skeleton video: source (green) + stage-2 output (red), optional target rest (blue).
    Stage-1 skeleton is omitted by design.
  - Mesh video: source skinned mesh (green) + stage-2 skinned on target (red).
    Optional stage-1 mesh via --show_stage1_mesh. Default uses all faces (solid skin).

Camera knobs (also in yaml):
  - camera_zoom: >1 moves camera closer to subject
  - view_elev / view_azim: orbit to see head vs tail, etc.

Recommended usage:
  1. Self reconstruction: --self_recon
  2. Cross retarget: input BVH/shape -> target BVH/shape
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from datasets.smal33_motion_io import (
    NUM_JOINTS,
    SMAL33_PARENTS,
    build_model_inputs,
    build_motion_world,
    character_label_from_path,
    dump_json,
    get_inp_from_bvh,
    load_mesh_from_npz,
    load_retnet,
    load_shape_retnet,
    load_shape_vector,
    load_stats,
    rest_skel_to_world,
    setup_cuda_device,
    world_joints_from_motion,
)
from src.linear_blend_skin import linear_blend_skinning

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect stage-2 SMAL33 outputs (skeleton + skinned mesh)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/inspect_shape_aware_smal33_cfg.yaml",
    )
    parser.add_argument(
        "--stage1_weights",
        type=Path,
        default=None,
        help="Stage-1 checkpoint (only needed when --show_stage1_mesh).",
    )
    parser.add_argument(
        "--stage2_weights",
        type=Path,
        default=None,
        help="Stage-2 shape-aware checkpoint.",
    )
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=None,
        help=(
            "Deprecated fallback for Planet Zoo layouts. Cross-retarget inspect "
            "uses load_inp_data.inp_shape_path / tgt_shape_path for mesh assets."
        ),
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=REPO_ROOT / "work_dir/train_shapeaware_smal33/inspect",
    )
    parser.add_argument("--device", type=int, default=0)
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
        "--no_skeleton_video",
        action="store_true",
        help="Skip skeleton compare mp4.",
    )
    parser.add_argument(
        "--no_mesh_video",
        action="store_true",
        help="Skip skinned mesh compare mp4.",
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
        help="Horizontal spacing between compared skeletons. Use 0 to overlap.",
    )
    parser.add_argument(
        "--mesh_spacing",
        type=float,
        default=None,
        help="Horizontal spacing between compared meshes. Defaults to 1.5 * compare_spacing.",
    )
    parser.add_argument(
        "--show_target_rest",
        action="store_true",
        help="Also draw the target rest skeleton in blue (skeleton view only).",
    )
    parser.add_argument(
        "--frame_margin",
        type=float,
        default=0.08,
        help="Axis margin ratio used by plots (before camera_zoom is applied).",
    )
    parser.add_argument(
        "--camera_zoom",
        type=float,
        default=1.0,
        help="Camera zoom factor. >1 moves camera closer (tighter crop), <1 farther.",
    )
    parser.add_argument(
        "--view_elev",
        type=float,
        default=18.0,
        help="3D camera elevation in degrees (vertical angle).",
    )
    parser.add_argument(
        "--view_azim",
        type=float,
        default=-75.0,
        help="3D camera azimuth in degrees (horizontal orbit around subject).",
    )
    parser.add_argument(
        "--mesh_face_stride",
        type=int,
        default=1,
        help="Subsample mesh faces for faster rendering (1 = full solid skin).",
    )
    parser.add_argument(
        "--mesh_alpha",
        type=float,
        default=1.0,
        help="Mesh surface opacity (1.0 = opaque solid skin).",
    )
    parser.add_argument(
        "--show_stage1_mesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include stage-1 skinned mesh in mesh png/mp4 (default: source + stage2 only).",
    )
    parser.add_argument(
        "--mesh_shade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply simple face lighting for smoother skin (matplotlib-version safe).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate.",
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
def run_stage1_inference(model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device):
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


@torch.no_grad()
def run_stage2_inference(model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device):
    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)
    local_b, global_b, quat_b, _, _ = model(
        inp_batch["seq"],
        tgt_batch["seq"],
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
        phase="test",
    )
    return (
        local_b[0].cpu().numpy(),
        global_b[0].cpu().numpy(),
        quat_b[0].cpu().numpy(),
    )


def motion_tpose(skel_arr):
    return skel_arr[0].reshape(NUM_JOINTS, 3).astype(np.float32)


def trim_motion(motion, num_frames):
    out = dict(motion)
    for key in ("quat", "seq", "skel"):
        if key in out and out[key] is not None:
            out[key] = out[key][:num_frames]
    return out


@torch.no_grad()
def skin_mesh_sequence(quat_np, rest_skel_np, mesh_data, device):
    quat_t = torch.from_numpy(quat_np).float().to(device)
    rest_t = torch.from_numpy(rest_skel_np).float().to(device)
    verts_t = torch.from_numpy(mesh_data["vertices"]).float().to(device)
    weights_t = torch.from_numpy(mesh_data["skin_weights"]).float().to(device)
    out = linear_blend_skinning(
        SMAL33_PARENTS, quat_t, rest_t, verts_t, weights_t
    )
    return out.cpu().numpy().astype(np.float32)


def build_target_rest_world(tgt_motion):
    return rest_skel_to_world(tgt_motion["skel"][0], SMAL33_PARENTS)


def root_center_world(world_seq):
    return world_seq - world_seq[:, 0:1, :]


def root_center_pose(joints):
    return joints - joints[0:1]


def root_center_mesh(verts_seq):
    roots = verts_seq[:, 0:1, :]
    return verts_seq - roots


def prepare_visualization_data(input_world, output_world, target_rest_world, view_mode):
    if view_mode == "root_centered":
        vis_input = root_center_world(input_world)
        vis_output = root_center_world(output_world)
        vis_target = root_center_pose(target_rest_world)
    else:
        vis_input = input_world.copy()
        vis_output = output_world.copy()
        vis_target = target_rest_world.copy()
    return vis_input, vis_output, vis_target


def prepare_mesh_visualization(verts_seq, view_mode):
    if view_mode == "root_centered":
        return root_center_mesh(verts_seq)
    return verts_seq.copy()


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


def set_axes_from_points(
    ax,
    points,
    x_margin=0.15,
    y_margin=0.15,
    z_margin=0.15,
    view_elev=18.0,
    view_azim=-75.0,
    camera_zoom=1.0,
):
    zoom = max(float(camera_zoom), 1e-3)
    x_margin = float(x_margin) / zoom
    y_margin = float(y_margin) / zoom
    z_margin = float(z_margin) / zoom

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
    ax.view_init(elev=float(view_elev), azim=float(view_azim))


def ordered_frame_ids(num_frames):
    mid = max(num_frames // 2, 0)
    return {
        "frame_000": 0,
        "frame_mid": mid,
        "frame_last": max(num_frames - 1, 0),
    }


def subsample_faces(faces, stride):
    if stride <= 1:
        return faces
    return faces[::stride]


def mesh_to_plot_coords(verts):
    return np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=-1)


def compute_face_colors(verts, faces, color, shade=True):
    if not shade:
        return color
    rgb = np.array(to_rgb(color), dtype=np.float32)
    pts = verts[faces].astype(np.float64)
    normals = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    light = np.array([0.2, 1.0, 0.3], dtype=np.float64)
    light /= np.linalg.norm(light)
    intensity = 0.4 + 0.6 * np.clip(normals @ light, 0.0, 1.0)
    return np.clip(rgb[None, :] * intensity[:, None], 0.0, 1.0)


def make_mesh_collection(ax, verts, faces, color, alpha=1.0, shade=True):
    plot_verts = mesh_to_plot_coords(verts)
    tri = plot_verts[faces]
    coll_kwargs = {
        "edgecolor": "none",
        "alpha": alpha,
        "linewidths": 0.0,
        "antialiased": True,
    }
    if shade:
        coll_kwargs["facecolors"] = compute_face_colors(plot_verts, faces, color, shade=True)
    else:
        coll_kwargs["facecolor"] = color
    coll = Poly3DCollection(tri, **coll_kwargs)
    coll._mesh_color = color
    coll._mesh_shade = shade
    ax.add_collection3d(coll)
    return coll


def update_mesh_collection(coll, verts, faces):
    plot_verts = mesh_to_plot_coords(verts)
    tri = plot_verts[faces]
    coll.set_verts(tri)
    if getattr(coll, "_mesh_shade", False):
        coll.set_facecolor(
            compute_face_colors(
                plot_verts,
                faces,
                getattr(coll, "_mesh_color", "gray"),
                shade=True,
            )
        )


def shift_mesh_x(verts_seq, x_shift):
    out = verts_seq.copy()
    out[:, :, 0] += x_shift
    return out


def view_axis_kwargs(frame_margin, view_elev, view_azim, camera_zoom):
    return {
        "x_margin": frame_margin,
        "y_margin": frame_margin,
        "z_margin": frame_margin,
        "view_elev": view_elev,
        "view_azim": view_azim,
        "camera_zoom": camera_zoom,
    }


def save_static_skeleton_frames(
    input_world,
    output_world,
    target_rest_world,
    parents,
    save_dir,
    spacing,
    show_target_rest,
    frame_margin,
    view_elev,
    view_azim,
    camera_zoom,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in ordered_frame_ids(len(output_world)).items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, input_world[idx], parents, color="green", x_shift=-spacing)
        draw_skeleton(ax, output_world[idx], parents, color="red", x_shift=0.0)
        points = [
            input_world[idx] + np.array([-spacing, 0.0, 0.0], dtype=np.float32),
            output_world[idx],
        ]
        title = f"{name}: green=source, red=stage2"
        if show_target_rest:
            draw_skeleton(ax, target_rest_world, parents, color="blue", x_shift=spacing)
            points.append(
                target_rest_world + np.array([spacing, 0.0, 0.0], dtype=np.float32)
            )
            title += ", blue=target_rest"
        set_axes_from_points(
            ax,
            points,
            **view_axis_kwargs(frame_margin, view_elev, view_azim, camera_zoom),
        )
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def build_mesh_compare_layers(
    source_mesh,
    stage1_mesh,
    stage2_mesh,
    inp_faces,
    tgt_faces,
    spacing,
    show_stage1_mesh,
):
    layers = [
        {
            "seq": source_mesh,
            "faces": inp_faces,
            "color": "green",
            "shift": -spacing,
            "label": "green=source",
        },
        {
            "seq": stage2_mesh,
            "faces": tgt_faces,
            "color": "red",
            "shift": spacing,
            "label": "red=stage2@target",
        },
    ]
    if show_stage1_mesh:
        layers = [
            layers[0],
            {
                "seq": stage1_mesh,
                "faces": tgt_faces,
                "color": "darkorange",
                "shift": 0.0,
                "label": "orange=stage1@target",
            },
            layers[1],
        ]
    return layers


def mesh_compare_title(layers):
    return ", ".join(layer["label"] for layer in layers)


def save_static_mesh_frames(
    layers,
    save_dir,
    frame_margin,
    mesh_alpha,
    mesh_shade,
    view_elev,
    view_azim,
    camera_zoom,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    title_base = mesh_compare_title(layers)
    num_frames = len(layers[0]["seq"])

    for name, idx in ordered_frame_ids(num_frames).items():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        all_pts = []
        for layer in layers:
            verts = shift_mesh_x(layer["seq"][idx : idx + 1], layer["shift"])[0]
            make_mesh_collection(
                ax,
                verts,
                layer["faces"],
                layer["color"],
                mesh_alpha,
                shade=mesh_shade,
            )
            all_pts.append(mesh_to_plot_coords(verts)[None])
        set_axes_from_points(
            ax,
            [p.reshape(-1, 3) for p in all_pts],
            **view_axis_kwargs(frame_margin, view_elev, view_azim, camera_zoom),
        )
        ax.set_title(f"{name}: {title_base}", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def save_skeleton_compare_video(
    input_world,
    output_world,
    target_rest_world,
    parents,
    save_path,
    spacing,
    show_target_rest,
    frame_margin,
    view_elev,
    view_azim,
    camera_zoom,
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
    title = "green=source, red=stage2"
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
        **view_axis_kwargs(frame_margin, view_elev, view_azim, camera_zoom),
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


def save_mesh_compare_video(
    layers,
    save_path,
    frame_margin,
    mesh_alpha,
    mesh_shade,
    view_elev,
    view_azim,
    camera_zoom,
    fps=30,
):
    shifted_seqs = [
        shift_mesh_x(layer["seq"], layer["shift"]) for layer in layers
    ]
    title = mesh_compare_title(layers)
    num_frames = len(layers[0]["seq"])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    all_pts = np.concatenate(
        [mesh_to_plot_coords(s.reshape(-1, 3)) for s in shifted_seqs],
        axis=0,
    )
    set_axes_from_points(
        ax,
        [all_pts],
        **view_axis_kwargs(frame_margin, view_elev, view_azim, camera_zoom),
    )
    ax.set_title(title, fontsize=10)

    collections = []
    for layer, seq in zip(layers, shifted_seqs):
        collections.append(
            make_mesh_collection(
                ax,
                seq[0],
                layer["faces"],
                layer["color"],
                mesh_alpha,
                shade=mesh_shade,
            )
        )

    def animate(frame_idx):
        changed = []
        for coll, layer, seq in zip(collections, layers, shifted_seqs):
            update_mesh_collection(coll, seq[frame_idx], layer["faces"])
            changed.append(coll)
        return changed

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(num_frames),
        interval=1000.0 / fps,
        blit=False,
    )
    ani.save(str(save_path), writer="ffmpeg", fps=fps)
    plt.close(fig)


def make_summary(
    inp_motion,
    tgt_motion,
    input_world,
    stage2_world,
    target_rest_world,
    quat_stage1,
    quat_stage2,
    local_stage2,
    global_stage2,
    stats,
    self_recon,
    view_mode,
    compare_spacing,
    mesh_spacing,
    show_target_rest,
    frame_margin,
    camera_zoom,
    view_elev,
    view_azim,
    mesh_face_stride,
    mesh_alpha,
    mesh_shade,
    show_stage1_mesh,
    inp_character,
    tgt_character,
):
    quat_norms = np.linalg.norm(quat_stage2, axis=-1)
    summary = {
        "num_frames": int(len(stage2_world)),
        "self_recon": bool(self_recon),
        "view_mode": view_mode,
        "compare_spacing": float(compare_spacing),
        "mesh_spacing": float(mesh_spacing),
        "show_target_rest": bool(show_target_rest),
        "show_stage1_mesh": bool(show_stage1_mesh),
        "frame_margin": float(frame_margin),
        "camera_zoom": float(camera_zoom),
        "view_elev": float(view_elev),
        "view_azim": float(view_azim),
        "mesh_face_stride": int(mesh_face_stride),
        "mesh_alpha": float(mesh_alpha),
        "mesh_shade": bool(mesh_shade),
        "input_character": inp_character,
        "target_character": tgt_character,
        "input_sequence": Path(inp_motion.get("_bvh_path", "")).stem,
        "target_sequence": Path(tgt_motion.get("_bvh_path", "")).stem,
        "inp_shape_path": inp_motion.get("_shape_path", ""),
        "tgt_shape_path": tgt_motion.get("_shape_path", ""),
        "quat_stage2_norm_mean": float(quat_norms.mean()),
        "quat_stage2_norm_max_abs_dev": float(np.max(np.abs(quat_norms - 1.0))),
    }
    local_denorm = local_stage2 * stats["local_std"] + stats["local_mean"]
    summary["stage2_local_abs_mean"] = float(np.mean(np.abs(local_denorm)))
    summary["stage2_global_abs_mean"] = float(np.mean(np.abs(global_stage2)))

    inp_root = input_world[:, 0]
    out_root = stage2_world[:, 0]
    summary["root_disp_input"] = float(np.linalg.norm(inp_root[-1] - inp_root[0]))
    summary["root_disp_stage2"] = float(np.linalg.norm(out_root[-1] - out_root[0]))
    if quat_stage1 is not None:
        summary["stage1_stage2_quat_mse"] = float(
            np.mean((quat_stage1 - quat_stage2) ** 2)
        )
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

    if p.show_stage1_mesh and not p.stage1_weights:
        raise SystemExit("--stage1_weights is required when --show_stage1_mesh is set.")
    if not p.stage2_weights:
        raise SystemExit("--stage2_weights is required (or set in config).")

    load_data = prepare_paths(p)
    device = setup_cuda_device(p.device)
    stats = load_stats(load_data["stats_path"])

    stage2_model = load_shape_retnet(p.stage2_weights, p.ret_model_args, device)
    stage1_model = None
    if p.show_stage1_mesh:
        stage1_model = load_retnet(p.stage1_weights, p.ret_model_args, device)

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

    inp_shape = load_shape_vector(load_data["inp_shape_path"])
    tgt_shape = load_shape_vector(load_data["tgt_shape_path"])

    quat_s1 = None
    if p.show_stage1_mesh:
        _, _, quat_s1 = run_stage1_inference(
            stage1_model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device
        )
    local_s2, global_s2, quat_s2 = run_stage2_inference(
        stage2_model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device
    )

    num_frames = len(inp_motion["quat"])
    if len(quat_s2) != num_frames:
        raise ValueError(
            f"Model output length {len(quat_s2)} != input length {num_frames}"
        )
    if quat_s1 is not None and len(quat_s1) != num_frames:
        raise ValueError(
            f"Model output length {len(quat_s1)} != input length {num_frames}"
        )
    # frame_candidates = [len(quat_s2), len(inp_motion["quat"]), len(tgt_motion["skel"])]
    # if quat_s1 is not None:
    #     frame_candidates.append(len(quat_s1))
    # num_frames = min(frame_candidates)
    # inp_motion = trim_motion(inp_motion, num_frames)
    # tgt_motion = trim_motion(tgt_motion, num_frames)
    # if quat_s1 is not None:
    #     quat_s1 = quat_s1[:num_frames]
    quat_s2 = quat_s2[:num_frames]
    local_s2 = local_s2[:num_frames]
    global_s2 = global_s2[:num_frames]

    inp_char = character_label_from_path(load_data["inp_shape_path"])
    tgt_char = character_label_from_path(load_data["tgt_shape_path"])
    inp_mesh = load_mesh_from_npz(load_data["inp_shape_path"])
    tgt_mesh = load_mesh_from_npz(load_data["tgt_shape_path"])
    inp_faces = subsample_faces(inp_mesh["faces"], p.mesh_face_stride)
    tgt_faces = subsample_faces(tgt_mesh["faces"], p.mesh_face_stride)

    inp_tpose = motion_tpose(inp_motion["skel"])
    tgt_tpose = motion_tpose(tgt_motion["skel"])
    source_quat = inp_motion["quat"].astype(np.float32)

    source_verts = skin_mesh_sequence(source_quat, inp_tpose, inp_mesh, device)
    stage1_verts = None
    if p.show_stage1_mesh:
        stage1_verts = skin_mesh_sequence(quat_s1, tgt_tpose, tgt_mesh, device)
    stage2_verts = skin_mesh_sequence(quat_s2, tgt_tpose, tgt_mesh, device)

    input_world, _, _ = build_motion_world(inp_motion)
    stage2_world = world_joints_from_motion(local_s2, global_s2, stats)
    target_rest_world = build_target_rest_world(tgt_motion)

    vis_input_world, vis_stage2_world, vis_target_rest_world = prepare_visualization_data(
        input_world,
        stage2_world,
        target_rest_world,
        p.view_mode,
    )
    vis_source_mesh = prepare_mesh_visualization(source_verts, p.view_mode)
    vis_stage1_mesh = None
    if stage1_verts is not None:
        vis_stage1_mesh = prepare_mesh_visualization(stage1_verts, p.view_mode)
    vis_stage2_mesh = prepare_mesh_visualization(stage2_verts, p.view_mode)

    skeleton_spacing = p.compare_spacing
    mesh_spacing = (
        p.mesh_spacing
        if p.mesh_spacing is not None
        else max(skeleton_spacing * 1.5, 2.5)
    )
    mesh_layers = build_mesh_compare_layers(
        vis_source_mesh,
        vis_stage1_mesh,
        vis_stage2_mesh,
        inp_faces,
        tgt_faces,
        mesh_spacing,
        p.show_stage1_mesh,
    )

    inp_name = inp_char
    tgt_name = tgt_char
    seq_name = Path(load_data["inp_bvh_path"]).stem
    mode_tag = "self" if p.self_recon else "cross"
    out_dir = p.save_path / f"{mode_tag}_{p.view_mode}_{inp_name}_to_{tgt_name}_{seq_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = dict(
        local_stage2=local_s2.astype(np.float32),
        global_stage2=global_s2.astype(np.float32),
        quat_stage2=quat_s2.astype(np.float32),
        input_world=input_world.astype(np.float32),
        stage2_world=stage2_world.astype(np.float32),
        target_rest_world=target_rest_world.astype(np.float32),
        source_mesh=source_verts.astype(np.float32),
        stage2_mesh=stage2_verts.astype(np.float32),
        vis_input_world=vis_input_world.astype(np.float32),
        vis_stage2_world=vis_stage2_world.astype(np.float32),
        vis_target_rest_world=vis_target_rest_world.astype(np.float32),
        vis_source_mesh=vis_source_mesh.astype(np.float32),
        vis_stage2_mesh=vis_stage2_mesh.astype(np.float32),
    )
    if quat_s1 is not None:
        npz_payload["quat_stage1"] = quat_s1.astype(np.float32)
    if stage1_verts is not None:
        npz_payload["stage1_mesh"] = stage1_verts.astype(np.float32)
        npz_payload["vis_stage1_mesh"] = vis_stage1_mesh.astype(np.float32)
    np.savez_compressed(out_dir / "predictions.npz", **npz_payload)

    summary = make_summary(
        inp_motion,
        tgt_motion,
        input_world,
        stage2_world,
        target_rest_world,
        quat_s1,
        quat_s2,
        local_s2,
        global_s2,
        stats,
        self_recon=p.self_recon,
        view_mode=p.view_mode,
        compare_spacing=skeleton_spacing,
        mesh_spacing=mesh_spacing,
        show_target_rest=p.show_target_rest,
        frame_margin=p.frame_margin,
        camera_zoom=p.camera_zoom,
        view_elev=p.view_elev,
        view_azim=p.view_azim,
        mesh_face_stride=p.mesh_face_stride,
        mesh_alpha=p.mesh_alpha,
        mesh_shade=p.mesh_shade,
        show_stage1_mesh=p.show_stage1_mesh,
        inp_character=inp_char,
        tgt_character=tgt_char,
    )
    dump_json(out_dir / "summary.json", summary)

    save_static_skeleton_frames(
        vis_input_world,
        vis_stage2_world,
        vis_target_rest_world,
        SMAL33_PARENTS,
        out_dir / "skeleton_frames",
        spacing=skeleton_spacing,
        show_target_rest=p.show_target_rest,
        frame_margin=p.frame_margin,
        view_elev=p.view_elev,
        view_azim=p.view_azim,
        camera_zoom=p.camera_zoom,
    )
    save_static_mesh_frames(
        mesh_layers,
        out_dir / "mesh_frames",
        frame_margin=p.frame_margin,
        mesh_alpha=p.mesh_alpha,
        mesh_shade=p.mesh_shade,
        view_elev=p.view_elev,
        view_azim=p.view_azim,
        camera_zoom=p.camera_zoom,
    )

    make_videos = not p.no_video
    if make_videos and not p.no_skeleton_video:
        save_skeleton_compare_video(
            vis_input_world,
            vis_stage2_world,
            vis_target_rest_world,
            SMAL33_PARENTS,
            out_dir / "skeleton_compare.mp4",
            spacing=skeleton_spacing,
            show_target_rest=p.show_target_rest,
            frame_margin=p.frame_margin,
            view_elev=p.view_elev,
            view_azim=p.view_azim,
            camera_zoom=p.camera_zoom,
            fps=p.fps,
        )
    if make_videos and not p.no_mesh_video:
        save_mesh_compare_video(
            mesh_layers,
            out_dir / "mesh_compare.mp4",
            frame_margin=p.frame_margin,
            mesh_alpha=p.mesh_alpha,
            mesh_shade=p.mesh_shade,
            view_elev=p.view_elev,
            view_azim=p.view_azim,
            camera_zoom=p.camera_zoom,
            fps=p.fps,
        )

    print("Saved inspection results:")
    print(f"  dir:              {out_dir}")
    print(f"  npz:              {out_dir / 'predictions.npz'}")
    print(f"  summary:          {out_dir / 'summary.json'}")
    print(f"  skeleton_frames:  {out_dir / 'skeleton_frames'}")
    print(f"  mesh_frames:      {out_dir / 'mesh_frames'}")
    if make_videos and not p.no_skeleton_video:
        print(f"  skeleton_video:   {out_dir / 'skeleton_compare.mp4'}")
    if make_videos and not p.no_mesh_video:
        print(f"  mesh_video:       {out_dir / 'mesh_compare.mp4'}")


if __name__ == "__main__":
    main()