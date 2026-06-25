#!/usr/bin/env python3
"""
Evaluate shape-aware SMAL33 checkpoints on a held-out character split.

Metrics (aligned with stage-1 eval + stage-2 geometry):
  - self-reconstruction: local/quat AE, FK error, jerk, limb-torso joint distance
  - cross-retarget: semantic matrix, twist, root velocity, jerk, limb-torso distance
  - cross-retarget geometry (mesh RDF): rep_lh/rh/tail_hind + penetration frame rates

Compare stage-2 checkpoints against an optional stage-1 baseline on the same val samples.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.smal33_motion_io import (
    ATTENTION_JOINTS,
    BODY_JOINTS,
    GLOBAL_DIM,
    LIMB_JOINTS,
    NUM_JOINTS,
    SMAL33_PARENTS,
    dump_json,
    list_sequences,
    load_retnet,
    load_shape_retnet,
    load_shape_vector,
    load_stats,
    load_window_from_sample,
    setup_cuda_device,
    split_characters,
)
from src.mesh_geometry_cache import build_mesh_geometry_cache
from src.forward_kinematics import FK
from src.model_shape_aware_smal33 import RetNet as ShapeRetNet
from src.model_skeleton_aware_smal33 import RetNet as SkeletonRetNet

REPO_ROOT = Path(__file__).resolve().parent

# Bone-part lists (match datasets/train_feeder_r2et_smal33.py).
TORSO_BONES = [0, 1, 2, 3, 4, 5, 6]
HEAD_BONES = [15, 16, 17]
LEFT_FRONT_BONES = [7, 8, 9, 10]
RIGHT_FRONT_BONES = [11, 12, 13, 14]
LEFT_HIND_BONES = [18, 19, 20, 21]
RIGHT_HIND_BONES = [22, 23, 24, 25]
TAIL_BONES = [26, 27, 28, 29, 30, 31, 32]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SMAL33 shape-aware checkpoints."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/train_q",
    )
    parser.add_argument(
        "--shape_path",
        type=Path,
        default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/train_shape",
    )
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=None,
        help="mesh npz dir for RDF eval (defaults to shape_path)",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/stats",
    )
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--checkpoint_glob", type=str, default=None)
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="stage-1 skeleton-aware .pt for comparison (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "work_dir/train_shapeaware_smal33/eval",
    )
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--min_frames", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_self_samples", type=int, default=100)
    parser.add_argument("--num_cross_pairs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--euler_ord", type=str, default="yzx")
    parser.add_argument("--sdf_grid_size", type=int, default=20)
    parser.add_argument("--geo_frame_stride", type=int, default=4)
    parser.add_argument(
        "--compute_front_rdf",
        action="store_true",
        default=False,
        help="include front-leg RDF in geometry metrics",
    )
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
    return parser


def resolve_checkpoints(args):
    if args.checkpoints:
        paths = []
        for item in args.checkpoints:
            paths.extend(sorted(glob.glob(item)))
        return sorted(set(paths))
    if args.checkpoint_glob:
        return sorted(glob.glob(args.checkpoint_glob))
    default_dir = REPO_ROOT / "work_dir/train_shapeaware_smal33"
    return sorted(glob.glob(str(default_dir / "r2et_shape_aware_smal33_ret-*.pt")))


def build_mesh_vertex_groups(mesh_file_dic):
    groups = {
        "torso": {},
        "head": {},
        "left_front": {},
        "right_front": {},
        "left_hind": {},
        "right_hind": {},
        "tail": {},
        "paws": {},
    }
    for mesh_name, fbx_data in mesh_file_dic.items():
        vertex_part_np = fbx_data["vertex_part"]
        vertex_num = vertex_part_np.shape[0]
        lst = {k: [] for k in groups.keys()}
        for i in range(vertex_num):
            part = int(vertex_part_np[i])
            if part in TORSO_BONES:
                lst["torso"].append(i)
            if part in HEAD_BONES:
                lst["head"].append(i)
            if part in LEFT_FRONT_BONES:
                lst["left_front"].append(i)
            if part in RIGHT_FRONT_BONES:
                lst["right_front"].append(i)
            if part in LEFT_HIND_BONES:
                lst["left_hind"].append(i)
            if part in RIGHT_HIND_BONES:
                lst["right_hind"].append(i)
            if part in TAIL_BONES:
                lst["tail"].append(i)
            if part in [10, 14, 21, 25]:
                lst["paws"].append(i)
        for key in groups:
            groups[key][mesh_name] = lst[key]
    return groups


def load_mesh_assets(mesh_path):
    mesh_file_dic = {}
    for npz_path in sorted(mesh_path.glob("*.npz")):
        if npz_path.name.startswith("."):
            continue
        mesh_file_dic[npz_path.stem] = np.load(npz_path)
    if not mesh_file_dic:
        raise FileNotFoundError(f"No mesh npz under {mesh_path}")
    mesh_groups = build_mesh_vertex_groups(mesh_file_dic)
    mesh_geom_cache = build_mesh_geometry_cache(mesh_file_dic, mesh_groups)
    return mesh_groups, mesh_geom_cache


def resolve_mesh_name(character, mesh_geom_cache):
    if character in mesh_geom_cache:
        return character
    for name in mesh_geom_cache:
        if character in name or name in character:
            return name
    raise KeyError(f"No mesh for character '{character}' in cache keys")


def load_shape_for_character(shape_path, character):
    npz = Path(shape_path) / f"{character}.npz"
    if not npz.exists():
        candidates = list(Path(shape_path).glob(f"*{character}*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No shape npz for character {character}")
        npz = candidates[0]
    return load_shape_vector(npz).astype(np.float32)


def to_batch(sample_a, sample_b, shape_a, shape_b, device):
    max_len = sample_a["seq"].shape[0]
    mask = torch.zeros((1, max_len), dtype=torch.float32, device=device)
    mask[0, : sample_a["mask_len"]] = 1.0
    return {
        "seqA": torch.from_numpy(sample_a["seq"])[None].to(device),
        "skelA": torch.from_numpy(sample_a["skel"])[None].to(device),
        "seqB": torch.from_numpy(sample_b["seq"])[None].to(device),
        "skelB": torch.from_numpy(sample_b["skel"])[None].to(device),
        "quatA": torch.from_numpy(sample_a["quat"])[None].to(device),
        "shapeA": torch.from_numpy(shape_a)[None].to(device),
        "shapeB": torch.from_numpy(shape_b)[None].to(device),
        "heightA": torch.from_numpy(sample_a["height"])[None].to(device),
        "heightB": torch.from_numpy(sample_b["height"])[None].to(device),
        "local_gt": torch.from_numpy(sample_b["local_gt"])[None].to(device),
        "global_in": torch.from_numpy(sample_a["global_in"])[None].to(device),
        "mask": mask,
    }


@torch.no_grad()
def forward_skeleton(model, batch, stats, parents, device):
    local_b, global_b, quat_b, _ = model(
        batch["seqA"],
        batch["seqB"],
        batch["skelA"],
        batch["skelB"],
        batch["shapeA"],
        batch["shapeB"],
        batch["quatA"],
        batch["heightA"],
        batch["heightB"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        parents,
    )
    return local_b, global_b, quat_b


@torch.no_grad()
def forward_shape(model, batch, stats, parents, device):
    local_b, global_b, quat_b, _, _ = model(
        batch["seqA"],
        batch["seqB"],
        batch["skelA"],
        batch["skelB"],
        batch["shapeA"],
        batch["shapeB"],
        batch["quatA"],
        batch["heightA"],
        batch["heightB"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        parents,
        phase="test",
    )
    return local_b, global_b, quat_b


def batch_skel_to_tpose(skel_b, local_mean, local_std):
    tpose = skel_b[:, 0, :].reshape(-1, NUM_JOINTS, 3)
    return tpose * local_std + local_mean


def denorm_local(local_rt, stats, device):
    local_mean = torch.from_numpy(stats["local_mean"]).float().to(device)
    local_std = torch.from_numpy(stats["local_std"]).float().to(device)
    return local_rt * local_std[:, None, :, :] + local_mean[:, None, :, :]


def fk_error(local_rt, quat_rt, skel_b0, stats, parents, device):
    local_mean = torch.from_numpy(stats["local_mean"]).float().to(device)
    local_std = torch.from_numpy(stats["local_std"]).float().to(device)
    parents_t = torch.from_numpy(parents).to(device)
    t_pose = batch_skel_to_tpose(skel_b0, local_mean, local_std)
    errs = []
    for t in range(local_rt.shape[1]):
        fk_pos = FK.run(parents_t, t_pose, quat_rt[:, t])
        fk_norm = (fk_pos - local_mean) / local_std
        err = torch.mean((fk_norm - local_rt[:, t]) ** 2).item()
        errs.append(err)
    return float(np.mean(errs))


def metric_limb_torso_min_dist(local_denorm):
    arr = local_denorm[0].cpu().numpy()
    dists = []
    for t in range(arr.shape[0]):
        body = arr[t, BODY_JOINTS]
        limbs = arr[t, LIMB_JOINTS]
        d = np.linalg.norm(body[:, None, :] - limbs[None, :, :], axis=-1)
        dists.append(float(d.min()))
    return float(np.mean(dists))


def metric_temporal_jerk(local_denorm):
    arr = local_denorm[0].cpu().numpy()
    if arr.shape[0] < 3:
        return 0.0
    acc = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    return float(np.mean(acc**2))


def metric_root_vel_ratio(global_in, global_out, height_a, height_b):
    vin = global_in[0].cpu().numpy()
    vout = global_out[0].cpu().numpy()
    ha = float(height_a[0, 0].item())
    hb = float(height_b[0, 0].item())
    norm_in = vin[:, :3] / max(ha, 1e-8)
    norm_out = vout[:, :3] / max(hb, 1e-8)
    return float(np.mean((norm_in - norm_out) ** 2))


def compute_geometry_metrics(
    quat_b,
    skel_b,
    char_b,
    mesh_groups,
    mesh_geom_cache,
    stats,
    parents,
    device,
    args,
):
    mesh_name = resolve_mesh_name(char_b, mesh_geom_cache)
    cache_entry = mesh_geom_cache[mesh_name]
    vertices = cache_entry["vertices"].to(device)
    sk_weights = cache_entry["skin_weights"].to(device)

    local_mean = torch.from_numpy(stats["local_mean"]).float().to(device)
    local_std = torch.from_numpy(stats["local_std"]).float().to(device)
    t_pose_b = batch_skel_to_tpose(skel_b, local_mean, local_std)[0]
    # t_pose_b = skel_b[:, 0, :].reshape(NUM_JOINTS, 3)
    # t_pose_b = t_pose_b * local_std + local_mean

    # parents_t = torch.from_numpy(parents).to(device)
    geo = ShapeRetNet.get_rep_eval_stats(
        parents,
        quat_b[0],
        t_pose_b,
        vertices,
        sk_weights,
        mesh_groups["torso"][mesh_name],
        mesh_groups["head"][mesh_name],
        mesh_groups["left_front"][mesh_name],
        mesh_groups["right_front"][mesh_name],
        mesh_groups["left_hind"][mesh_name],
        mesh_groups["right_hind"][mesh_name],
        mesh_groups["tail"][mesh_name],
        sdf_grid_size=args.sdf_grid_size,
        frame_stride=args.geo_frame_stride,
        hull_cache=cache_entry,
        compute_front_rdf=args.compute_front_rdf,
    )
    return geo


def avg_dict(rows, keys):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def geo_priority_score(geo_summary):
    """Lower is better: hind + tail penetration focus."""
    return (
        geo_summary["rep_lh"]
        + geo_summary["rep_rh"]
        + geo_summary["rep_tail_hind"]
        + geo_summary["pen_rate_lh"]
        + geo_summary["pen_rate_rh"]
        + geo_summary["pen_rate_tail_hind"]
    )


def delta_dict(current, baseline, keys):
    return {k: float(current[k] - baseline[k]) for k in keys}


def evaluate_checkpoint(
    model,
    stats,
    parents,
    val_samples,
    shape_path,
    mesh_groups,
    mesh_geom_cache,
    args,
    device,
    cross_pairs,
    forward_fn,
):
    rng = np.random.RandomState(args.seed)

    self_pool = val_samples
    if len(self_pool) > args.num_self_samples:
        idx = rng.choice(len(self_pool), args.num_self_samples, replace=False)
        self_pool = [self_pool[i] for i in idx]

    self_metrics = []
    for sample in tqdm(self_pool, desc="self-recon", leave=False):
        win = load_window_from_sample(sample, stats, args.max_length, rng=rng)
        shape = load_shape_for_character(shape_path, sample["character"])
        batch = to_batch(win, win, shape, shape, device)
        local_b, global_b, quat_b = forward_fn(
            model, batch, stats, parents, device
        )

        ae_reg = torch.ones((1, 1), device=device)
        quat_mean = torch.from_numpy(stats["quat_mean"]).float().to(device)
        quat_std = torch.from_numpy(stats["quat_std"]).float().to(device)
        quat_denorm = (
            batch["quatA"] * quat_std[:, None, :, :]
            + quat_mean[:, None, :, :]
        )
        local_ae, quat_ae = SkeletonRetNet.get_recon_loss(
            ATTENTION_JOINTS,
            NUM_JOINTS,
            ae_reg,
            batch["mask"],
            local_b,
            batch["local_gt"],
            quat_denorm,
            quat_b,
        )
        twist = SkeletonRetNet.get_rot_cons_loss(
            args.alpha, args.euler_ord, quat_b
        )
        local_denorm = denorm_local(local_b, stats, device)
        self_metrics.append(
            {
                "local_ae": float(local_ae.item()),
                "quat_ae": float(quat_ae.item()),
                "twist": float(twist.item()),
                "local_mse": float(
                    torch.mean((local_b - batch["local_gt"]) ** 2).item()
                ),
                "attn_mse": float(
                    torch.mean(
                        (local_b[:, :, ATTENTION_JOINTS] - batch["local_gt"][:, :, ATTENTION_JOINTS])
                        ** 2
                    ).item()
                ),
                "fk_error": fk_error(
                    local_b, quat_b, batch["skelB"], stats, parents, device
                ),
                "limb_torso_min_dist": metric_limb_torso_min_dist(local_denorm),
                "jerk": metric_temporal_jerk(local_denorm),
            }
        )

    cross_metrics = []
    for pair in tqdm(cross_pairs, desc="cross", leave=False):
        char_a, char_b = pair["char_a"], pair["char_b"]
        sample_a = pair["sample_a"]
        sample_b = pair["sample_b"]
        win_a = load_window_from_sample(sample_a, stats, args.max_length, rng=rng)
        win_b = load_window_from_sample(sample_b, stats, args.max_length, rng=rng)
        shape_a = load_shape_for_character(shape_path, char_a)
        shape_b = load_shape_for_character(shape_path, char_b)
        batch = to_batch(win_a, win_b, shape_a, shape_b, device)
        local_b, global_b, quat_b = forward_fn(
            model, batch, stats, parents, device
        )

        local_a = batch["seqA"][:, :, :-GLOBAL_DIM].reshape(1, -1, NUM_JOINTS, 3)
        local_a_denorm = denorm_local(local_a, stats, device)
        local_b_denorm = denorm_local(local_b, stats, device)
        norm_b, norm_a = SkeletonRetNet.get_rela_matrix(
            local_b_denorm, local_a_denorm, batch["heightB"], batch["heightA"]
        )
        sem = SkeletonRetNet.get_sem_loss(
            ATTENTION_JOINTS, NUM_JOINTS, norm_a, norm_b, batch["mask"]
        )
        twist = SkeletonRetNet.get_rot_cons_loss(
            args.alpha, args.euler_ord, quat_b
        )
        geo = compute_geometry_metrics(
            quat_b,
            batch["skelB"],
            char_b,
            mesh_groups,
            mesh_geom_cache,
            stats,
            parents,
            device,
            args,
        )

        row = {
            "sem": float(sem.item()),
            "twist": float(twist.item()),
            "root_vel_ratio_mse": metric_root_vel_ratio(
                batch["global_in"], global_b, batch["heightA"], batch["heightB"]
            ),
            "limb_torso_min_dist": metric_limb_torso_min_dist(local_b_denorm),
            "jerk": metric_temporal_jerk(local_b_denorm),
            "pair": f"{char_a}->{char_b}",
        }
        row.update(geo)
        cross_metrics.append(row)

    self_keys = [
        "local_ae",
        "quat_ae",
        "twist",
        "local_mse",
        "attn_mse",
        "fk_error",
        "limb_torso_min_dist",
        "jerk",
    ]
    cross_skeleton_keys = [
        "sem",
        "twist",
        "root_vel_ratio_mse",
        "limb_torso_min_dist",
        "jerk",
    ]
    cross_geo_keys = [
        "rep_lh",
        "rep_rh",
        "rep_tail_hind",
        "pen_rate_lh",
        "pen_rate_rh",
        "pen_rate_tail_hind",
    ]
    if args.compute_front_rdf:
        cross_geo_keys = ["rep_lf", "rep_rf", "pen_rate_lf", "pen_rate_rf"] + cross_geo_keys

    summary = {
        "self": avg_dict(self_metrics, self_keys),
        "cross": {
            **avg_dict(cross_metrics, cross_skeleton_keys),
            "geometry": avg_dict(cross_metrics, cross_geo_keys),
        },
        "num_self_samples": len(self_metrics),
        "num_cross_pairs": len(cross_metrics),
    }
    summary["cross"]["geometry"]["geo_priority_score"] = geo_priority_score(
        summary["cross"]["geometry"]
    )
    return summary, self_metrics, cross_metrics


def build_cross_pairs(val_samples, num_pairs, seed):
    rng = np.random.RandomState(seed)
    chars = sorted({s["character"] for s in val_samples})
    if len(chars) < 2:
        raise ValueError("Need at least 2 val characters for cross-retarget eval.")
    char_to_samples = {}
    for s in val_samples:
        char_to_samples.setdefault(s["character"], []).append(s)

    pairs = []
    for _ in range(num_pairs):
        char_a, char_b = rng.choice(chars, size=2, replace=False)
        sample_a = char_to_samples[char_a][rng.randint(0, len(char_to_samples[char_a]))]
        sample_b = char_to_samples[char_b][rng.randint(0, len(char_to_samples[char_b]))]
        pairs.append(
            {
                "char_a": char_a,
                "char_b": char_b,
                "sample_a": sample_a,
                "sample_b": sample_b,
            }
        )
    return pairs


def print_checkpoint_summary(name, summary):
    g = summary["cross"]["geometry"]
    print(
        f"  self: local_ae={summary['self']['local_ae']:.4f} "
        f"quat_ae={summary['self']['quat_ae']:.4f} fk={summary['self']['fk_error']:.6f}"
    )
    print(
        f"  cross: sem={summary['cross']['sem']:.4f} "
        f"root_vel={summary['cross']['root_vel_ratio_mse']:.6f} "
        f"limb_torso={summary['cross']['limb_torso_min_dist']:.4f}"
    )
    print(
        f"  geo: rep_lh={g['rep_lh']:.4f} rep_rh={g['rep_rh']:.4f} "
        f"rep_tail={g['rep_tail_hind']:.4f} "
        f"pen_lh={g['pen_rate_lh']:.3f} pen_rh={g['pen_rate_rh']:.3f} "
        f"pen_tail={g['pen_rate_tail_hind']:.3f} "
        f"score={g['geo_priority_score']:.4f}"
    )


def main():
    parser = parse_args()
    p = parser.parse_args()
    if p.config is not None and p.config.exists():
        with open(p.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**cfg)
        p = parser.parse_args()

    mesh_path = p.mesh_path or p.shape_path
    device = setup_cuda_device(p.device)
    stats = load_stats(p.stats_path)
    parents = SMAL33_PARENTS

    all_samples = list_sequences(p.data_path, min_frames=p.min_frames)
    train_chars, val_chars = split_characters(
        all_samples, val_ratio=p.val_ratio, seed=p.seed
    )
    val_samples = [s for s in all_samples if s["character"] in val_chars]

    ckpts = resolve_checkpoints(p)
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    mesh_groups, mesh_geom_cache = load_mesh_assets(Path(mesh_path))
    cross_pairs = build_cross_pairs(val_samples, p.num_cross_pairs, p.seed)

    p.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "data_path": str(p.data_path),
            "shape_path": str(p.shape_path),
            "mesh_path": str(mesh_path),
            "val_ratio": p.val_ratio,
            "val_characters": sorted(val_chars),
            "train_characters_count": len(train_chars),
            "val_samples": len(val_samples),
            "max_length": p.max_length,
            "seed": p.seed,
            "num_self_samples": p.num_self_samples,
            "num_cross_pairs": p.num_cross_pairs,
            "sdf_grid_size": p.sdf_grid_size,
            "geo_frame_stride": p.geo_frame_stride,
            "compute_front_rdf": p.compute_front_rdf,
        },
        "checkpoints": {},
    }

    print(f"Val characters ({len(val_chars)}): {sorted(val_chars)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Checkpoints: {len(ckpts)}")
    print(f"Cross pairs (fixed seed): {len(cross_pairs)}")

    baseline_summary = None
    if p.baseline_checkpoint:
        print(f"\nEvaluating baseline: {Path(p.baseline_checkpoint).name}")
        t0 = time.time()
        baseline_model = load_retnet(
            p.baseline_checkpoint, p.ret_model_args, device
        )
        baseline_summary, _, _ = evaluate_checkpoint(
            baseline_model,
            stats,
            parents,
            val_samples,
            p.shape_path,
            mesh_groups,
            mesh_geom_cache,
            p,
            device,
            cross_pairs,
            forward_skeleton,
        )
        baseline_summary["elapsed_sec"] = time.time() - t0
        report["baseline"] = {
            "file": Path(p.baseline_checkpoint).name,
            "summary": baseline_summary,
        }
        print_checkpoint_summary("baseline", baseline_summary)

    for ckpt in ckpts:
        name = Path(ckpt).name
        print(f"\nEvaluating {name}")
        t0 = time.time()
        model = load_shape_retnet(ckpt, p.ret_model_args, device)
        summary, self_rows, cross_rows = evaluate_checkpoint(
            model,
            stats,
            parents,
            val_samples,
            p.shape_path,
            mesh_groups,
            mesh_geom_cache,
            p,
            device,
            cross_pairs,
            forward_shape,
        )
        summary["elapsed_sec"] = time.time() - t0

        if baseline_summary is not None:
            geo_keys = list(summary["cross"]["geometry"].keys())
            skel_keys = ["sem", "twist", "root_vel_ratio_mse", "limb_torso_min_dist", "jerk"]
            summary["delta_vs_baseline"] = {
                "cross_skeleton": delta_dict(
                    summary["cross"], baseline_summary["cross"], skel_keys
                ),
                "cross_geometry": delta_dict(
                    summary["cross"]["geometry"],
                    baseline_summary["cross"]["geometry"],
                    geo_keys,
                ),
            }

        report["checkpoints"][name] = summary
        dump_json(
            p.output_dir / f"{name.replace('.pt', '')}_detail.json",
            {"self": self_rows, "cross": cross_rows},
        )
        print_checkpoint_summary(name, summary)
        if baseline_summary is not None:
            dg = summary["delta_vs_baseline"]["cross_geometry"]
            print(
                f"  vs baseline geo delta: rep_rh={dg['rep_rh']:+.4f} "
                f"rep_tail={dg['rep_tail_hind']:+.4f} "
                f"pen_rh={dg['pen_rate_rh']:+.3f} "
                f"pen_tail={dg['pen_rate_tail_hind']:+.3f}"
            )

    best_geo = min(
        report["checkpoints"].items(),
        key=lambda kv: kv[1]["cross"]["geometry"]["geo_priority_score"],
    )
    report["recommended_checkpoint"] = {
        "file": best_geo[0],
        "reason": "lowest cross.geometry.geo_priority_score (hind+tail RDF + penetration rates)",
        "geo_priority_score": best_geo[1]["cross"]["geometry"]["geo_priority_score"],
    }
    dump_json(p.output_dir / "eval_summary.json", report)
    print(f"\nRecommended checkpoint (geometry): {best_geo[0]}")
    print(f"Report saved to: {p.output_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
