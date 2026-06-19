#!/usr/bin/env python3
"""
Evaluate skeleton-aware SMAL33 checkpoints on a held-out character split.

Metrics:
  - self-reconstruction: local/quat AE, FK error, attention-joint MSE
  - cross-retarget: semantic matrix error, twist, root-velocity ratio, limb-torso distance, jerk
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
    load_stats,
    load_window_from_sample,
    setup_cuda_device,
    split_characters,
)
from src.forward_kinematics import FK
from src.model_skeleton_aware_smal33 import RetNet

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SMAL33 skeleton-aware checkpoints.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data_path", type=Path, default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/train_q")
    parser.add_argument("--shape_path", type=Path, default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/train_shape")
    parser.add_argument("--stats_path", type=Path, default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/stats")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="ret .pt files; glob ok")
    parser.add_argument("--checkpoint_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "work_dir/train_skeleton_aware_smal33/eval")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--min_frames", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_self_samples", type=int, default=200)
    parser.add_argument("--num_cross_pairs", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--nu", type=float, default=100.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--euler_ord", type=str, default="yzx")
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
    default_dir = REPO_ROOT / "work_dir/train_skeleton_aware_smal33"
    return sorted(glob.glob(str(default_dir / "r2et_skeleton_aware_smal33_ret-*.pt")))


def load_shape_for_character(shape_path, character):
    npz = Path(shape_path) / f"{character}.npz"
    if not npz.exists():
        candidates = list(Path(shape_path).glob(f"*{character}*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No shape npz for character {character}")
        npz = candidates[0]
    from datasets.smal33_motion_io import load_shape_vector

    return load_shape_vector(npz)


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
def forward_retarget(model, batch, stats, parents, device):
    parents_t = torch.from_numpy(parents).to(device)
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


def fk_error(local_rt, quat_rt, skel_b0, stats, parents, device):
    local_mean = torch.from_numpy(stats["local_mean"]).float().to(device)
    local_std = torch.from_numpy(stats["local_std"]).float().to(device)
    parents_t = torch.from_numpy(parents).to(device)

    t_pose = batch_skel_to_tpose(skel_b0, local_mean, local_std)
    # quat_rt from model forward is already denormalized (see model_skeleton_aware_smal33.py).

    errs = []
    for t in range(local_rt.shape[1]):
        fk_pos = FK.run(parents_t, t_pose, quat_rt[:, t])
        fk_norm = (fk_pos - local_mean) / local_std
        err = torch.mean((fk_norm - local_rt[:, t]) ** 2).item()
        errs.append(err)
    return float(np.mean(errs))


def batch_skel_to_tpose(skel_b, local_mean, local_std):
    bs, tlen, _ = skel_b.shape
    tpose = skel_b[:, 0, :].reshape(bs, NUM_JOINTS, 3)
    return tpose * local_std + local_mean


def denorm_local(local_rt, stats, device):
    local_mean = torch.from_numpy(stats["local_mean"]).float().to(device)
    local_std = torch.from_numpy(stats["local_std"]).float().to(device)
    return local_rt * local_std[:, None, :, :] + local_mean[:, None, :, :]


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
    return float(np.mean(acc ** 2))


def metric_root_vel_ratio(global_in, global_out, height_a, height_b):
    vin = global_in[0].cpu().numpy()
    vout = global_out[0].cpu().numpy()
    ha = float(height_a[0, 0].item())
    hb = float(height_b[0, 0].item())
    norm_in = vin[:, :3] / max(ha, 1e-8)
    norm_out = vout[:, :3] / max(hb, 1e-8)
    return float(np.mean((norm_in - norm_out) ** 2))


def evaluate_checkpoint(model, stats, parents, val_samples, shape_path, args, device):
    rng = np.random.RandomState(args.seed)
    self_pool = val_samples
    if len(self_pool) > args.num_self_samples:
        idx = rng.choice(len(self_pool), args.num_self_samples, replace=False)
        self_pool = [self_pool[i] for i in idx]

    self_metrics = []
    for sample in tqdm(self_pool, desc="self-recon", leave=False):
        win = load_window_from_sample(sample, stats, args.max_length, rng=rng)
        shape = load_shape_for_character(shape_path, sample["character"]).astype(np.float32)
        batch = to_batch(win, win, shape, shape, device)
        local_b, global_b, quat_b = forward_retarget(model, batch, stats, parents, device)

        ae_reg = torch.ones((1, 1), device=device)
        quat_mean = torch.from_numpy(stats["quat_mean"]).float().to(device)
        quat_std = torch.from_numpy(stats["quat_std"]).float().to(device)
        quat_denorm = (
            batch["quatA"] * quat_std[:, None, :, :]
            + quat_mean[:, None, :, :]
        )
        local_ae, quat_ae = RetNet.get_recon_loss(
            ATTENTION_JOINTS,
            NUM_JOINTS,
            ae_reg,
            batch["mask"],
            local_b,
            batch["local_gt"],
            quat_denorm,
            quat_b,
        )
        twist = RetNet.get_rot_cons_loss(args.alpha, args.euler_ord, quat_b)
        local_denorm = denorm_local(local_b, stats, device)
        local_mse = torch.mean((local_b - batch["local_gt"]) ** 2).item()
        attn_mse = torch.mean((local_b[:, :, ATTENTION_JOINTS] - batch["local_gt"][:, :, ATTENTION_JOINTS]) ** 2).item()
        fk_err = fk_error(local_b, quat_b, batch["skelB"], stats, parents, device)

        self_metrics.append(
            {
                "local_ae": float(local_ae.item()),
                "quat_ae": float(quat_ae.item()),
                "twist": float(twist.item()),
                "local_mse": local_mse,
                "attn_mse": attn_mse,
                "fk_error": fk_err,
                "limb_torso_min_dist": metric_limb_torso_min_dist(local_denorm),
                "jerk": metric_temporal_jerk(local_denorm),
            }
        )

    cross_metrics = []
    chars = sorted({s["character"] for s in val_samples})
    char_to_samples = {}
    for s in val_samples:
        char_to_samples.setdefault(s["character"], []).append(s)

    for _ in tqdm(range(args.num_cross_pairs), desc="cross", leave=False):
        char_a, char_b = rng.choice(chars, size=2, replace=False)
        sample_a = char_to_samples[char_a][rng.randint(0, len(char_to_samples[char_a]))]
        sample_b = char_to_samples[char_b][rng.randint(0, len(char_to_samples[char_b]))]
        win_a = load_window_from_sample(sample_a, stats, args.max_length, rng=rng)
        win_b = load_window_from_sample(sample_b, stats, args.max_length, rng=rng)
        shape_a = load_shape_for_character(shape_path, char_a).astype(np.float32)
        shape_b = load_shape_for_character(shape_path, char_b).astype(np.float32)
        batch = to_batch(win_a, win_b, shape_a, shape_b, device)
        local_b, global_b, quat_b = forward_retarget(model, batch, stats, parents, device)

        local_a = batch["seqA"][:, :, :-GLOBAL_DIM].reshape(1, -1, NUM_JOINTS, 3)
        local_a_denorm = denorm_local(local_a, stats, device)
        local_b_denorm = denorm_local(local_b, stats, device)
        norm_b, norm_a = RetNet.get_rela_matrix(
            local_b_denorm, local_a_denorm, batch["heightB"], batch["heightA"]
        )
        sem = RetNet.get_sem_loss(ATTENTION_JOINTS, NUM_JOINTS, norm_a, norm_b, batch["mask"])
        twist = RetNet.get_rot_cons_loss(args.alpha, args.euler_ord, quat_b)

        cross_metrics.append(
            {
                "sem": float(sem.item()),
                "twist": float(twist.item()),
                "root_vel_ratio_mse": metric_root_vel_ratio(
                    batch["global_in"], global_b, batch["heightA"], batch["heightB"]
                ),
                "limb_torso_min_dist": metric_limb_torso_min_dist(local_b_denorm),
                "jerk": metric_temporal_jerk(local_b_denorm),
                "pair": f"{char_a}->{char_b}",
            }
        )

    def avg_dict(rows, keys):
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    summary = {
        "self": avg_dict(
            self_metrics,
            ["local_ae", "quat_ae", "twist", "local_mse", "attn_mse", "fk_error", "limb_torso_min_dist", "jerk"],
        ),
        "cross": avg_dict(
            cross_metrics,
            ["sem", "twist", "root_vel_ratio_mse", "limb_torso_min_dist", "jerk"],
        ),
        "num_self_samples": len(self_metrics),
        "num_cross_pairs": len(cross_metrics),
    }
    return summary, self_metrics, cross_metrics


def main():
    parser = parse_args()
    p = parser.parse_args()
    if p.config is not None and p.config.exists():
        with open(p.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**cfg)
        p = parser.parse_args()

    device = setup_cuda_device(p.device)
    stats = load_stats(p.stats_path)
    parents = SMAL33_PARENTS

    all_samples = list_sequences(p.data_path, min_frames=p.min_frames)
    train_chars, val_chars = split_characters(all_samples, val_ratio=p.val_ratio, seed=p.seed)
    val_samples = [s for s in all_samples if s["character"] in val_chars]

    ckpts = resolve_checkpoints(p)
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    p.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "data_path": str(p.data_path),
            "val_ratio": p.val_ratio,
            "val_characters": val_chars,
            "train_characters_count": len(train_chars),
            "val_samples": len(val_samples),
            "max_length": p.max_length,
            "seed": p.seed,
        },
        "checkpoints": {},
    }

    print(f"Val characters ({len(val_chars)}): {val_chars}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Checkpoints: {len(ckpts)}")

    for ckpt in ckpts:
        name = Path(ckpt).name
        print(f"\nEvaluating {name}")
        t0 = time.time()
        model = load_retnet(ckpt, p.ret_model_args, device)
        summary, self_rows, cross_rows = evaluate_checkpoint(
            model, stats, parents, val_samples, p.shape_path, p, device
        )
        elapsed = time.time() - t0
        summary["elapsed_sec"] = elapsed
        report["checkpoints"][name] = summary
        dump_json(p.output_dir / f"{name.replace('.pt', '')}_detail.json", {"self": self_rows, "cross": cross_rows})
        print(
            f"  self: local_ae={summary['self']['local_ae']:.4f} quat_ae={summary['self']['quat_ae']:.4f} "
            f"fk={summary['self']['fk_error']:.6f}"
        )
        print(
            f"  cross: sem={summary['cross']['sem']:.4f} root_vel={summary['cross']['root_vel_ratio_mse']:.6f} "
            f"limb_torso={summary['cross']['limb_torso_min_dist']:.4f}"
        )

    best = min(
        report["checkpoints"].items(),
        key=lambda kv: kv[1]["self"]["local_ae"] + kv[1]["cross"]["sem"],
    )
    report["recommended_checkpoint"] = {
        "file": best[0],
        "reason": "lowest combined self.local_ae + cross.sem",
        "score": float(best[1]["self"]["local_ae"] + best[1]["cross"]["sem"]),
    }
    dump_json(p.output_dir / "eval_summary.json", report)
    print(f"\nRecommended checkpoint: {best[0]}")
    print(f"Report saved to: {p.output_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
