#!/usr/bin/env python3
"""
Evaluate skeleton-aware SMAL33 checkpoints on a held-out character split.

Metrics:
  - self-reconstruction: local/quat AE, FK error, attention-joint MSE
  - cross-retarget (in-domain): semantic matrix error, twist, root-velocity ratio, ...
  - cross-retarget (external): optional shepherd -> batch2_dogs pairs when target_data_path is set
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
    SEQ_TAIL_DIM,
    SMAL33_PARENTS,
    dump_json,
    get_height_from_skel,
    list_sequences,
    load_retnet,
    load_shape_for_character,
    load_stats,
    load_window_from_sample,
    setup_cuda_device,
    shape_roots_from_args,
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
    parser.add_argument(
        "--inp_shape_path",
        type=Path,
        default=None,
        help="Optional source-character shape .npz directory for cross eval.",
    )
    parser.add_argument(
        "--tgt_shape_path",
        type=Path,
        default=None,
        help="Optional target-character shape .npz directory for cross eval.",
    )
    parser.add_argument("--stats_path", type=Path, default=REPO_ROOT / "datasets/Planet_Zoo_FBX-smal2/stats")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="ret .pt files; glob ok")
    parser.add_argument("--checkpoint_glob", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "work_dir/train_skeleton_aware_smal33/eval")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--min_frames", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--num_self_samples", type=int, default=200)
    parser.add_argument(
        "--num_cross_pairs",
        type=int,
        default=200,
        help="Number of in-domain cross pairs (val characters from data_path).",
    )
    parser.add_argument(
        "--target_data_path",
        type=Path,
        default=None,
        help="Optional static target motion root (e.g. batch2_dogs/batch2_dogs_q).",
    )
    parser.add_argument(
        "--target_shape_path",
        type=Path,
        default=None,
        help="Shape .npz directory for external targets. Defaults to tgt_shape_path.",
    )
    parser.add_argument(
        "--target_min_frames",
        type=int,
        default=2,
        help="Minimum frames required for external target entries (rest pose = 2).",
    )
    parser.add_argument(
        "--num_cross_external_pairs",
        type=int,
        default=0,
        help="Number of cross pairs from val motion -> external target skeletons. 0 disables.",
    )
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


def list_external_target_samples(data_path, min_frames=2):
    """Load rest-pose target entries (e.g. batch2_dogs) for cross-retarget eval."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"target_data_path does not exist: {data_path}")

    samples = []
    for char_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
        for seq_path in sorted(char_dir.glob("*_seq.npy")):
            stem = seq_path.name[: -len("_seq.npy")]
            skel_path = char_dir / f"{stem}_skel.npy"
            if not skel_path.exists():
                continue
            seq = np.load(seq_path)
            if seq.shape[0] < min_frames:
                continue
            samples.append(
                {
                    "character": char_dir.name,
                    "sequence": stem,
                    "shape_key": stem,
                    "seq_path": seq_path,
                    "skel_path": skel_path,
                    "num_frames": int(seq.shape[0]),
                }
            )
    return samples


def load_shape_for_target(shape_root, folder_name, sequence_stem):
    """Resolve shape npz by sequence stem first, then folder name."""
    for key in (sequence_stem, folder_name):
        try:
            return load_shape_for_character(shape_root, character=key)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No shape npz for sequence '{sequence_stem}' or folder '{folder_name}' under {shape_root}"
    )


def load_external_target_window(source_win, target_sample, stats):
    """Build a target window aligned to source_win length (static external skeleton)."""
    seq_len = source_win["seq"].shape[0]
    skel_raw = np.load(target_sample["skel_path"])
    sequence = np.load(target_sample["seq_path"])
    local_raw = np.reshape(sequence[:, :-SEQ_TAIL_DIM], (sequence.shape[0], NUM_JOINTS, 3))
    skel_raw = skel_raw.copy()
    skel_raw[:, 0, :] = local_raw[:, 0, :]

    skel_w = np.repeat(skel_raw[0:1], seq_len, axis=0)
    skel_n = (skel_w - stats["local_mean"]) / stats["local_std"]
    height = get_height_from_skel(skel_w[0])

    return {
        "seq": source_win["seq"].copy(),
        "skel": skel_n.reshape(seq_len, -1).astype(np.float32),
        "quat": source_win["quat"].copy(),
        "local_gt": source_win["local_gt"].copy(),
        "global_in": source_win["global_in"].copy(),
        "height": np.array([height], dtype=np.float32),
        "mask_len": source_win["mask_len"],
        "character": target_sample["character"],
        "sequence": target_sample["sequence"],
        "shape_key": target_sample["shape_key"],
    }


def compute_cross_metrics(batch, local_b, global_b, quat_b, stats, device, args, pair_label):
    local_a = batch["seqA"][:, :, :-GLOBAL_DIM].reshape(1, -1, NUM_JOINTS, 3)
    local_a_denorm = denorm_local(local_a, stats, device)
    local_b_denorm = denorm_local(local_b, stats, device)
    norm_b, norm_a = RetNet.get_rela_matrix(
        local_b_denorm, local_a_denorm, batch["heightB"], batch["heightA"]
    )
    sem = RetNet.get_sem_loss(ATTENTION_JOINTS, NUM_JOINTS, norm_a, norm_b, batch["mask"])
    twist = RetNet.get_rot_cons_loss(args.alpha, args.euler_ord, quat_b)
    return {
        "sem": float(sem.item()),
        "twist": float(twist.item()),
        "root_vel_ratio_mse": metric_root_vel_ratio(
            batch["global_in"], global_b, batch["heightA"], batch["heightB"]
        ),
        "limb_torso_min_dist": metric_limb_torso_min_dist(local_b_denorm),
        "jerk": metric_temporal_jerk(local_b_denorm),
        "pair": pair_label,
    }


def evaluate_checkpoint(
    model,
    stats,
    parents,
    val_samples,
    inp_shape_root,
    tgt_shape_root,
    args,
    device,
    external_targets=None,
    external_shape_root=None,
):
    rng = np.random.RandomState(args.seed)
    self_pool = val_samples
    if len(self_pool) > args.num_self_samples:
        idx = rng.choice(len(self_pool), args.num_self_samples, replace=False)
        self_pool = [self_pool[i] for i in idx]

    self_metrics = []
    for sample in tqdm(self_pool, desc="self-recon", leave=False):
        win = load_window_from_sample(sample, stats, args.max_length, rng=rng)
        shape = load_shape_for_character(inp_shape_root, sample["character"]).astype(np.float32)
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
        shape_a = load_shape_for_character(inp_shape_root, char_a).astype(np.float32)
        shape_b = load_shape_for_character(tgt_shape_root, char_b).astype(np.float32)
        batch = to_batch(win_a, win_b, shape_a, shape_b, device)
        local_b, global_b, quat_b = forward_retarget(model, batch, stats, parents, device)
        cross_metrics.append(
            compute_cross_metrics(
                batch,
                local_b,
                global_b,
                quat_b,
                stats,
                device,
                args,
                pair_label=f"{char_a}->{char_b}",
            )
        )

    cross_external_metrics = []
    if external_targets and args.num_cross_external_pairs > 0:
        if external_shape_root is None:
            raise ValueError("external_shape_root is required when evaluating external cross pairs.")
        ext_pool = external_targets
        for _ in tqdm(range(args.num_cross_external_pairs), desc="cross-external", leave=False):
            sample_a = val_samples[rng.randint(0, len(val_samples))]
            target_b = ext_pool[rng.randint(0, len(ext_pool))]
            win_a = load_window_from_sample(sample_a, stats, args.max_length, rng=rng)
            win_b = load_external_target_window(win_a, target_b, stats)
            shape_a = load_shape_for_character(
                inp_shape_root, sample_a["character"]
            ).astype(np.float32)
            shape_b = load_shape_for_target(
                external_shape_root,
                target_b["character"],
                target_b["shape_key"],
            ).astype(np.float32)
            batch = to_batch(win_a, win_b, shape_a, shape_b, device)
            local_b, global_b, quat_b = forward_retarget(model, batch, stats, parents, device)
            pair_label = (
                f"{sample_a['character']}/{sample_a['sequence']}"
                f"->{target_b['shape_key']}"
            )
            cross_external_metrics.append(
                compute_cross_metrics(
                    batch,
                    local_b,
                    global_b,
                    quat_b,
                    stats,
                    device,
                    args,
                    pair_label=pair_label,
                )
            )

    def avg_dict(rows, keys):
        if not rows:
            return {k: float("nan") for k in keys}
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    cross_keys = ["sem", "twist", "root_vel_ratio_mse", "limb_torso_min_dist", "jerk"]
    summary = {
        "self": avg_dict(
            self_metrics,
            ["local_ae", "quat_ae", "twist", "local_mse", "attn_mse", "fk_error", "limb_torso_min_dist", "jerk"],
        ),
        "cross": avg_dict(cross_metrics, cross_keys),
        "cross_external": avg_dict(cross_external_metrics, cross_keys),
        "num_self_samples": len(self_metrics),
        "num_cross_pairs": len(cross_metrics),
        "num_cross_external_pairs": len(cross_external_metrics),
    }
    return summary, self_metrics, cross_metrics, cross_external_metrics


def main():
    parser = parse_args()
    p = parser.parse_args()
    if p.config is not None and p.config.exists():
        with open(p.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**cfg)
        p = parser.parse_args()

    inp_shape_root, tgt_shape_root = shape_roots_from_args(
        p.shape_path, p.inp_shape_path, p.tgt_shape_path
    )
    external_shape_root = p.target_shape_path or tgt_shape_root
    device = setup_cuda_device(p.device)
    stats = load_stats(p.stats_path)
    parents = SMAL33_PARENTS

    all_samples = list_sequences(p.data_path, min_frames=p.min_frames)
    train_chars, val_chars = split_characters(all_samples, val_ratio=p.val_ratio, seed=p.seed)
    val_samples = [s for s in all_samples if s["character"] in val_chars]

    external_targets = []
    if p.target_data_path is not None:
        external_targets = list_external_target_samples(
            p.target_data_path, min_frames=p.target_min_frames
        )
        if not external_targets:
            raise SystemExit(
                f"No external target samples found under {p.target_data_path} "
                f"with target_min_frames={p.target_min_frames}."
            )
        if p.num_cross_external_pairs <= 0:
            print(
                f"Loaded {len(external_targets)} external targets but num_cross_external_pairs=0; "
                "skipping external cross eval."
            )
    elif p.num_cross_external_pairs > 0:
        raise SystemExit(
            "num_cross_external_pairs > 0 requires --target_data_path to be set."
        )

    ckpts = resolve_checkpoints(p)
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    p.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "data_path": str(p.data_path),
            "shape_path": str(p.shape_path),
            "inp_shape_path": str(inp_shape_root),
            "tgt_shape_path": str(tgt_shape_root),
            "target_data_path": str(p.target_data_path) if p.target_data_path else None,
            "target_shape_path": str(external_shape_root) if external_targets else None,
            "target_min_frames": p.target_min_frames,
            "num_cross_external_pairs": p.num_cross_external_pairs,
            "external_target_count": len(external_targets),
            "val_ratio": p.val_ratio,
            "val_characters": val_chars,
            "train_characters_count": len(train_chars),
            "val_samples": len(val_samples),
            "max_length": p.max_length,
            "min_frames": p.min_frames,
            "seed": p.seed,
        },
        "checkpoints": {},
    }

    print(f"Val characters ({len(val_chars)}): {val_chars}")
    print(f"Val samples: {len(val_samples)}")
    if external_targets:
        print(
            f"External targets: {len(external_targets)} "
            f"(cross_external pairs per ckpt: {p.num_cross_external_pairs})"
        )
    print(f"Checkpoints: {len(ckpts)}")

    for ckpt in ckpts:
        name = Path(ckpt).name
        print(f"\nEvaluating {name}")
        t0 = time.time()
        model = load_retnet(ckpt, p.ret_model_args, device)
        summary, self_rows, cross_rows, cross_ext_rows = evaluate_checkpoint(
            model,
            stats,
            parents,
            val_samples,
            inp_shape_root,
            tgt_shape_root,
            p,
            device,
            external_targets=external_targets or None,
            external_shape_root=external_shape_root if external_targets else None,
        )
        elapsed = time.time() - t0
        summary["elapsed_sec"] = elapsed
        report["checkpoints"][name] = summary
        dump_json(
            p.output_dir / f"{name.replace('.pt', '')}_detail.json",
            {
                "self": self_rows,
                "cross": cross_rows,
                "cross_external": cross_ext_rows,
            },
        )
        print(
            f"  self: local_ae={summary['self']['local_ae']:.4f} quat_ae={summary['self']['quat_ae']:.4f} "
            f"fk={summary['self']['fk_error']:.6f}"
        )
        print(
            f"  cross: sem={summary['cross']['sem']:.4f} root_vel={summary['cross']['root_vel_ratio_mse']:.6f} "
            f"limb_torso={summary['cross']['limb_torso_min_dist']:.4f}"
        )
        if cross_ext_rows:
            ext = summary["cross_external"]
            print(
                f"  cross_ext: sem={ext['sem']:.4f} root_vel={ext['root_vel_ratio_mse']:.6f} "
                f"limb_torso={ext['limb_torso_min_dist']:.4f} jerk={ext['jerk']:.6f}"
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
