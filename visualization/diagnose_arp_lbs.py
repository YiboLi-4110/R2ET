#!/usr/bin/env python3
"""
Diagnose ARP BVH -> LBS inconsistencies.

This script is intentionally independent from Blender. It checks whether the ARP
BVH quaternions are consistent with:

1) the ARP BVH's own parsed rest skeleton
2) the target rest skeleton currently used by export_fourway_compare_smal33.py

If (1) is good but (2) is bad, the ARP retarget/BMAP may be fine and the bug is
in our export/render interpretation of ARP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.smal33_motion_io import (  # noqa: E402
    NUM_JOINTS,
    SMAL33_PARENTS,
    get_inp_from_bvh,
    motion_local_positions,
)
from Quaternions import Quaternions  # noqa: E402


PAW_JOINTS = {
    "left_front": 10,
    "right_front": 14,
    "left_hind": 21,
    "right_hind": 25,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose ARP LBS consistency.")
    parser.add_argument("--config", type=Path, default=_PROJECT_ROOT / "config/visualization_compare_smal33.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest_index", type=Path, default=None)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--arp_bvh", type=Path, default=None)
    parser.add_argument("--target_bvh", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest_args(args) -> list[dict[str, Any]]:
    if args.manifest:
        return [json.loads(args.manifest.read_text(encoding="utf-8"))]
    if args.manifest_index:
        payload = json.loads(args.manifest_index.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--manifest_index must point to a JSON list")
        if args.case_id:
            payload = [m for m in payload if m.get("case_id") == args.case_id]
            if not payload:
                raise SystemExit(f"case_id not found in manifest index: {args.case_id}")
        return payload
    if args.arp_bvh and args.target_bvh:
        return [
            {
                "case_id": args.case_id or args.arp_bvh.stem,
                "arp_bvh_path": str(args.arp_bvh),
                "target_bvh_path": str(args.target_bvh),
            }
        ]
    raise SystemExit("Provide --manifest, --manifest_index, or both --arp_bvh and --target_bvh.")


def align_frames(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if len(arr) == num_frames:
        return arr
    if len(arr) <= 1:
        return np.repeat(arr, num_frames, axis=0)
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(np.int64)
    return arr[idx]


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    xy, yz, xz = qx * y2, qy * z2, qx * z2
    return np.stack(
        [
            np.stack([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1),
            np.stack([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1),
            np.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def fk_numpy(local_offsets: np.ndarray, quat: np.ndarray) -> np.ndarray:
    local_offsets = np.asarray(local_offsets, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    rot = quat_to_matrix(quat)
    frames = quat.shape[0]
    local_tf = np.zeros((frames, NUM_JOINTS, 4, 4), dtype=np.float64)
    local_tf[:, :, 3, 3] = 1.0
    local_tf[:, :, :3, :3] = rot
    local_tf[:, :, :3, 3] = local_offsets[None]

    global_tf = np.zeros_like(local_tf)
    global_tf[:, 0] = local_tf[:, 0]
    for joint, parent in enumerate(SMAL33_PARENTS):
        if joint == 0:
            continue
        global_tf[:, joint] = np.matmul(global_tf[:, parent], local_tf[:, joint])
    return global_tf[:, :, :3, 3]


def local_basis_alignment_quats(source_skel: np.ndarray, target_skel: np.ndarray) -> np.ndarray:
    source = np.asarray(source_skel, dtype=np.float64).reshape(NUM_JOINTS, 3)
    target = np.asarray(target_skel, dtype=np.float64).reshape(NUM_JOINTS, 3)
    corrections = np.zeros((NUM_JOINTS, 4), dtype=np.float64)
    corrections[:, 0] = 1.0
    for joint in range(1, NUM_JOINTS):
        src = source[joint]
        tgt = target[joint]
        if np.linalg.norm(src) < 1e-8 or np.linalg.norm(tgt) < 1e-8:
            continue
        corrections[joint] = Quaternions.between(tgt[None], src[None]).qs[0]
    return corrections


def align_quats_to_target_basis(
    quat: np.ndarray,
    source_skel: np.ndarray,
    target_skel: np.ndarray,
    order: str = "right",
) -> np.ndarray:
    corrections = local_basis_alignment_quats(source_skel, target_skel)
    corrections = np.broadcast_to(corrections[None], quat.shape)
    if order == "right":
        aligned = Quaternions(corrections) * Quaternions(quat)
    elif order == "left":
        aligned = Quaternions(quat) * Quaternions(corrections)
    else:
        raise ValueError(f"Unknown basis align order: {order}")
    return aligned.normalized().qs


def norm(x: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt(np.sum(np.square(x), axis=axis))


def error_stats(delta: np.ndarray) -> dict[str, float]:
    dist = norm(delta, axis=-1)
    return {
        "mean": float(np.mean(dist)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def skeleton_compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64).reshape(NUM_JOINTS, 3)
    cand = np.asarray(candidate, dtype=np.float64).reshape(NUM_JOINTS, 3)
    ref_len = norm(ref[1:])
    cand_len = norm(cand[1:])
    dots = np.sum(ref[1:] * cand[1:], axis=-1) / np.maximum(ref_len * cand_len, 1e-8)
    ratios = cand_len / np.maximum(ref_len, 1e-8)
    return {
        "bone_len_ratio_mean": float(np.mean(ratios)),
        "bone_len_ratio_min": float(np.min(ratios)),
        "bone_len_ratio_max": float(np.max(ratios)),
        "bone_dir_dot_mean": float(np.mean(dots)),
        "bone_dir_dot_min": float(np.min(dots)),
        "bone_dir_dot_lt_0_count": int(np.sum(dots < 0.0)),
        "bone_dir_dot_lt_0_5_count": int(np.sum(dots < 0.5)),
    }


def paw_stats(local_pos: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(local_pos) > 1:
        for name, idx in PAW_JOINTS.items():
            v = norm(local_pos[1:, idx] - local_pos[:-1, idx])
            out[f"{name}_vel_p95"] = float(np.percentile(v, 95))
            out[f"{name}_vel_max"] = float(np.max(v))
    front_dist = norm(local_pos[:, PAW_JOINTS["left_front"]] - local_pos[:, PAW_JOINTS["right_front"]])
    hind_dist = norm(local_pos[:, PAW_JOINTS["left_hind"]] - local_pos[:, PAW_JOINTS["right_hind"]])
    out["front_lr_dist_min"] = float(np.min(front_dist))
    out["front_lr_dist_min_frame"] = int(np.argmin(front_dist))
    out["hind_lr_dist_min"] = float(np.min(hind_dist))
    out["hind_lr_dist_min_frame"] = int(np.argmin(hind_dist))
    return out


def parse_motion(path: str | Path, axis: str, forward_mode: str, canonicalize: bool):
    return get_inp_from_bvh(
        str(path),
        axis_transform=axis,
        forward_mode=forward_mode,
        canonicalize_bind_pose=canonicalize,
    )


def target_bvh_from_manifest(
    manifest: dict[str, Any],
    configured_axis: str,
) -> tuple[str, str, bool]:
    if manifest.get("target_bvh_path"):
        return str(manifest["target_bvh_path"]), configured_axis, True
    if manifest.get("ours_bvh_path"):
        rest_copy = Path(manifest["ours_bvh_path"]).with_name(
            Path(manifest["ours_bvh_path"]).name.replace("_retarget.bvh", "_target_rest.bvh")
        )
        if rest_copy.exists():
            # Generated target-rest copies have already gone through export's
            # target parsing path; do not apply the configured axis transform
            # a second time when diagnosing older manifests.
            return str(rest_copy), "none", False
    raise FileNotFoundError(
        "target_bvh_path is missing from manifest and generated *_target_rest.bvh was not found. "
        "Re-run export with intermediates kept, or pass --target_bvh explicitly."
    )


def diagnose_case(manifest: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    motion_cfg = cfg.get("motion", {})
    target_axis = motion_cfg.get("tgt_axis_transform", "none")
    target_forward = motion_cfg.get("tgt_forward_mode", "body")
    config_arp_axis = motion_cfg.get("arp_axis_transform", "none")
    config_arp_forward = motion_cfg.get("arp_forward_mode", "body")

    target_bvh, target_parse_axis, target_parse_canonicalize = target_bvh_from_manifest(
        manifest,
        target_axis,
    )
    target_motion = parse_motion(
        target_bvh,
        target_parse_axis,
        target_forward,
        target_parse_canonicalize,
    )
    if target_motion is None:
        raise RuntimeError(f"Failed to parse target BVH: {target_bvh}")
    target_skel = target_motion["skel"][0].astype(np.float64)

    candidates = [
        ("config_current", config_arp_axis, config_arp_forward, False),
        ("config_current_canonicalized", config_arp_axis, config_arp_forward, True),
        ("target_axis_canonicalized", target_axis, target_forward, True),
        ("target_axis_not_canonicalized", target_axis, target_forward, False),
        ("none_not_canonicalized", "none", config_arp_forward, False),
        ("none_canonicalized", "none", config_arp_forward, True),
        ("shepherd_y_z_x_canonicalized", "shepherd_y_z_x", target_forward, True),
    ]

    variants = []
    seen = set()
    for label, axis, forward, canonicalize in candidates:
        key = (axis, forward, canonicalize)
        if key in seen:
            continue
        seen.add(key)
        arp_motion = parse_motion(manifest["arp_bvh_path"], axis, forward, canonicalize)
        if arp_motion is None:
            variants.append({"label": label, "error": "failed to parse ARP BVH"})
            continue

        frames = min(len(arp_motion["quat"]), len(motion_local_positions(arp_motion)))
        arp_quat = align_frames(arp_motion["quat"].astype(np.float64), frames)
        arp_seq_pos = align_frames(motion_local_positions(arp_motion).astype(np.float64), frames)
        arp_skel = arp_motion["skel"][0].astype(np.float64)

        fk_with_arp_skel = fk_numpy(arp_skel, arp_quat)
        fk_with_target_skel = fk_numpy(target_skel, arp_quat)
        right_aligned_quat = align_quats_to_target_basis(arp_quat, arp_skel, target_skel, order="right")
        left_aligned_quat = align_quats_to_target_basis(arp_quat, arp_skel, target_skel, order="left")
        fk_with_right_aligned_target_skel = fk_numpy(target_skel, right_aligned_quat)
        fk_with_left_aligned_target_skel = fk_numpy(target_skel, left_aligned_quat)
        own_err = error_stats(fk_with_arp_skel - arp_seq_pos)
        target_err = error_stats(fk_with_target_skel - arp_seq_pos)
        right_aligned_target_err = error_stats(fk_with_right_aligned_target_skel - arp_seq_pos)
        left_aligned_target_err = error_stats(fk_with_left_aligned_target_skel - arp_seq_pos)
        if right_aligned_target_err["mean"] <= left_aligned_target_err["mean"]:
            best_aligned_order = "right"
            best_aligned_target_err = right_aligned_target_err
            best_aligned_fk = fk_with_right_aligned_target_skel
        else:
            best_aligned_order = "left"
            best_aligned_target_err = left_aligned_target_err
            best_aligned_fk = fk_with_left_aligned_target_skel

        variant = {
            "label": label,
            "axis_transform": axis,
            "forward_mode": forward,
            "canonicalize_bind_pose": canonicalize,
            "frames": int(frames),
            "arp_bind_pose_canonicalized": bool(arp_motion.get("_bind_pose_canonicalized", False)),
            "arp_skel_vs_target_skel": skeleton_compare(target_skel, arp_skel),
            "fk_arp_quat_with_arp_skel_vs_arp_seq": own_err,
            "fk_arp_quat_with_target_skel_vs_arp_seq": target_err,
            "fk_basis_aligned_right_quat_with_target_skel_vs_arp_seq": right_aligned_target_err,
            "fk_basis_aligned_left_quat_with_target_skel_vs_arp_seq": left_aligned_target_err,
            "best_basis_aligned_order": best_aligned_order,
            "fk_basis_aligned_quat_with_target_skel_vs_arp_seq": best_aligned_target_err,
            "target_skel_penalty_mean": float(target_err["mean"] - own_err["mean"]),
            "basis_aligned_penalty_mean": float(best_aligned_target_err["mean"] - own_err["mean"]),
            "paw_stats_from_arp_seq": paw_stats(arp_seq_pos),
            "paw_stats_from_target_skel_fk": paw_stats(fk_with_target_skel),
            "paw_stats_from_basis_aligned_target_skel_fk": paw_stats(best_aligned_fk),
        }
        variants.append(variant)

    usable = [v for v in variants if "error" not in v]
    best_for_current_lbs = min(
        usable,
        key=lambda v: v["fk_arp_quat_with_target_skel_vs_arp_seq"]["mean"],
        default=None,
    )
    best_self_consistency = min(
        usable,
        key=lambda v: v["fk_arp_quat_with_arp_skel_vs_arp_seq"]["mean"],
        default=None,
    )
    best_basis_aligned_lbs = min(
        usable,
        key=lambda v: v["fk_basis_aligned_quat_with_target_skel_vs_arp_seq"]["mean"],
        default=None,
    )
    return {
        "case_id": manifest.get("case_id"),
        "arp_bvh_path": manifest["arp_bvh_path"],
        "target_bvh_path": target_bvh,
        "target_parse": {
            "axis_transform": target_parse_axis,
            "forward_mode": target_forward,
            "canonicalize_bind_pose": target_parse_canonicalize,
        },
        "current_export_arp_parse": {
            "axis_transform": config_arp_axis,
            "forward_mode": config_arp_forward,
            "canonicalize_bind_pose": False,
            "lbs_skeleton": "target_skel",
            "basis_align": bool(motion_cfg.get("arp_lbs_basis_align", True)),
        },
        "best_for_current_lbs": best_for_current_lbs,
        "best_basis_aligned_lbs": best_basis_aligned_lbs,
        "best_self_consistency": best_self_consistency,
        "variants": variants,
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    manifests = load_manifest_args(args)
    reports = [diagnose_case(manifest, cfg) for manifest in manifests]
    payload: Any = reports[0] if len(reports) == 1 else reports
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[diag-arp-lbs] wrote: {args.output}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    for report in reports:
        cur = report["best_for_current_lbs"]
        aligned = report["best_basis_aligned_lbs"]
        self_best = report["best_self_consistency"]
        print(
            "[diag-arp-lbs] "
            f"{report['case_id']}: best_current_lbs={cur['label'] if cur else None} "
            f"target_err_mean={cur['fk_arp_quat_with_target_skel_vs_arp_seq']['mean'] if cur else None} "
            f"best_basis_aligned={aligned['label'] if aligned else None} "
            f"aligned_err_mean={aligned['fk_basis_aligned_quat_with_target_skel_vs_arp_seq']['mean'] if aligned else None} "
            f"best_self={self_best['label'] if self_best else None} "
            f"self_err_mean={self_best['fk_arp_quat_with_arp_skel_vs_arp_seq']['mean'] if self_best else None}"
        )


if __name__ == "__main__":
    main()
