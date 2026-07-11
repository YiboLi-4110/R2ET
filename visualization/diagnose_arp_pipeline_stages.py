#!/usr/bin/env python3
"""
Diagnose where ARP lane orientation mismatch is introduced.

This script compares ARP meshes against a reference lane (copyquat/ours) across
pipeline stages:
  1) raw Blender-world mesh cache
  2) after axis conversion (Blender Z-up -> LBS Y-up)
  3) after optional horizontal translation lock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ARP pipeline stage mismatches.")
    parser.add_argument("--fourway_npz", type=str, required=True, help="Path to fourway_compare.npz")
    parser.add_argument("--arp_mesh_npz", type=str, required=True, help="Path to *_arp_mesh.npz")
    parser.add_argument(
        "--reference_lane",
        type=str,
        default="copyquat",
        choices=["copyquat", "ours"],
        help="Reference target lane in fourway npz",
    )
    parser.add_argument(
        "--lock_anchor",
        type=str,
        default="first",
        choices=["first", "median"],
        help="Anchor used by lock-horizontal stage simulation",
    )
    parser.add_argument("--out_json", type=str, default=None, help="Optional output json path")
    return parser.parse_args()


def align_frames(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if len(arr) == num_frames:
        return arr
    if len(arr) <= 1:
        return np.repeat(arr, num_frames, axis=0)
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(np.int64)
    return arr[idx]


def blender_z_up_to_lbs_y_up(vertices: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    return verts[..., [0, 2, 1]]


def bbox_center_per_frame(vertices: np.ndarray) -> np.ndarray:
    mn = vertices.min(axis=1)
    mx = vertices.max(axis=1)
    return (mn + mx) * 0.5


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) <= 1:
        return verts, np.zeros((len(verts), 3), dtype=np.float32)
    centers = bbox_center_per_frame(verts)
    if anchor == "median":
        ref = np.median(centers, axis=0, keepdims=True).astype(np.float32)
    else:
        ref = centers[:1].astype(np.float32)
    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    return verts - drift[:, None, :], drift


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    centers = bbox_center_per_frame(vertices)
    return vertices - centers[:, None, :]


def apply_candidate_transform(vertices: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float64).copy()
    if name == "identity":
        return out
    if name == "yaw_180":
        out[..., 0] *= -1.0
        out[..., 2] *= -1.0
        return out
    if name == "mirror_x":
        out[..., 0] *= -1.0
        return out
    if name == "mirror_z":
        out[..., 2] *= -1.0
        return out
    raise ValueError(f"Unknown candidate: {name}")


def candidate_shape_relation(subject: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if subject.shape != reference.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch: subject={tuple(subject.shape)} ref={tuple(reference.shape)}",
        }
    s = center_vertices(subject).astype(np.float64)
    r = center_vertices(reference).astype(np.float64)
    candidates = ["identity", "yaw_180", "mirror_x", "mirror_z"]
    scores = {}
    for name in candidates:
        t = apply_candidate_transform(s, name)
        rmse = np.sqrt(np.mean((t - r) ** 2, axis=(1, 2)))
        scores[name] = {
            "rmse_mean": float(rmse.mean()),
            "rmse_p95": float(np.percentile(rmse, 95.0)),
        }
    best = min(scores, key=lambda k: scores[k]["rmse_mean"])
    return {"comparable": True, "best_candidate": best, "scores": scores}


def trajectory_metrics(vertices: np.ndarray, ref_vertices: np.ndarray) -> dict[str, Any]:
    c = bbox_center_per_frame(vertices).astype(np.float64)
    r = bbox_center_per_frame(ref_vertices).astype(np.float64)
    c2 = c[:, [0, 2]]
    r2 = r[:, [0, 2]]
    disp = c2[-1] - c2[0]
    span = c.max(axis=0) - c.min(axis=0)
    step = c[1:] - c[:-1]
    if np.linalg.norm(disp) < 1e-8:
        fwd = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd = disp / np.linalg.norm(disp)
    right = np.array([fwd[1], -fwd[0]], dtype=np.float64)
    lateral = (c2 - c2[0]) @ right
    ref_fwd = (r2 - r2[0]) @ fwd
    sub_fwd = (c2 - c2[0]) @ fwd
    corr = 0.0
    if np.std(ref_fwd) > 1e-8 and np.std(sub_fwd) > 1e-8:
        corr = float(np.corrcoef(ref_fwd, sub_fwd)[0, 1])
    return {
        "center_span_xyz": span.tolist(),
        "center_span_xz": float(np.linalg.norm(span[[0, 2]])),
        "step_speed_xz_mean": float(np.linalg.norm(step[:, [0, 2]], axis=-1).mean()) if len(step) else 0.0,
        "lateral_std": float(np.std(lateral)),
        "lateral_sign_changes": int(np.sum((np.sign(lateral[1:]) * np.sign(lateral[:-1])) < 0)) if len(lateral) > 1 else 0,
        "forward_corr_with_reference": corr,
    }


def main() -> None:
    args = parse_args()
    fourway_path = Path(args.fourway_npz)
    arp_mesh_path = Path(args.arp_mesh_npz)
    if not fourway_path.exists():
        raise FileNotFoundError(f"fourway npz not found: {fourway_path}")
    if not arp_mesh_path.exists():
        raise FileNotFoundError(f"arp mesh npz not found: {arp_mesh_path}")

    fourway = np.load(str(fourway_path))
    ref_key = "copyquat_vertices" if args.reference_lane == "copyquat" else "ours_vertices"
    if ref_key not in fourway.files:
        raise KeyError(f"reference key missing in fourway npz: {ref_key}")
    ref = np.asarray(fourway[ref_key], dtype=np.float32)
    num_frames = int(ref.shape[0])

    arp_payload = np.load(str(arp_mesh_path))
    arp_raw_zup = np.asarray(arp_payload["vertices"], dtype=np.float32)
    arp_raw_zup = align_frames(arp_raw_zup, num_frames)

    # Stage A: raw Blender-world Z-up (map into common axis for comparison only).
    # We map to Y-up here for fair per-vertex relation with reference lanes.
    stage_axis = blender_z_up_to_lbs_y_up(arp_raw_zup)
    # Stage B: after lock-horizontal (same as current export pipeline).
    stage_locked, drift = remove_horizontal_drift(stage_axis, vertical_axis=1, anchor=args.lock_anchor)

    report: dict[str, Any] = {
        "inputs": {
            "fourway_npz": str(fourway_path),
            "arp_mesh_npz": str(arp_mesh_path),
            "reference_lane": args.reference_lane,
            "reference_key": ref_key,
            "lock_anchor": args.lock_anchor,
        },
        "shapes": {
            "reference": list(ref.shape),
            "arp_raw_zup": list(arp_raw_zup.shape),
            "arp_axis_converted": list(stage_axis.shape),
            "arp_locked": list(stage_locked.shape),
        },
        "stage_comparison_to_reference": {
            "axis_converted": candidate_shape_relation(stage_axis, ref),
            "locked_horizontal": candidate_shape_relation(stage_locked, ref),
        },
        "stage_trajectory": {
            "reference": trajectory_metrics(ref, ref),
            "axis_converted": trajectory_metrics(stage_axis, ref),
            "locked_horizontal": trajectory_metrics(stage_locked, ref),
            "lock_drift_norm": {
                "max": float(np.linalg.norm(drift, axis=-1).max()) if len(drift) else 0.0,
                "p95": float(np.percentile(np.linalg.norm(drift, axis=-1), 95.0)) if len(drift) else 0.0,
            },
        },
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[diag] wrote: {out_path}")


if __name__ == "__main__":
    main()
