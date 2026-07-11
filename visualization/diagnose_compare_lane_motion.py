#!/usr/bin/env python3
"""
Diagnose lane-level motion/orientation mismatches in fourway_compare.npz.

Focus:
1) Is ARP globally opposite (180 yaw) or mirrored (left/right inversion)?
2) Does ARP trajectory show abnormal lateral "S-curve" wobble vs CopyQuat/R2ET?
3) Quantify center drift and per-frame motion statistics for each lane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose fourway lane motion/orientation.")
    parser.add_argument("--npz", type=str, required=True, help="Path to fourway_compare.npz")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path")
    parser.add_argument(
        "--reference_lane",
        type=str,
        default="copyquat",
        choices=["copyquat", "ours", "source"],
        help="Reference lane for ARP orientation/shape comparisons",
    )
    return parser.parse_args()


def bbox_center_per_frame(vertices_t_v_3: np.ndarray) -> np.ndarray:
    mn = vertices_t_v_3.min(axis=1)
    mx = vertices_t_v_3.max(axis=1)
    return (mn + mx) * 0.5


def trajectory_stats(vertices_t_v_3: np.ndarray) -> dict[str, Any]:
    # Convention in this project: Y-up local vertices before Blender render conversion.
    # Horizontal plane => X/Z, vertical => Y.
    centers = bbox_center_per_frame(vertices_t_v_3).astype(np.float64)
    delta = centers[1:] - centers[:-1]
    speed_all = np.linalg.norm(delta, axis=-1)
    speed_xz = np.linalg.norm(delta[:, [0, 2]], axis=-1)
    speed_y = np.abs(delta[:, 1])

    span = centers.max(axis=0) - centers.min(axis=0)
    return {
        "frames": int(len(vertices_t_v_3)),
        "center_span_xyz": span.tolist(),
        "center_span_xz": float(np.linalg.norm(span[[0, 2]])),
        "step_speed_all_mean": float(speed_all.mean()) if len(speed_all) else 0.0,
        "step_speed_all_p95": float(np.percentile(speed_all, 95.0)) if len(speed_all) else 0.0,
        "step_speed_xz_mean": float(speed_xz.mean()) if len(speed_xz) else 0.0,
        "step_speed_xz_p95": float(np.percentile(speed_xz, 95.0)) if len(speed_xz) else 0.0,
        "step_speed_y_mean": float(speed_y.mean()) if len(speed_y) else 0.0,
        "step_speed_y_p95": float(np.percentile(speed_y, 95.0)) if len(speed_y) else 0.0,
    }


def lane_centers(vertices_t_v_3: np.ndarray) -> np.ndarray:
    return bbox_center_per_frame(vertices_t_v_3).astype(np.float64)


def lateral_wobble_against_reference(
    centers: np.ndarray,
    ref_centers: np.ndarray,
) -> dict[str, Any]:
    """
    Measure side-to-side wobble in the horizontal plane.

    Uses reference lane's overall movement direction as the forward axis.
    """
    if len(centers) <= 1 or len(ref_centers) <= 1:
        return {"lateral_std": 0.0, "lateral_sign_changes": 0, "forward_corr_with_ref": 0.0}

    c = centers[:, [0, 2]]
    r = ref_centers[:, [0, 2]]
    disp_ref = r[-1] - r[0]
    norm = np.linalg.norm(disp_ref)
    if norm < 1e-8:
        forward = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward = disp_ref / norm
    right = np.array([forward[1], -forward[0]], dtype=np.float64)

    c_rel = c - c[0]
    r_rel = r - r[0]
    lateral = c_rel @ right
    fwd = c_rel @ forward
    ref_fwd = r_rel @ forward

    signs = np.sign(lateral)
    sign_changes = int(np.sum((signs[1:] * signs[:-1]) < 0))
    corr = 0.0
    if np.std(fwd) > 1e-8 and np.std(ref_fwd) > 1e-8:
        corr = float(np.corrcoef(fwd, ref_fwd)[0, 1])

    return {
        "lateral_std": float(np.std(lateral)),
        "lateral_p95_abs": float(np.percentile(np.abs(lateral), 95.0)),
        "lateral_sign_changes": sign_changes,
        "forward_corr_with_ref": corr,
    }


def center_vertices(vertices_t_v_3: np.ndarray) -> np.ndarray:
    centers = bbox_center_per_frame(vertices_t_v_3)
    return vertices_t_v_3 - centers[:, None, :]


def apply_candidate_transform(vertices: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float64).copy()
    # Horizontal plane in this stage is X/Z, vertical is Y.
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
    if name == "mirror_x_then_yaw_180":
        out[..., 0] *= -1.0
        out[..., 0] *= -1.0
        out[..., 2] *= -1.0
        return out
    raise ValueError(f"Unknown transform candidate: {name}")


def shape_rmse_candidates(
    arp_vertices: np.ndarray,
    ref_vertices: np.ndarray,
) -> dict[str, Any]:
    """
    Compare centered ARP vs centered reference under simple global transforms.

    This is a direct per-vertex RMSE metric; it requires equal topology/order.
    """
    if arp_vertices.shape != ref_vertices.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch: arp={tuple(arp_vertices.shape)} ref={tuple(ref_vertices.shape)}",
        }

    a = center_vertices(arp_vertices).astype(np.float64)
    r = center_vertices(ref_vertices).astype(np.float64)

    candidates = ["identity", "yaw_180", "mirror_x", "mirror_z"]
    scores = {}
    for name in candidates:
        t = apply_candidate_transform(a, name)
        rmse_t = np.sqrt(np.mean((t - r) ** 2, axis=(1, 2)))
        scores[name] = {
            "rmse_mean": float(rmse_t.mean()),
            "rmse_p95": float(np.percentile(rmse_t, 95.0)),
        }

    best = min(scores, key=lambda k: scores[k]["rmse_mean"])
    return {
        "comparable": True,
        "best_candidate": best,
        "scores": scores,
    }


def load_lane_vertices(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    lanes = {}
    mapping = {
        "source": "source_vertices",
        "ours": "ours_vertices",
        "arp": "arp_vertices",
        "copyquat": "copyquat_vertices",
    }
    for lane, key in mapping.items():
        if key in npz.files:
            lanes[lane] = np.asarray(npz[key], dtype=np.float32)
    return lanes


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    payload = np.load(str(npz_path))
    lanes = load_lane_vertices(payload)
    if "arp" not in lanes:
        raise KeyError("NPZ has no arp_vertices.")
    if args.reference_lane not in lanes:
        raise KeyError(f"Reference lane '{args.reference_lane}' missing in NPZ.")

    report: dict[str, Any] = {
        "npz_path": str(npz_path),
        "available_lanes": sorted(lanes.keys()),
        "reference_lane": args.reference_lane,
        "lane_shapes": {k: list(v.shape) for k, v in lanes.items()},
        "trajectory_stats": {},
        "wobble_vs_reference": {},
    }

    centers = {k: lane_centers(v) for k, v in lanes.items()}
    ref_centers = centers[args.reference_lane]
    for lane, verts in lanes.items():
        report["trajectory_stats"][lane] = trajectory_stats(verts)
        report["wobble_vs_reference"][lane] = lateral_wobble_against_reference(
            centers[lane],
            ref_centers,
        )

    report["arp_shape_relation_to_reference"] = shape_rmse_candidates(
        lanes["arp"],
        lanes[args.reference_lane],
    )

    # Also compare ARP against "ours" when available, because both are target lanes.
    if "ours" in lanes:
        report["arp_shape_relation_to_ours"] = shape_rmse_candidates(
            lanes["arp"],
            lanes["ours"],
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[diag] wrote: {out_path}")


if __name__ == "__main__":
    main()
