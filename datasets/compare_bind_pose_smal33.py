#!/usr/bin/env python3
"""
Step-B diagnostic: compare SMAL33 bind-pose geometry between datasets.

This script answers whether shepherd motion skeletons and batch2_dogs rest
skeletons share the same bind-pose semantics after preprocessing, beyond the
scalar +Z forward check from check_smal33_preprocess.py.

Typical usage (run from repo root):

  python datasets/compare_bind_pose_smal33.py \
    --shepherd_q ./datasets/shepherd/smal@shepherd/train_q \
    --batch2_q ./datasets/shepherd/batch2_dogs/batch2_dogs_q

Optional BVH round-trip check for a few pairs:

  python datasets/compare_bind_pose_smal33.py \
    --shepherd_q ./datasets/shepherd/smal@shepherd/train_q \
    --batch2_q ./datasets/shepherd/batch2_dogs/batch2_dogs_q \
    --shepherd_bvh_root ./datasets/shepherd/smal@shepherd/train_char \
    --batch2_bvh_root ./datasets/shepherd/batch2_dogs/batch2_dogs_char \
    --axis_transform shepherd_y_z_x \
    --shepherd_forward_mode body \
    --batch2_forward_mode body \
    --compare_bvh 5

Read-only: does not modify any data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.smal33_motion_io import (  # noqa: E402
    SMAL33_PARENTS,
    estimate_forward,
    get_inp_from_bvh,
)

PARENTS = SMAL33_PARENTS
JOINT_LEFT_SCAPULA, JOINT_RIGHT_SCAPULA = 8, 12
JOINT_LEFT_THIGH, JOINT_RIGHT_THIGH = 18, 22
JOINT_HEAD, JOINT_ROOT = 16, 0
JOINT_TAIL1 = 26


def norm(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    return vec / (np.linalg.norm(vec) + 1e-8)


def xz_unit(vec3: np.ndarray) -> np.ndarray:
    v = np.asarray(vec3, dtype=np.float64).copy()
    v[1] = 0.0
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float64)
    v /= n
    return v[[0, 2]]


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def offsets_to_global(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64).reshape(len(PARENTS), 3)
    out = np.zeros_like(offsets)
    for idx, parent in enumerate(PARENTS):
        if parent == -1:
            out[idx] = offsets[idx]
        else:
            out[idx] = out[parent] + offsets[idx]
    return out


def bind_features(skel_frame0: np.ndarray, forward_mode: str = "body") -> dict:
    """
    Compute bind-pose direction features from one skel frame (local offsets).
    """
    global_joints = offsets_to_global(skel_frame0)
    g = global_joints

    across = (g[JOINT_LEFT_THIGH] - g[JOINT_RIGHT_THIGH]) + (
        g[JOINT_LEFT_SCAPULA] - g[JOINT_RIGHT_SCAPULA]
    )
    across_u = norm(across)

    shoulder = 0.5 * (g[JOINT_LEFT_SCAPULA] + g[JOINT_RIGHT_SCAPULA])
    hip = 0.5 * (g[JOINT_LEFT_THIGH] + g[JOINT_RIGHT_THIGH])
    body = norm(shoulder - hip)
    body_h = norm(np.array([body[0], 0.0, body[2]], dtype=np.float64))

    proc_forward = norm(np.cross(across_u, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
    proc_forward_h = norm(np.array([proc_forward[0], 0.0, proc_forward[2]], dtype=np.float64))

    head = norm(g[JOINT_HEAD] - g[JOINT_ROOT])
    head_h = norm(np.array([head[0], 0.0, head[2]], dtype=np.float64))

    tail = norm(g[JOINT_TAIL1] - g[JOINT_ROOT])
    tail_h = norm(np.array([tail[0], 0.0, tail[2]], dtype=np.float64))

    canonical = estimate_forward(g[None, ...], mode=forward_mode)[0]
    canonical_h = norm(np.array([canonical[0], 0.0, canonical[2]], dtype=np.float64))

    bbox = g.max(axis=0) - g.min(axis=0)

    return {
        "across": across_u,
        "across_xz": xz_unit(across_u),
        "body": body,
        "body_xz": xz_unit(body),
        "process_forward": proc_forward,
        "process_forward_xz": xz_unit(proc_forward),
        "head_xz": xz_unit(head),
        "tail_xz": xz_unit(tail),
        "canonical_forward_xz": xz_unit(canonical),
        "dot_body_vs_process_forward": float(np.dot(body_h, proc_forward_h)),
        "dot_head_vs_process_forward": float(np.dot(head_h, proc_forward_h)),
        "dot_tail_vs_process_forward": float(np.dot(tail_h, proc_forward_h)),
        "dot_body_vs_canonical_forward": float(np.dot(body_h, canonical_h)),
        "bbox": bbox.tolist(),
        "bbox_dominant_axis": ("x", "y", "z")[int(np.argmax(bbox))],
    }


def summarize_angles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def list_skel_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(root.glob("*/*_skel.npy"))
    if not files:
        raise FileNotFoundError(f"No *_skel.npy found under: {root}")
    return files


def load_skel_frame0(skel_path: Path) -> np.ndarray:
    skel = np.load(skel_path)
    if skel.ndim != 3 or skel.shape[1:] != (len(PARENTS), 3):
        raise ValueError(f"Unexpected skel shape in {skel_path}: {skel.shape}")
    return skel[0]


def resolve_bvh_path(
    skel_path: Path,
    q_root: Path,
    bvh_root: Path | None,
) -> Path | None:
    if bvh_root is None:
        return None
    rel = skel_path.relative_to(q_root)
    stem = skel_path.name[: -len("_skel.npy")]
    candidate = bvh_root / rel.parent / f"{stem}.bvh"
    return candidate if candidate.exists() else None


def collect_dataset_features(
    q_root: Path,
    label: str,
    forward_mode: str,
    limit: int | None = None,
) -> list[dict]:
    rows = []
    skel_files = list_skel_files(q_root)
    if limit is not None:
        skel_files = skel_files[:limit]

    for skel_path in skel_files:
        skel0 = load_skel_frame0(skel_path)
        feat = bind_features(skel0, forward_mode=forward_mode)
        rel = skel_path.relative_to(q_root)
        rows.append(
            {
                "dataset": label,
                "character": rel.parts[0] if len(rel.parts) > 1 else rel.parent.name,
                "sequence": skel_path.name[: -len("_skel.npy")],
                "skel_path": str(skel_path),
                **feat,
            }
        )
    return rows


def mean_feature_vector(rows: list[dict], key: str) -> np.ndarray:
    vecs = [row[key] for row in rows if np.all(np.isfinite(row[key]))]
    if not vecs:
        return np.array([np.nan, np.nan], dtype=np.float64)
    arr = np.stack(vecs, axis=0)
    m = arr.mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-8)


def compare_groups(
    shepherd_rows: list[dict],
    batch2_rows: list[dict],
) -> dict:
    keys = (
        "process_forward_xz",
        "body_xz",
        "head_xz",
        "tail_xz",
        "across_xz",
        "canonical_forward_xz",
    )
    shepherd_mean = {k: mean_feature_vector(shepherd_rows, k) for k in keys}
    batch2_mean = {k: mean_feature_vector(batch2_rows, k) for k in keys}

    pairwise = {k: [] for k in keys}
    for srow in shepherd_rows:
        for brow in batch2_rows:
            for k in keys:
                pairwise[k].append(angle_deg(srow[k], brow[k]))

    shepherd_vs_batch2_mean = {
        k: angle_deg(shepherd_mean[k], batch2_mean[k]) for k in keys
    }

    # Most relevant for the observed ~90 deg cross-retarget issue.
    primary_angles = [
        angle_deg(srow["process_forward_xz"], brow["process_forward_xz"])
        for srow in shepherd_rows
        for brow in batch2_rows
    ]

    near_90 = [
        a for a in primary_angles if np.isfinite(a) and 75.0 <= a <= 105.0
    ]
    near_0 = [
        a for a in primary_angles if np.isfinite(a) and a <= 15.0
    ]

    return {
        "shepherd_mean_xz": {k: shepherd_mean[k].tolist() for k in keys},
        "batch2_mean_xz": {k: batch2_mean[k].tolist() for k in keys},
        "shepherd_vs_batch2_mean_angle_deg": shepherd_vs_batch2_mean,
        "pairwise_angle_deg": {k: summarize_angles(pairwise[k]) for k in keys},
        "process_forward_pairwise_near_90deg_ratio": (
            float(len(near_90) / max(len(primary_angles), 1))
        ),
        "process_forward_pairwise_near_0deg_ratio": (
            float(len(near_0) / max(len(primary_angles), 1))
        ),
    }


def compare_bvh_roundtrip(
    q_root: Path,
    bvh_root: Path,
    forward_mode: str,
    axis_transform: str,
    limit: int,
) -> list[dict]:
    rows = []
    skel_files = list_skel_files(q_root)[:limit]
    for skel_path in skel_files:
        bvh_path = resolve_bvh_path(skel_path, q_root, bvh_root)
        if bvh_path is None:
            rows.append(
                {
                    "skel_path": str(skel_path),
                    "bvh_path": None,
                    "status": "missing_bvh",
                }
            )
            continue

        npy_skel0 = load_skel_frame0(skel_path)
        motion = get_inp_from_bvh(
            str(bvh_path),
            axis_transform=axis_transform,
            forward_mode=forward_mode,
        )
        if motion is None:
            rows.append(
                {
                    "skel_path": str(skel_path),
                    "bvh_path": str(bvh_path),
                    "status": "bvh_parse_failed",
                }
            )
            continue

        bvh_skel0 = motion["skel"][0]
        npy_feat = bind_features(npy_skel0, forward_mode=forward_mode)
        bvh_feat = bind_features(bvh_skel0, forward_mode=forward_mode)
        rows.append(
            {
                "skel_path": str(skel_path),
                "bvh_path": str(bvh_path),
                "status": "ok",
                "skel_max_abs_diff": float(
                    np.max(np.abs(npy_skel0.astype(np.float64) - bvh_skel0.astype(np.float64)))
                ),
                "process_forward_angle_deg": angle_deg(
                    npy_feat["process_forward_xz"], bvh_feat["process_forward_xz"]
                ),
            }
        )
    return rows


def compact_row(row: dict) -> dict:
    return {
        "dataset": row["dataset"],
        "character": row["character"],
        "sequence": row["sequence"],
        "process_forward_xz": [float(x) for x in row["process_forward_xz"]],
        "body_xz": [float(x) for x in row["body_xz"]],
        "head_xz": [float(x) for x in row["head_xz"]],
        "tail_xz": [float(x) for x in row["tail_xz"]],
        "across_xz": [float(x) for x in row["across_xz"]],
        "canonical_forward_xz": [float(x) for x in row["canonical_forward_xz"]],
        "dot_body_vs_process_forward": row["dot_body_vs_process_forward"],
        "dot_head_vs_process_forward": row["dot_head_vs_process_forward"],
        "bbox_dominant_axis": row["bbox_dominant_axis"],
    }


def print_report(summary: dict, sample_rows: int = 5) -> None:
    print("=== Bind-pose geometry comparison (Step B) ===")
    print(json.dumps(summary["counts"], indent=2, ensure_ascii=False))
    print()
    print("Mean XZ directions:")
    print(json.dumps(summary["comparison"]["shepherd_mean_xz"], indent=2))
    print(json.dumps(summary["comparison"]["batch2_mean_xz"], indent=2))
    print()
    print("Shepherd mean vs batch2 mean angle (deg):")
    print(json.dumps(summary["comparison"]["shepherd_vs_batch2_mean_angle_deg"], indent=2))
    print()
    print("Pairwise angle stats (deg):")
    print(json.dumps(summary["comparison"]["pairwise_angle_deg"], indent=2))
    print()
    print(
        "process_forward pairwise near 90 deg ratio:",
        summary["comparison"]["process_forward_pairwise_near_90deg_ratio"],
    )
    print(
        "process_forward pairwise near 0 deg ratio:",
        summary["comparison"]["process_forward_pairwise_near_0deg_ratio"],
    )

    if summary.get("bvh_roundtrip"):
        print()
        print("BVH round-trip checks:")
        print(json.dumps(summary["bvh_roundtrip"], indent=2, ensure_ascii=False))

    print()
    print(f"Sample shepherd rows (first {sample_rows}):")
    for row in summary["samples"]["shepherd"][:sample_rows]:
        print(json.dumps(row, ensure_ascii=False))

    print()
    print(f"Sample batch2 rows (first {sample_rows}):")
    for row in summary["samples"]["batch2"][:sample_rows]:
        print(json.dumps(row, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SMAL33 bind-pose geometry between shepherd and batch2_dogs."
    )
    parser.add_argument(
        "--shepherd_q",
        type=Path,
        required=True,
        help="Path to smal@shepherd train_q root.",
    )
    parser.add_argument(
        "--batch2_q",
        type=Path,
        required=True,
        help="Path to batch2_dogs_q root.",
    )
    parser.add_argument(
        "--shepherd_forward_mode",
        type=str,
        default="body",
        choices=["body", "across"],
        help="Forward mode used when interpreting shepherd skel.",
    )
    parser.add_argument(
        "--batch2_forward_mode",
        type=str,
        default="body",
        choices=["body", "across"],
        help="Forward mode used when interpreting batch2 skel.",
    )
    parser.add_argument(
        "--limit_shepherd",
        type=int,
        default=None,
        help="Optional cap on number of shepherd skel files.",
    )
    parser.add_argument(
        "--limit_batch2",
        type=int,
        default=None,
        help="Optional cap on number of batch2 skel files.",
    )
    parser.add_argument(
        "--shepherd_bvh_root",
        type=Path,
        default=None,
        help="Optional BVH root for shepherd round-trip checks.",
    )
    parser.add_argument(
        "--batch2_bvh_root",
        type=Path,
        default=None,
        help="Optional BVH root for batch2 round-trip checks.",
    )
    parser.add_argument(
        "--axis_transform",
        type=str,
        default="shepherd_y_z_x",
        help="Axis transform for optional BVH round-trip.",
    )
    parser.add_argument(
        "--compare_bvh",
        type=int,
        default=0,
        help="If >0, compare this many npy-vs-bvh skel pairs per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--sample_rows",
        type=int,
        default=5,
        help="Number of sample rows to print per dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    shepherd_rows = collect_dataset_features(
        args.shepherd_q,
        label="shepherd",
        forward_mode=args.shepherd_forward_mode,
        limit=args.limit_shepherd,
    )
    batch2_rows = collect_dataset_features(
        args.batch2_q,
        label="batch2_dogs",
        forward_mode=args.batch2_forward_mode,
        limit=args.limit_batch2,
    )

    comparison = compare_groups(shepherd_rows, batch2_rows)

    summary = {
        "counts": {
            "shepherd": len(shepherd_rows),
            "batch2_dogs": len(batch2_rows),
            "pairwise": len(shepherd_rows) * len(batch2_rows),
        },
        "settings": {
            "shepherd_q": str(args.shepherd_q),
            "batch2_q": str(args.batch2_q),
            "shepherd_forward_mode": args.shepherd_forward_mode,
            "batch2_forward_mode": args.batch2_forward_mode,
        },
        "comparison": comparison,
        "samples": {
            "shepherd": [compact_row(r) for r in shepherd_rows[: args.sample_rows]],
            "batch2": [compact_row(r) for r in batch2_rows[: args.sample_rows]],
        },
    }

    if args.compare_bvh > 0:
        summary["bvh_roundtrip"] = {}
        if args.shepherd_bvh_root is not None:
            summary["bvh_roundtrip"]["shepherd"] = compare_bvh_roundtrip(
                args.shepherd_q,
                args.shepherd_bvh_root,
                args.shepherd_forward_mode,
                args.axis_transform,
                args.compare_bvh,
            )
        if args.batch2_bvh_root is not None:
            summary["bvh_roundtrip"]["batch2_dogs"] = compare_bvh_roundtrip(
                args.batch2_q,
                args.batch2_bvh_root,
                args.batch2_forward_mode,
                args.axis_transform,
                args.compare_bvh,
            )

    print_report(summary, sample_rows=args.sample_rows)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print()
        print(f"Wrote full report to: {args.output}")


if __name__ == "__main__":
    main()
