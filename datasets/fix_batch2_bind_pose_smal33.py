#!/usr/bin/env python3
"""
Fix batch2_dogs bind-pose orientation to match shepherd (+Z forward bind geometry).

Updates:
  - batch2_dogs_q: *_skel.npy, *_quat.npy (seq left unchanged; already +Z canonical)
  - batch2_dogs_shape: *.npz skeleton / rest_vertices / joint_shape widths

Run after preprocess_q_smal33.py and extract_shape_smal33.py for batch2_dogs:

  python datasets/fix_batch2_bind_pose_smal33.py \
    --q_path ./datasets/shepherd/batch2_dogs/batch2_dogs_q \
    --shape_path ./datasets/shepherd/batch2_dogs/batch2_dogs_shape \
    --forward_mode body

Verify:

  python datasets/compare_bind_pose_smal33.py \
    --shepherd_q ./datasets/shepherd/smal@shepherd/train_q \
    --batch2_q ./datasets/shepherd/batch2_dogs/batch2_dogs_q
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from bind_pose_canonicalize_smal33 import (  # noqa: E402
    bind_forward_dot,
    canonicalize_shape_npz_arrays,
    canonicalize_skel_quat_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Canonicalize batch2_dogs bind pose to shepherd +Z forward convention."
    )
    parser.add_argument(
        "--q_path",
        type=Path,
        required=True,
        help="batch2_dogs_q root containing *_skel.npy triplets.",
    )
    parser.add_argument(
        "--shape_path",
        type=Path,
        default=None,
        help="Optional batch2_dogs_shape root containing *.npz files.",
    )
    parser.add_argument(
        "--forward_mode",
        choices=["body", "across"],
        default="body",
        help="Forward estimator used to detect misaligned bind pose.",
    )
    parser.add_argument(
        "--dot_threshold",
        type=float,
        default=0.9,
        help="Apply correction when bind forward dot with +Z falls below this value.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report actions without writing files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def iter_q_triplets(q_path: Path):
    for skel_path in sorted(q_path.glob("*/*_skel.npy")):
        stem = skel_path.name[: -len("_skel.npy")]
        quat_path = skel_path.with_name(f"{stem}_quat.npy")
        seq_path = skel_path.with_name(f"{stem}_seq.npy")
        if not quat_path.exists() or not seq_path.exists():
            continue
        yield skel_path, quat_path, seq_path


def fix_q_root(q_path: Path, forward_mode: str, dot_threshold: float, dry_run: bool) -> dict:
    stats = {
        "total": 0,
        "fixed": 0,
        "already_ok": 0,
        "dots_before": [],
        "dots_after": [],
    }

    for skel_path, quat_path, seq_path in iter_q_triplets(q_path):
        stats["total"] += 1
        skel = np.load(skel_path)
        quat = np.load(quat_path)
        dot_before = bind_forward_dot(skel[0], forward_mode=forward_mode)
        stats["dots_before"].append(dot_before)

        skel_new, quat_new, changed = canonicalize_skel_quat_arrays(
            skel,
            quat,
            forward_mode=forward_mode,
            dot_threshold=dot_threshold,
        )
        dot_after = bind_forward_dot(skel_new[0], forward_mode=forward_mode)
        stats["dots_after"].append(dot_after)

        if changed:
            stats["fixed"] += 1
            if not dry_run:
                np.save(skel_path, skel_new)
                np.save(quat_path, quat_new)
        else:
            stats["already_ok"] += 1

    return stats


def fix_shape_root(shape_path: Path, forward_mode: str, dot_threshold: float, dry_run: bool) -> dict:
    stats = {
        "total": 0,
        "fixed": 0,
        "already_ok": 0,
        "dots_before": [],
        "dots_after": [],
    }
    if shape_path is None:
        return stats

    for npz_path in sorted(shape_path.glob("*.npz")):
        stats["total"] += 1
        payload = dict(np.load(npz_path))
        skeleton = payload["skeleton"]
        skel0 = skeleton[0] if skeleton.ndim == 3 else skeleton
        dot_before = bind_forward_dot(skel0, forward_mode=forward_mode)
        stats["dots_before"].append(dot_before)

        payload_new, changed = canonicalize_shape_npz_arrays(
            payload,
            forward_mode=forward_mode,
            dot_threshold=dot_threshold,
        )
        skel_after = payload_new["skeleton"]
        skel0_after = skel_after[0] if skel_after.ndim == 3 else skel_after
        dot_after = bind_forward_dot(skel0_after, forward_mode=forward_mode)
        stats["dots_after"].append(dot_after)

        if changed:
            stats["fixed"] += 1
            if not dry_run:
                np.savez(npz_path, **payload_new)
        else:
            stats["already_ok"] += 1

    return stats


def summarize_dots(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def main():
    args = parse_args()
    q_path = args.q_path.resolve()
    if not q_path.is_dir():
        raise SystemExit(f"q_path does not exist: {q_path}")

    shape_path = args.shape_path.resolve() if args.shape_path is not None else None
    if shape_path is not None and not shape_path.is_dir():
        raise SystemExit(f"shape_path does not exist: {shape_path}")

    q_stats = fix_q_root(q_path, args.forward_mode, args.dot_threshold, args.dry_run)
    shape_stats = (
        fix_shape_root(shape_path, args.forward_mode, args.dot_threshold, args.dry_run)
        if shape_path is not None
        else None
    )

    report = {
        "settings": {
            "q_path": str(q_path),
            "shape_path": str(shape_path) if shape_path is not None else None,
            "forward_mode": args.forward_mode,
            "dot_threshold": args.dot_threshold,
            "dry_run": args.dry_run,
        },
        "q": {
            **q_stats,
            "dots_before_summary": summarize_dots(q_stats["dots_before"]),
            "dots_after_summary": summarize_dots(q_stats["dots_after"]),
        },
    }
    if shape_stats is not None:
        report["shape"] = {
            **shape_stats,
            "dots_before_summary": summarize_dots(shape_stats["dots_before"]),
            "dots_after_summary": summarize_dots(shape_stats["dots_after"]),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
