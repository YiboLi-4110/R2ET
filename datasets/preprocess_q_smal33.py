#!/usr/bin/env python3
"""
Generate SMAL33 local/global motion data from BVH files.

Defaults preserve the original Planet Zoo behavior. For shepherd data, use:
  --axis_transform shepherd_y_negz_x --forward_mode body
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smal33_motion_io import get_inp_from_bvh


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SMAL33 BVH files into train_q arrays.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("./datasets/Planet_Zoo_FBX-smal2/train_char"),
        help="Root directory containing per-character BVH subdirectories.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("./datasets/Planet_Zoo_FBX-smal2/train_q"),
        help="Output root for *_seq.npy, *_quat.npy, *_skel.npy.",
    )
    parser.add_argument(
        "--axis_transform",
        default="none",
        help="Optional coordinate transform. Use shepherd_y_negz_x for smal@shepherd.",
    )
    parser.add_argument(
        "--forward_mode",
        choices=["across", "body"],
        default="across",
        help="Canonical forward estimator. Planet Zoo default is across; shepherd should use body.",
    )
    parser.add_argument(
        "--post_axis_yaw_deg",
        type=float,
        default=0.0,
        help=(
            "Optional world yaw (degrees about +Y) applied after axis_transform. "
            "Default 0 preserves existing behavior. Use 90 for ARP cat_actions / "
            "batch2_dogs so they face +Z like dog_actions after shepherd_y_z_x."
        ),
    )
    parser.add_argument("--overwrite_existing", action="store_true")
    return parser.parse_args()


def iter_bvh_files(data_path):
    for folder in sorted(p for p in data_path.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for bvh_path in sorted(folder.glob("*.bvh")):
            yield folder.name, bvh_path


def save_motion(motion, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}_quat.npy", motion["quat"])
    np.save(output_dir / f"{stem}_seq.npy", motion["seq"])
    np.save(output_dir / f"{stem}_skel.npy", motion["skel"])


def main():
    args = parse_args()
    data_path = args.data_path.resolve()
    save_path = args.save_path.resolve()
    if not data_path.exists():
        raise SystemExit(f"data_path does not exist: {data_path}")

    print(f"Processing: {data_path}")
    print(f"Saving to:   {save_path}")
    print(
        f"axis_transform={args.axis_transform}, forward_mode={args.forward_mode}, "
        f"post_axis_yaw_deg={args.post_axis_yaw_deg}"
    )

    total = processed = skipped = failed = 0
    for folder, bvh_path in iter_bvh_files(data_path):
        total += 1
        out_dir = save_path / folder
        out_seq = out_dir / f"{bvh_path.stem}_seq.npy"
        if out_seq.exists() and not args.overwrite_existing:
            skipped += 1
            continue
        try:
            motion = get_inp_from_bvh(
                bvh_path,
                axis_transform=args.axis_transform,
                forward_mode=args.forward_mode,
                post_axis_yaw_deg=args.post_axis_yaw_deg,
            )
            if motion is None:
                skipped += 1
                print(f"SKIP (<=1 frame): {bvh_path}")
                continue
            save_motion(motion, out_dir, bvh_path.stem)
            processed += 1
            print(f"OK: {bvh_path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR: {bvh_path}: {exc!r}")

    print(f"Done. total={total} processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
