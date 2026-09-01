#!/usr/bin/env python3
"""
Rotate SMAL33 BVH clips by a world yaw about +Y (root translation + root quat).

Use this to rewrite on-disk BVHs so that after ``shepherd_y_z_x`` they face +Z
like dog_actions, without changing fbx2bvh defaults.

Recommended for ARP-exported cat_actions / batch2_dogs:
  # Preview facing after axis transform (no writes):
  python ./rotate_bvh_yaw_smal33.py \\
    --data_path ./shepherd/cat_actions/train_char/Walk \\
    --report_only --axis_transform shepherd_y_z_x

  # Rewrite BVHs in place (back up first!), then re-preprocess with yaw=0:
  python ./rotate_bvh_yaw_smal33.py \\
    --data_path ./shepherd/cat_actions/train_char \\
    --yaw_deg 90 --recursive --overwrite_existing

Prefer the non-destructive preprocess flag when possible:
  python ./preprocess_q_smal33.py ... --post_axis_yaw_deg 90
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTSIDE_CODE = REPO_ROOT / "outside-code"
DATASETS_DIR = Path(__file__).resolve().parent
for path in (OUTSIDE_CODE, DATASETS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Animation  # noqa: E402
import BVH  # noqa: E402

from smal33_motion_io import (  # noqa: E402
    apply_axis_transform_anim,
    apply_world_yaw_anim,
    estimate_forward,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply optional world yaw to SMAL33 BVH files (default no-op)."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="BVH file, or directory of .bvh files / character subdirs.",
    )
    parser.add_argument(
        "--yaw_deg",
        type=float,
        default=90.0,
        help="Yaw degrees about +Y in BVH/native space before axis_transform. "
        "For cat_actions alignment after shepherd_y_z_x, try 90 "
        "(equivalent to preprocess --post_axis_yaw_deg 90 when applied in "
        "after_axis mode; see --space).",
    )
    parser.add_argument(
        "--space",
        choices=["native", "after_axis"],
        default="after_axis",
        help=(
            "Where to apply yaw. 'after_axis' (default) matches "
            "preprocess --post_axis_yaw_deg: axis_transform then yaw, then "
            "inverse axis back to native BVH storage. 'native' yaws in file "
            "space only (Blender ZX-plane tweak)."
        ),
    )
    parser.add_argument(
        "--axis_transform",
        default="shepherd_y_z_x",
        help="Used when --space after_axis (and for --report_only).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output root (mirrors relative layout). Default: overwrite in place "
        "when --overwrite_existing, else write alongside with suffix.",
    )
    parser.add_argument(
        "--suffix",
        default="_yaw",
        help="Filename stem suffix when not overwriting (default: _yaw).",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only print facing yaw after axis_transform (+ optional yaw); no writes.",
    )
    parser.add_argument(
        "--backup_dir",
        type=Path,
        default=None,
        help="If set with in-place overwrite, copy originals here first.",
    )
    return parser.parse_args()


def iter_bvh_files(data_path: Path, recursive: bool):
    if data_path.is_file():
        if data_path.suffix.lower() != ".bvh":
            raise SystemExit(f"Not a .bvh file: {data_path}")
        yield data_path
        return
    pattern = "**/*.bvh" if recursive else "*.bvh"
    files = sorted(data_path.glob(pattern))
    if not files and not recursive:
        # Character-root layout: one level of subdirs.
        for folder in sorted(p for p in data_path.iterdir() if p.is_dir()):
            files.extend(sorted(folder.glob("*.bvh")))
        files = sorted(files)
    if not files:
        raise SystemExit(f"No .bvh files under {data_path}")
    for path in files:
        yield path


def facing_report(anim) -> tuple[np.ndarray, float]:
    global_pos = Animation.positions_global(anim)
    forward = estimate_forward(global_pos, mode="body")
    forward = forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-8)
    mean_f = forward.mean(axis=0)
    mean_f[1] = 0.0
    mean_f = mean_f / (np.linalg.norm(mean_f) + 1e-8)
    yaw = float(np.arctan2(mean_f[0], mean_f[2]) * 180.0 / np.pi)
    return mean_f, yaw


def inverse_axis_transform_name(axis_transform: str | None) -> np.ndarray:
    from smal33_motion_io import axis_transform_matrix

    return np.linalg.inv(axis_transform_matrix(axis_transform))


def apply_matrix_to_anim(anim, matrix: np.ndarray):
    """Apply a linear basis change to positions/offsets/quats (same as axis_transform)."""
    from Quaternions import Quaternions

    anim.positions = np.einsum("ij,...j->...i", matrix, anim.positions)
    anim.offsets = np.einsum("ij,...j->...i", matrix, anim.offsets)
    rotations = Quaternions(anim.rotations.qs).transforms()
    transformed = np.einsum("ij,...jk,lk->...il", matrix, rotations, matrix)
    anim.rotations.qs = Quaternions.from_transforms(transformed).normalized().qs
    orients = Quaternions(anim.orients.qs).transforms()
    transformed_o = np.einsum("ij,...jk,lk->...il", matrix, orients, matrix)
    anim.orients.qs = Quaternions.from_transforms(transformed_o).normalized().qs
    return anim


def rotate_anim(anim, yaw_deg: float, space: str, axis_transform: str):
    if space == "native":
        return apply_world_yaw_anim(anim, yaw_deg)

    # after_axis: axis -> yaw -> inverse axis (store back in native BVH frame)
    anim = apply_axis_transform_anim(anim, axis_transform)
    anim = apply_world_yaw_anim(anim, yaw_deg)
    inv = inverse_axis_transform_name(axis_transform)
    anim = apply_matrix_to_anim(anim, inv)
    return anim


def resolve_out_path(
    src: Path,
    data_path: Path,
    output_path: Path | None,
    suffix: str,
    overwrite: bool,
) -> Path:
    if overwrite and output_path is None:
        return src
    if output_path is not None:
        try:
            rel = src.relative_to(data_path if data_path.is_dir() else data_path.parent)
        except ValueError:
            rel = Path(src.name)
        out = output_path / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


def main():
    args = parse_args()
    data_path = args.data_path.resolve()
    if not data_path.exists():
        raise SystemExit(f"data_path does not exist: {data_path}")

    files = list(iter_bvh_files(data_path, recursive=args.recursive))
    print(
        f"Found {len(files)} BVH(s). space={args.space} yaw_deg={args.yaw_deg} "
        f"axis_transform={args.axis_transform} report_only={args.report_only}"
    )

    for src in files:
        anim, names, ftime = BVH.load(str(src))

        # Report facing in after-axis space (comparable to dog_actions).
        probe = apply_axis_transform_anim(
            BVH.load(str(src))[0], args.axis_transform
        )
        f0, yaw0 = facing_report(probe)
        probe_yawed = apply_world_yaw_anim(
            apply_axis_transform_anim(BVH.load(str(src))[0], args.axis_transform),
            args.yaw_deg,
        )
        f1, yaw1 = facing_report(probe_yawed)
        print(
            f"{src}: after_axis yaw_to_+Z={yaw0:.1f} deg fwd={np.round(f0, 3)}; "
            f"with_yaw({args.yaw_deg}) -> {yaw1:.1f} deg fwd={np.round(f1, 3)}"
        )

        if args.report_only:
            continue

        out = resolve_out_path(
            src,
            data_path,
            args.output_path.resolve() if args.output_path else None,
            args.suffix,
            args.overwrite_existing,
        )
        if out.exists() and not args.overwrite_existing and out != src:
            print(f"SKIP exists: {out}")
            continue

        if args.backup_dir is not None and out == src:
            backup_root = args.backup_dir.resolve()
            try:
                rel = src.relative_to(data_path if data_path.is_dir() else data_path.parent)
            except ValueError:
                rel = Path(src.name)
            backup_path = backup_root / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not backup_path.exists():
                shutil.copy2(src, backup_path)

        anim, names, ftime = BVH.load(str(src))
        anim = rotate_anim(anim, args.yaw_deg, args.space, args.axis_transform)
        out.parent.mkdir(parents=True, exist_ok=True)
        BVH.save(str(out), anim, names, ftime)
        print(f"WROTE {out}")

    print("Done.")


if __name__ == "__main__":
    main()
