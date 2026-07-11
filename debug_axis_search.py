#!/usr/bin/env python3
"""
Diagnose fixed axis transforms for SMAL33 preprocessed samples.

Given one or more *_seq.npy files, this script compares the current coordinate
semantics with all 48 axis permutations/sign flips. It is intended to answer:

  1. Is skel[0] bind identity pose in the same Y-up/+Z-forward space as seq?
  2. Is a source dataset using a fixed axis convention different from Planet Zoo?
  3. Which transform makes left-right across horizontal X-like and body forward
     agree with process_positions' forward = cross(across, Y)?

This script is read-only. It does not modify data.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PARENTS = np.array(
    [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 16,
        0, 18, 19, 20, 0, 22, 23, 24, 0, 26, 27, 28, 29, 30, 31,
    ],
    dtype=np.int64,
)

PAW_IDS = [10, 14, 21, 25]
AXIS_NAMES = ("x", "y", "z")


def norm(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


def offsets_to_global(offsets: np.ndarray) -> np.ndarray:
    out = np.zeros_like(offsets, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    for idx, parent in enumerate(PARENTS):
        if parent == -1:
            out[idx] = offsets[idx]
        else:
            out[idx] = out[parent] + offsets[idx]
    return out


def load_pair(seq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not seq_path.name.endswith("_seq.npy"):
        raise ValueError(f"Expected *_seq.npy path, got: {seq_path}")
    stem = seq_path.name[: -len("_seq.npy")]
    skel_path = seq_path.with_name(f"{stem}_skel.npy")
    if not skel_path.exists():
        raise FileNotFoundError(f"Missing paired skel file: {skel_path}")
    seq = np.load(seq_path)
    skel = np.load(skel_path)
    seq0 = seq[0, : 33 * 3].reshape(33, 3).astype(np.float64)
    bind_global = offsets_to_global(skel[0].reshape(33, 3))
    return seq0, bind_global


def transform_points(points: np.ndarray, perm: tuple[int, int, int], signs: tuple[int, int, int]) -> np.ndarray:
    signs_arr = np.asarray(signs, dtype=np.float64)
    return points[:, perm] * signs_arr[None, :]


def transform_name(perm: tuple[int, int, int], signs: tuple[int, int, int]) -> str:
    parts = []
    for out_axis, src_idx, sign in zip(AXIS_NAMES, perm, signs):
        prefix = "" if sign > 0 else "-"
        parts.append(f"new_{out_axis}={prefix}old_{AXIS_NAMES[src_idx]}")
    return ", ".join(parts)


def floor_axis_stats(joints: np.ndarray) -> tuple[str, dict[str, float]]:
    paws = joints[PAW_IDS]
    scores = {
        name: float(np.percentile(paws[:, idx], 95) - np.percentile(paws[:, idx], 5))
        for idx, name in enumerate(AXIS_NAMES)
    }
    return min(scores, key=scores.get), scores


def geom_features(joints: np.ndarray) -> dict[str, object]:
    hip = (joints[18] + joints[22]) / 2.0
    shoulder = (joints[8] + joints[12]) / 2.0
    body = norm(shoulder - hip)
    body_h = body.copy()
    body_h[1] = 0.0
    body_h = norm(body_h)

    across = norm((joints[18] - joints[22]) + (joints[8] - joints[12]))
    proc_forward = np.cross(across, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    proc_forward = norm(proc_forward)

    floor_axis, floor_scores = floor_axis_stats(joints)
    bbox = joints.max(axis=0) - joints.min(axis=0)
    dot_body_forward = float(np.dot(body_h, proc_forward))

    return {
        "bbox": bbox,
        "floor_axis": floor_axis,
        "floor_scores": floor_scores,
        "body": body,
        "body_h": body_h,
        "across": across,
        "process_forward": proc_forward,
        "dot_body_forward": dot_body_forward,
        "abs_body_y": abs(float(body[1])),
        "abs_across_y": abs(float(across[1])),
        "abs_across_x": abs(float(across[0])),
        "abs_forward_z": abs(float(proc_forward[2])),
    }


def passes_strict(feat: dict[str, object]) -> bool:
    return (
        feat["floor_axis"] == "y"
        and float(feat["process_forward"][2]) > 0.0
        and float(feat["abs_across_x"]) >= 0.75
        and float(feat["abs_across_y"]) <= 0.25
        and float(feat["dot_body_forward"]) >= 0.50
    )


def score_features(feat: dict[str, object]) -> float:
    # Strictly prefer Planet-Zoo-like semantics:
    # - paws are stable on Y (Y-up)
    # - left-right across is mostly X, not Y
    # - process forward points to +Z, not just +/-Z
    # - horizontal body direction agrees with process forward
    if feat["floor_axis"] != "y":
        return -100.0

    forward_z = float(feat["process_forward"][2])
    if forward_z <= 0.0:
        return -50.0 + forward_z

    return (
        4.0 * forward_z
        + 3.0 * float(feat["dot_body_forward"])
        + 2.5 * float(feat["abs_across_x"])
        - 4.0 * float(feat["abs_across_y"])
        - 0.5 * float(feat["abs_body_y"])
    )


def iter_axis_transforms():
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1, 1), repeat=3):
            yield perm, signs


def axis_key(perm: tuple[int, int, int], signs: tuple[int, int, int]) -> str:
    return transform_name(perm, signs)


def format_vec(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec), precision=4, suppress_small=True)


def print_features(label: str, feat: dict[str, object]) -> None:
    print(f"{label}")
    print(f"  floor_axis: {feat['floor_axis']} {feat['floor_scores']}")
    print(f"  bbox xyz: {format_vec(feat['bbox'])}")
    print(f"  body hip->shoulder: {format_vec(feat['body'])}")
    print(f"  across left-right:  {format_vec(feat['across'])}")
    print(f"  process_forward:    {format_vec(feat['process_forward'])}")
    print(f"  dot(body_h, process_forward): {feat['dot_body_forward']:.4f}")
    print(f"  |across_y|={feat['abs_across_y']:.4f}, |body_y|={feat['abs_body_y']:.4f}")


def analyze_pose(name: str, joints: np.ndarray, top_k: int) -> None:
    identity_feat = geom_features(joints)
    print(f"\n--- {name}: identity ---")
    print_features("", identity_feat)

    rows = []
    for perm, signs in iter_axis_transforms():
        transformed = transform_points(joints, perm, signs)
        feat = geom_features(transformed)
        rows.append((score_features(feat), perm, signs, feat))
    rows.sort(key=lambda row: row[0], reverse=True)

    print(f"\n--- {name}: top {top_k} axis transforms ---")
    for rank, (score, perm, signs, feat) in enumerate(rows[:top_k], start=1):
        ok = " strict=YES" if passes_strict(feat) else " strict=no"
        print(f"\n#{rank} score={score:.4f}{ok} :: {transform_name(perm, signs)}")
        print_features(" ", feat)


def best_transform(joints: np.ndarray) -> tuple[float, tuple[int, int, int], tuple[int, int, int], dict[str, object]]:
    rows = []
    for perm, signs in iter_axis_transforms():
        feat = geom_features(transform_points(joints, perm, signs))
        rows.append((score_features(feat), perm, signs, feat))
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows[0]


def expand_seq_paths(inputs: list[Path]) -> list[Path]:
    out = []
    for path in inputs:
        if path.is_dir():
            out.extend(sorted(path.rglob("*_seq.npy")))
        elif path.name.endswith("_seq.npy"):
            out.append(path)
        else:
            raise ValueError(f"Input must be a directory or *_seq.npy file: {path}")
    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for path in out:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def summarize(paths: list[Path], max_files: int | None = None) -> None:
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise SystemExit("No *_seq.npy files found.")

    counters = {"seq0": Counter(), "skel": Counter()}
    strict_counts = {"seq0": 0, "skel": 0}
    score_sums = defaultdict(float)
    examples = defaultdict(list)

    for seq_path in paths:
        seq0, bind_global = load_pair(seq_path)
        for label, joints in (("seq0", seq0), ("skel", bind_global)):
            score, perm, signs, feat = best_transform(joints)
            key = axis_key(perm, signs)
            counters[label][key] += 1
            score_sums[(label, key)] += score
            if passes_strict(feat):
                strict_counts[label] += 1
            if len(examples[(label, key)]) < 3:
                examples[(label, key)].append(str(seq_path))

    print(f"Analyzed {len(paths)} file(s).")
    for label in ("seq0", "skel"):
        print("\n" + "=" * 100)
        print(f"SUMMARY: {label}")
        print(f"strict pass ratio: {strict_counts[label]}/{len(paths)} = {strict_counts[label] / len(paths):.3f}")
        for rank, (key, count) in enumerate(counters[label].most_common(), start=1):
            avg_score = score_sums[(label, key)] / count
            print(f"\n#{rank} count={count} ratio={count / len(paths):.3f} avg_score={avg_score:.4f}")
            print(f"  {key}")
            for ex in examples[(label, key)]:
                print(f"  example: {ex}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search axis transforms for SMAL33 seq/skel semantic alignment."
    )
    parser.add_argument(
        "seq_paths",
        nargs="+",
        type=Path,
        help="One or more *_seq.npy files or directories containing *_seq.npy files.",
    )
    parser.add_argument("--top_k", type=int, default=8, help="Number of transforms to print.")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print aggregate best-transform counts across all inputs.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of expanded *_seq.npy files for quick sampling.",
    )
    args = parser.parse_args()

    seq_paths = expand_seq_paths(args.seq_paths)
    if args.summary_only:
        summarize(seq_paths, max_files=args.max_files)
        return

    if args.max_files is not None:
        seq_paths = seq_paths[: args.max_files]

    for seq_path in seq_paths:
        print("\n" + "=" * 100)
        print(f"FILE: {seq_path}")
        seq0, bind_global = load_pair(seq_path)
        analyze_pose("seq0 canonical local pose", seq0, args.top_k)
        analyze_pose("skel[0] bind identity global", bind_global, args.top_k)


if __name__ == "__main__":
    main()
