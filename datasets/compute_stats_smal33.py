#!/usr/bin/env python3
"""
Compute motion / quaternion / shape normalization statistics for SMAL33 pet data.

The logic mirrors datasets/train_feeder_r2et.py::Feeder.load_data(), but runs as a
standalone preprocessing step so stats can be prepared before skeleton-aware training.

Expected inputs:
  - train_q/: *_seq.npy, *_skel.npy, *_quat.npy produced by preprocess_q_smal33.py
  - train_shape/: *.npz produced by extract_shape_smal33.py

Default outputs (under stats/):
  - smal33_local_motion_mean.npy   shape (1, J, 3)
  - smal33_local_motion_std.npy
  - smal33_global_motion_mean.npy  shape (1, 4)  # root vel xyz + rvelocity
  - smal33_global_motion_std.npy
  - smal33_quat_mean.npy           shape (1, J, 4)
  - smal33_quat_std.npy
  - smal33_shape_mean_xyz.npy      shape (3,)  # mean over all joints/characters per axis
  - smal33_shape_std_xyz.npy       shape (3,)
  - smal33_stats_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "Planet_Zoo_FBX-smal2"
NUM_JOINTS = 33
GLOBAL_DIM = 4
SEQ_TAIL_DIM = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SMAL33 motion/shape stats for skeleton-aware training."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "train_q",
        help="Directory with per-character folders of *_seq/_skel/_quat .npy files.",
    )
    parser.add_argument(
        "--shape_path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "train_shape",
        help="Primary directory with *.npz shape files (e.g. smal@shepherd/train_shape).",
    )
    parser.add_argument(
        "--extra_shape_paths",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Additional shape directories merged into shape stats "
            "(e.g. batch2_dogs/batch2_dogs_shape for target skeletons)."
        ),
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=DEFAULT_DATA_ROOT / "stats",
        help="Directory to write computed statistics.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=32,
        help="Skip sequences shorter than this many frames (recommended: 32).",
    )
    parser.add_argument(
        "--use_legacy_names",
        action="store_true",
        help="Also save mixamo_*.npy aliases for drop-in use with unmodified feeders.",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional path for summary JSON. Defaults to <stats_path>/smal33_stats_summary.json.",
    )
    return parser.parse_args()


def list_character_dirs(data_path: Path):
    return sorted(
        p
        for p in data_path.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def load_motion_samples(data_path: Path, min_frames: int):
    samples = []
    skipped_short = 0
    skipped_missing = 0

    for char_dir in list_character_dirs(data_path):
        seq_files = sorted(char_dir.glob("*_seq.npy"))
        for seq_path in seq_files:
            stem = seq_path.name[: -len("_seq.npy")]
            skel_path = char_dir / f"{stem}_skel.npy"
            quat_path = char_dir / f"{stem}_quat.npy"

            if not skel_path.exists() or not quat_path.exists():
                skipped_missing += 1
                continue

            sequence = np.load(seq_path)
            if sequence.ndim != 2 or sequence.shape[1] < SEQ_TAIL_DIM:
                raise ValueError(
                    f"Unexpected seq shape {sequence.shape} in {seq_path}. "
                    f"Expected (T, J*3+{SEQ_TAIL_DIM})."
                )
            if sequence.shape[0] < min_frames:
                skipped_short += 1
                continue

            positions = np.load(skel_path)
            quat = np.load(quat_path)

            local = np.reshape(sequence[:, :-SEQ_TAIL_DIM], (sequence.shape[0], -1, 3))
            global_offset = sequence[:, -SEQ_TAIL_DIM:-4]
            positions = positions.copy()
            positions[:, 0, :] = local[:, 0, :]

            if local.shape[1] != NUM_JOINTS:
                raise ValueError(
                    f"Expected {NUM_JOINTS} joints, got {local.shape[1]} in {seq_path}."
                )
            if quat.shape[1] != NUM_JOINTS:
                raise ValueError(
                    f"Expected {NUM_JOINTS} quat joints, got {quat.shape[1]} in {quat_path}."
                )
            if global_offset.shape[1] != GLOBAL_DIM:
                raise ValueError(
                    f"Expected global dim {GLOBAL_DIM}, got {global_offset.shape[1]} in {seq_path}."
                )

            samples.append(
                {
                    "character": char_dir.name,
                    "sequence": stem,
                    "local": local,
                    "global": global_offset,
                    "skel": positions,
                    "quat": quat,
                }
            )

    return samples, {
        "num_sequences": len(samples),
        "skipped_short": skipped_short,
        "skipped_missing": skipped_missing,
    }


def iter_shape_files(shape_paths):
    seen = set()
    for shape_path in shape_paths:
        shape_path = Path(shape_path)
        if not shape_path.exists():
            raise FileNotFoundError(f"shape path does not exist: {shape_path}")
        for shape_file in sorted(shape_path.glob("*.npz")):
            if shape_file.name.startswith("."):
                continue
            key = shape_file.stem
            if key in seen:
                continue
            seen.add(key)
            yield shape_file


def load_shape_stats(shape_paths):
    shape_paths = [Path(p) for p in shape_paths]
    shape_files = list(iter_shape_files(shape_paths))
    if not shape_files:
        roots = ", ".join(str(p) for p in shape_paths)
        raise FileNotFoundError(f"No .npz shape files found under: {roots}")

    shape_vectors = []
    characters = []
    for shape_file in shape_files:
        payload = np.load(shape_file)
        full_width = payload["full_width"].astype(np.single)
        joint_shape = payload["joint_shape"].astype(np.single)
        if joint_shape.shape[0] != NUM_JOINTS:
            raise ValueError(
                f"Expected {NUM_JOINTS} joints in {shape_file}, got {joint_shape.shape[0]}."
            )
        shape_vector = np.divide(joint_shape, full_width[None, :])
        shape_vectors.append(shape_vector)
        characters.append(shape_file.stem)

    shape_array = np.concatenate(shape_vectors, axis=0)
    return {
        "characters": characters,
        "shape_mean": shape_array.mean(axis=0),
        "shape_std": shape_array.std(axis=0),
        "shape_sources": [str(p.parent) for p in shape_files],
    }


def compute_motion_stats(samples):
    if not samples:
        raise RuntimeError("No motion sequences passed the filters; cannot compute stats.")

    train_local = [item["local"] for item in samples]
    train_global = [item["global"] for item in samples]
    train_skel = [item["skel"] for item in samples]
    all_quats = [item["quat"] for item in samples]
    t_skel = [item["skel"][0:1] for item in samples]

    allframes_n_skel = np.concatenate(train_local + t_skel)
    allframes_quat = np.concatenate(all_quats)
    allframes_global = np.concatenate(train_global)

    local_mean = allframes_n_skel.mean(axis=0)[None, :]
    local_std = allframes_n_skel.std(axis=0)[None, :]
    global_mean = allframes_global.mean(axis=0)[None, :]
    global_std = allframes_global.std(axis=0)[None, :]
    quat_mean = allframes_quat.mean(axis=0)[None, :]
    quat_std = allframes_quat.std(axis=0)[None, :]

    total_frames = int(sum(item["local"].shape[0] for item in samples))
    return {
        "local_mean": local_mean.astype(np.float32),
        "local_std": local_std.astype(np.float32),
        "global_mean": global_mean.astype(np.float32),
        "global_std": global_std.astype(np.float32),
        "quat_mean": quat_mean.astype(np.float32),
        "quat_std": quat_std.astype(np.float32),
        "num_sequences": len(samples),
        "num_frames": total_frames,
        "num_characters": len({item["character"] for item in samples}),
    }


def save_stats(stats_path: Path, motion_stats, shape_stats, use_legacy_names: bool):
    stats_path.mkdir(parents=True, exist_ok=True)

    files = {
        "smal33_local_motion_mean.npy": motion_stats["local_mean"],
        "smal33_local_motion_std.npy": motion_stats["local_std"],
        "smal33_global_motion_mean.npy": motion_stats["global_mean"],
        "smal33_global_motion_std.npy": motion_stats["global_std"],
        "smal33_quat_mean.npy": motion_stats["quat_mean"],
        "smal33_quat_std.npy": motion_stats["quat_std"],
        "smal33_shape_mean_xyz.npy": shape_stats["shape_mean"].astype(np.float32),
        "smal33_shape_std_xyz.npy": shape_stats["shape_std"].astype(np.float32),
    }

    legacy_map = {
        "mixamo_local_motion_mean.npy": "smal33_local_motion_mean.npy",
        "mixamo_local_motion_std.npy": "smal33_local_motion_std.npy",
        "mixamo_global_motion_mean.npy": "smal33_global_motion_mean.npy",
        "mixamo_global_motion_std.npy": "smal33_global_motion_std.npy",
        "mixamo_quat_mean.npy": "smal33_quat_mean.npy",
        "mixamo_quat_std.npy": "smal33_quat_std.npy",
        "mixamo_shape_mean_xyz.npy": "smal33_shape_mean_xyz.npy",
        "mixamo_shape_std_xyz.npy": "smal33_shape_std_xyz.npy",
    }

    saved_paths = []
    for filename, array in files.items():
        out_path = stats_path / filename
        np.save(out_path, array)
        saved_paths.append(str(out_path))

    if use_legacy_names:
        for legacy_name, source_name in legacy_map.items():
            out_path = stats_path / legacy_name
            np.save(out_path, files[source_name])
            saved_paths.append(str(out_path))

    return saved_paths


def build_summary(args, load_info, motion_stats, shape_stats, saved_paths):
    zero_local_std = int((motion_stats["local_std"] == 0).sum())
    zero_global_std = int((motion_stats["global_std"] == 0).sum())
    zero_quat_std = int((motion_stats["quat_std"] == 0).sum())
    zero_shape_std = int((shape_stats["shape_std"] == 0).sum())

    return {
        "data_path": str(args.data_path.resolve()),
        "shape_path": str(args.shape_path.resolve()),
        "extra_shape_paths": [str(p.resolve()) for p in args.extra_shape_paths],
        "stats_path": str(args.stats_path.resolve()),
        "min_frames": args.min_frames,
        "num_joints": NUM_JOINTS,
        "motion": {
            "num_sequences_used": motion_stats["num_sequences"],
            "num_frames_used": motion_stats["num_frames"],
            "num_characters_used": motion_stats["num_characters"],
            "skipped_short": load_info["skipped_short"],
            "skipped_missing_sidecars": load_info["skipped_missing"],
            "shapes": {
                "local_mean": list(motion_stats["local_mean"].shape),
                "local_std": list(motion_stats["local_std"].shape),
                "global_mean": list(motion_stats["global_mean"].shape),
                "global_std": list(motion_stats["global_std"].shape),
                "quat_mean": list(motion_stats["quat_mean"].shape),
                "quat_std": list(motion_stats["quat_std"].shape),
            },
            "zero_std_counts": {
                "local": zero_local_std,
                "global": zero_global_std,
                "quat": zero_quat_std,
            },
        },
        "shape": {
            "num_characters": len(shape_stats["characters"]),
            "characters": shape_stats["characters"],
            "shape_sources": shape_stats.get("shape_sources", []),
            "shape_mean_shape": list(shape_stats["shape_mean"].shape),
            "shape_std_shape": list(shape_stats["shape_std"].shape),
            "zero_std_count": zero_shape_std,
        },
        "saved_files": saved_paths,
        "notes": [
            "Motion stats follow datasets/train_feeder_r2et.py aggregation rules.",
            "Local stats include all motion frames plus the first T-pose frame of each sequence.",
            "Global stats use seq[:, -8:-4] (root velocity xyz + rvelocity).",
            "Training feeders still expect mixamo_* filenames unless you pass --use_legacy_names "
            "or update the loader paths.",
        ],
    }


def main():
    args = parse_args()
    if not args.data_path.exists():
        raise SystemExit(f"data_path does not exist: {args.data_path}")
    if not args.shape_path.exists():
        raise SystemExit(f"shape_path does not exist: {args.shape_path}")

    samples, load_info = load_motion_samples(args.data_path, args.min_frames)

    if not samples:
        raise SystemExit(
            "No valid motion sequences found. "
            f"Checked {args.data_path} with min_frames={args.min_frames}."
        )

    motion_stats = compute_motion_stats(samples)
    shape_roots = [args.shape_path, *args.extra_shape_paths]
    shape_stats = load_shape_stats(shape_roots)
    saved_paths = save_stats(
        args.stats_path, motion_stats, shape_stats, args.use_legacy_names
    )

    summary = build_summary(args, load_info, motion_stats, shape_stats, saved_paths)
    summary_path = args.summary_json or (args.stats_path / "smal33_stats_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Used sequences: {motion_stats['num_sequences']}")
    print(f"Used frames: {motion_stats['num_frames']}")
    print(f"Skipped short (<{args.min_frames}): {load_info['skipped_short']}")
    print(f"Skipped missing sidecars: {load_info['skipped_missing']}")
    print(f"Shape characters: {len(shape_stats['characters'])}")
    print(f"Stats saved to: {args.stats_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
