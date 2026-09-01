#!/usr/bin/env python3
"""
List BVH files under shepherd train_char whose frame count meets a threshold.

Scans each immediate subdirectory of data_path for *.bvh files, reads the
Frames: header (no full motion load), and writes qualifying paths to a txt file.

Example:
  python datasets/list_shepherd_bvh_min_frames.py
  python datasets/list_shepherd_bvh_min_frames.py \\
      --data_path ./datasets/shepherd/smal@shepherd/train_char \\
      --min_frames 120 \\
      --output ./datasets/shepherd/smal@shepherd/train_char_bvh_ge120.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FRAMES_RE = re.compile(r"^\s*Frames:\s+(\d+)\s*$")


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "datasets/shepherd/smal@shepherd/train_char"
    default_output = (
        repo_root / "datasets/shepherd/smal@shepherd/train_char_bvh_ge120.txt"
    )

    parser = argparse.ArgumentParser(
        description="List shepherd train_char BVH files with at least N frames."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=default_data,
        help="Root directory containing per-action subdirectories with BVH files.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=120,
        help="Minimum frame count (inclusive).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output txt file; one BVH path per line.",
    )
    parser.add_argument(
        "--path_style",
        choices=["relative", "absolute"],
        default="relative",
        help="How to write paths in the output file.",
    )
    return parser.parse_args()


def read_bvh_frame_count(bvh_path: Path) -> int:
    """Parse the Frames: line from a BVH header without loading motion data."""
    with bvh_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FRAMES_RE.match(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"Frames: header not found in {bvh_path}")


def iter_bvh_files(data_path: Path):
    for folder in sorted(p for p in data_path.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for bvh_path in sorted(folder.glob("*.bvh")):
            yield folder.name, bvh_path


def format_path(bvh_path: Path, path_style: str, base: Path) -> str:
    resolved = bvh_path.resolve()
    if path_style == "absolute":
        return str(resolved)
    try:
        return str(resolved.relative_to(base.resolve()))
    except ValueError:
        return str(resolved)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_path = args.data_path.resolve()
    output_path = args.output.resolve()

    if not data_path.exists():
        raise SystemExit(f"data_path does not exist: {data_path}")
    if args.min_frames < 1:
        raise SystemExit(f"min_frames must be >= 1, got {args.min_frames}")

    total_bvh = 0
    missing = 0
    errors: list[tuple[str, str]] = []
    kept: list[tuple[int, str, Path]] = []

    for folder_name, bvh_path in iter_bvh_files(data_path):
        total_bvh += 1
        try:
            num_frames = read_bvh_frame_count(bvh_path)
        except OSError as exc:
            errors.append((str(bvh_path), f"read error: {exc!r}"))
            continue
        except ValueError as exc:
            errors.append((str(bvh_path), str(exc)))
            continue

        if num_frames < args.min_frames:
            missing += 1
            continue

        kept.append((num_frames, folder_name, bvh_path))

    kept.sort(key=lambda item: (-item[0], item[1], item[2].name))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# data_path: {data_path}\n")
        handle.write(f"# min_frames: {args.min_frames}\n")
        handle.write(f"# total_bvh: {total_bvh}\n")
        handle.write(f"# matched: {len(kept)}\n")
        handle.write(f"# skipped_short: {missing}\n")
        handle.write(f"# errors: {len(errors)}\n")
        handle.write("# format: <num_frames>\\t<action_folder>\\t<bvh_path>\n")
        for num_frames, folder_name, bvh_path in kept:
            path_str = format_path(
                bvh_path,
                args.path_style,
                repo_root if args.path_style == "relative" else data_path,
            )
            handle.write(f"{num_frames}\t{folder_name}\t{path_str}\n")

    print(f"data_path:      {data_path}")
    print(f"min_frames:     {args.min_frames}")
    print(f"total_bvh:      {total_bvh}")
    print(f"matched (>=):   {len(kept)}")
    print(f"skipped_short:  {missing}")
    print(f"errors:         {len(errors)}")
    print(f"output:         {output_path}")

    if kept:
        print("\nMatched files:")
        for num_frames, folder_name, bvh_path in kept:
            print(f"  [{num_frames:4d}] {folder_name}/{bvh_path.name}")

    if errors:
        print("\nErrors:")
        for path_str, message in errors[:20]:
            print(f"  {path_str}: {message}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
