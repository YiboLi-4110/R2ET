#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run with Blender, for example:\n"
        "  blender -b -P ./datasets/check_fbx_frame_stats.py -- "
        "--data_path ./datasets/Planet_Zoo_FBX-smal2\n"
    ) from exc


def parse_args():
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(
        description="Collect frame-count and duration statistics for FBX motion files."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("./Planet_Zoo_FBX-smal2/train_char"),
        help="Root directory containing .fbx files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.fbx",
        help="Filename glob for recursive search under data_path.",
    )
    parser.add_argument(
        "--min_frame_end",
        type=int,
        default=60,
        help="Simulate the current fbx2bvh.py rule: frame_end = max(min_frame_end, raw_frame_end).",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10,
        help="How many short/padded examples to print.",
    )
    parser.add_argument(
        "--cleanup_mode",
        type=str,
        choices=["purge", "factory"],
        default="purge",
        help="How to clean the Blender scene between files. 'purge' is usually much faster.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many Blender worker processes to run in parallel. Use 1 for single-process mode.",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Value passed to Blender '-t'. Use 1 to minimize oversubscription, 0 for Blender auto.",
    )
    parser.add_argument(
        "--max_cpu_cores",
        type=int,
        default=0,
        help="Approximate total CPU-core budget across all workers. 0 disables this cap.",
    )
    parser.add_argument(
        "--blender_binary",
        type=Path,
        default=None,
        help="Optional explicit Blender binary path for parallel worker mode.",
    )
    parser.add_argument(
        "--worker_manifest",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker_count",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def reset_blender_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def purge_blender_scene():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    datablock_names = [
        "actions",
        "armatures",
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "curves",
        "grease_pencils",
        "node_groups",
    ]
    datablocks = [getattr(bpy.data, name) for name in datablock_names if hasattr(bpy.data, name)]
    for collection in datablocks:
        for datablock in list(collection):
            collection.remove(datablock, do_unlink=True)

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world)

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
    except Exception:
        pass


def cleanup_blender_scene(cleanup_mode):
    if cleanup_mode == "factory":
        reset_blender_scene()
    else:
        purge_blender_scene()


def percentile_dict(values, percentiles):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p:02d}": float(np.percentile(arr, p)) for p in percentiles}


def stats_dict(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    result = {
        "count": int(arr.shape[0]),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }
    result.update(percentile_dict(values, [1, 5, 10, 25, 75, 90, 95, 99]))
    return result


def bucket_counts(values):
    buckets = {
        "1-15": 0,
        "16-31": 0,
        "32-47": 0,
        "48-59": 0,
        "60-89": 0,
        "90-119": 0,
        "120-239": 0,
        "240+": 0,
    }
    for value in values:
        if value <= 15:
            buckets["1-15"] += 1
        elif value <= 31:
            buckets["16-31"] += 1
        elif value <= 47:
            buckets["32-47"] += 1
        elif value <= 59:
            buckets["48-59"] += 1
        elif value <= 89:
            buckets["60-89"] += 1
        elif value <= 119:
            buckets["90-119"] += 1
        elif value <= 239:
            buckets["120-239"] += 1
        else:
            buckets["240+"] += 1
    return buckets


def summarize_thresholds(values, thresholds):
    total = len(values)
    summary = {}
    for threshold in thresholds:
        count = sum(v < threshold for v in values)
        summary[f"lt_{threshold}"] = {
            "count": int(count),
            "ratio": float(count / total) if total else 0.0,
        }
    return summary


def safe_relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_fbx_paths(data_root, glob_pattern, worker_manifest=None, worker_id=0, worker_count=1):
    if worker_manifest is None:
        fbx_paths = sorted(data_root.rglob(glob_pattern))
    else:
        lines = worker_manifest.read_text(encoding="utf-8").splitlines()
        fbx_paths = [Path(line) for line in lines if line.strip()]

    if worker_count <= 1:
        return fbx_paths
    return [path for index, path in enumerate(fbx_paths) if index % worker_count == worker_id]


def collect_fbx_stats(fbx_path, data_root, min_frame_end, cleanup_mode):
    cleanup_blender_scene(cleanup_mode)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

    actions = list(bpy.data.actions)
    if not actions:
        return {
            "path": safe_relative(fbx_path, data_root),
            "status": "error",
            "error": "No animation action found after FBX import.",
        }

    frame_starts = [int(action.frame_range[0]) for action in actions]
    frame_ends = [int(action.frame_range[1]) for action in actions]

    raw_frame_start = min(frame_starts)
    raw_frame_end = max(frame_ends)
    raw_frame_count = raw_frame_end - raw_frame_start + 1
    exported_frame_end = max(min_frame_end, raw_frame_end)
    exported_frame_count = exported_frame_end - raw_frame_start + 1
    padded_tail_frames = exported_frame_count - raw_frame_count

    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    duration_seconds = raw_frame_count / fps if fps > 0 else None

    return {
        "path": safe_relative(fbx_path, data_root),
        "status": "ok",
        "action_count": len(actions),
        "action_names": [action.name for action in actions],
        "raw_frame_start": int(raw_frame_start),
        "raw_frame_end": int(raw_frame_end),
        "raw_frame_count": int(raw_frame_count),
        "fps": float(fps),
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "exported_frame_end_if_current_script": int(exported_frame_end),
        "exported_frame_count_if_current_script": int(exported_frame_count),
        "padded_tail_frames_if_current_script": int(padded_tail_frames),
        "would_be_padded_by_current_script": bool(padded_tail_frames > 0),
    }


def build_summary(records, min_frame_end):
    ok_records = [record for record in records if record["status"] == "ok"]
    error_records = [record for record in records if record["status"] != "ok"]

    raw_frame_counts = [record["raw_frame_count"] for record in ok_records]
    durations = [record["duration_seconds"] for record in ok_records if record["duration_seconds"] is not None]
    padded_records = [
        record for record in ok_records if record["would_be_padded_by_current_script"]
    ]
    padded_tail_frames = [
        record["padded_tail_frames_if_current_script"] for record in padded_records
    ]

    summary = {
        "total_files": len(records),
        "ok_files": len(ok_records),
        "error_files": len(error_records),
        "min_frame_end_rule_simulated": int(min_frame_end),
        "raw_frame_count_stats": stats_dict(raw_frame_counts),
        "duration_seconds_stats": stats_dict(durations),
        "raw_frame_count_buckets": bucket_counts(raw_frame_counts),
        "raw_frame_count_thresholds": summarize_thresholds(
            raw_frame_counts, [32, 48, 60, 90, 120]
        ),
        "files_padded_by_current_script": {
            "count": len(padded_records),
            "ratio": float(len(padded_records) / len(ok_records)) if ok_records else 0.0,
            "total_added_tail_frames": int(sum(padded_tail_frames)),
            "added_tail_frame_stats": stats_dict(padded_tail_frames),
        },
    }
    return summary


def print_report(summary, records, max_examples):
    print("=== FBX Frame Statistics Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    ok_records = [record for record in records if record["status"] == "ok"]
    error_records = [record for record in records if record["status"] != "ok"]
    short_records = sorted(ok_records, key=lambda record: record["raw_frame_count"])
    padded_records = sorted(
        [record for record in ok_records if record["would_be_padded_by_current_script"]],
        key=lambda record: (
            -record["padded_tail_frames_if_current_script"],
            record["raw_frame_count"],
            record["path"],
        ),
    )

    if short_records:
        print(f"=== Shortest {min(max_examples, len(short_records))} FBX Files ===")
        for record in short_records[:max_examples]:
            print(
                f"{record['path']}: "
                f"raw_frames={record['raw_frame_count']}, "
                f"fps={record['fps']:.3f}, "
                f"duration_s={record['duration_seconds']:.3f}, "
                f"raw_range=[{record['raw_frame_start']}, {record['raw_frame_end']}]"
            )
        print()

    if padded_records:
        print(
            f"=== Top {min(max_examples, len(padded_records))} Files Most Affected By Current Script ==="
        )
        for record in padded_records[:max_examples]:
            print(
                f"{record['path']}: "
                f"raw_frames={record['raw_frame_count']}, "
                f"exported_frames={record['exported_frame_count_if_current_script']}, "
                f"added_tail={record['padded_tail_frames_if_current_script']}, "
                f"raw_range=[{record['raw_frame_start']}, {record['raw_frame_end']}]"
            )
        print()

    if error_records:
        print(f"=== Errors ({len(error_records)}) ===")
        for record in error_records[:max_examples]:
            print(f"{record['path']}: {record['error']}")
        print()


def resolve_worker_settings(args):
    workers = max(1, args.workers)
    threads_per_worker = max(0, args.threads_per_worker)

    if args.max_cpu_cores > 0:
        if threads_per_worker == 0:
            print(
                "max_cpu_cores was set while threads_per_worker=0 (Blender auto). "
                "For predictable limits, forcing threads_per_worker=1."
            )
            threads_per_worker = 1
        max_workers = max(1, args.max_cpu_cores // max(1, threads_per_worker))
        if workers > max_workers:
            print(
                f"Reducing workers from {workers} to {max_workers} "
                f"to respect max_cpu_cores={args.max_cpu_cores}."
            )
            workers = max_workers

    return workers, threads_per_worker


def find_blender_binary(args):
    candidates = []
    if args.blender_binary is not None:
        candidates.append(args.blender_binary)
    if getattr(bpy.app, "binary_path", None):
        candidates.append(Path(bpy.app.binary_path))
    if sys.argv:
        candidates.append(Path(sys.argv[0]))

    which_blender = shutil.which("blender")
    if which_blender is not None:
        candidates.append(Path(which_blender))

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())

    raise RuntimeError(
        "Could not determine Blender binary path automatically. "
        "Please pass --blender_binary /path/to/blender."
    )


def save_output(json_out, data_root, min_frame_end, summary, records):
    output = {
        "data_root": str(data_root),
        "min_frame_end_rule_simulated": int(min_frame_end),
        "summary": summary,
        "records": records,
    }

    if json_out is not None:
        json_out = json_out.resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON report to: {json_out}")


def run_single_worker(args, data_root):
    fbx_paths = iter_fbx_paths(
        data_root,
        args.glob,
        worker_manifest=args.worker_manifest,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
    )
    if not fbx_paths:
        raise SystemExit(f"No files matched {args.glob!r} under {data_root}")

    worker_label = ""
    if args.worker_count > 1:
        worker_label = f" [worker {args.worker_id + 1}/{args.worker_count}]"
    print(f"Scanning {len(fbx_paths)} FBX files under: {data_root}{worker_label}")

    records = []
    for index, fbx_path in enumerate(fbx_paths, start=1):
        print(f"[{index}/{len(fbx_paths)}]{worker_label} {safe_relative(fbx_path, data_root)}")
        try:
            record = collect_fbx_stats(
                fbx_path,
                data_root,
                args.min_frame_end,
                args.cleanup_mode,
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "path": safe_relative(fbx_path, data_root),
                "status": "error",
                "error": repr(exc),
            }
        records.append(record)

    summary = build_summary(records, args.min_frame_end)
    print_report(summary, records, args.max_examples)
    save_output(args.json_out, data_root, args.min_frame_end, summary, records)


def run_parallel_workers(args, data_root):
    blender_binary = find_blender_binary(args)
    script_path = Path(__file__).resolve()
    all_fbx_paths = sorted(data_root.rglob(args.glob))
    if not all_fbx_paths:
        raise SystemExit(f"No files matched {args.glob!r} under {data_root}")

    workers, threads_per_worker = resolve_worker_settings(args)
    if workers <= 1:
        return run_single_worker(args, data_root)

    output_root = args.json_out.parent.resolve() if args.json_out is not None else data_root
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fbx_frame_stats_", dir=str(output_root)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "all_fbx_files.txt"
        manifest_path.write_text(
            "\n".join(str(path.resolve()) for path in all_fbx_paths) + "\n",
            encoding="utf-8",
        )

        print(
            f"Launching {workers} Blender workers with threads_per_worker={threads_per_worker} "
            f"for {len(all_fbx_paths)} FBX files."
        )

        processes = []
        worker_json_paths = []
        log_paths = []
        for worker_id in range(workers):
            worker_json = tmpdir_path / f"worker_{worker_id:02d}.json"
            worker_log = tmpdir_path / f"worker_{worker_id:02d}.log"
            worker_json_paths.append(worker_json)
            log_paths.append(worker_log)

            cmd = [
                blender_binary,
                "--background",
                "--factory-startup",
            ]
            if threads_per_worker > 0:
                cmd.extend(["-t", str(threads_per_worker)])
            cmd.extend(
                [
                    "--python",
                    str(script_path),
                    "--",
                    "--data_path",
                    str(data_root),
                    "--glob",
                    args.glob,
                    "--min_frame_end",
                    str(args.min_frame_end),
                    "--cleanup_mode",
                    args.cleanup_mode,
                    "--workers",
                    "1",
                    "--worker_manifest",
                    str(manifest_path),
                    "--worker_id",
                    str(worker_id),
                    "--worker_count",
                    str(workers),
                    "--json_out",
                    str(worker_json),
                    "--max_examples",
                    "0",
                ]
            )

            print(f"Starting worker {worker_id + 1}/{workers} -> {worker_log.name}")
            with worker_log.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                )
            processes.append(process)

        failed_workers = []
        for worker_id, process in enumerate(processes):
            return_code = process.wait()
            if return_code != 0:
                failed_workers.append((worker_id, return_code, log_paths[worker_id]))
            else:
                print(f"Worker {worker_id + 1}/{workers} finished successfully.")

        if failed_workers:
            details = "\n".join(
                f"worker {worker_id + 1}: exit_code={return_code}, log={log_path}"
                for worker_id, return_code, log_path in failed_workers
            )
            raise RuntimeError(f"Some worker processes failed:\n{details}")

        records = []
        for worker_json in worker_json_paths:
            if not worker_json.exists():
                raise RuntimeError(f"Worker output JSON was not created: {worker_json}")
            worker_output = json.loads(worker_json.read_text(encoding="utf-8"))
            records.extend(worker_output["records"])

    summary = build_summary(records, args.min_frame_end)
    print_report(summary, records, args.max_examples)
    save_output(args.json_out, data_root, args.min_frame_end, summary, records)


def main():
    args = parse_args()
    data_root = args.data_path.resolve()
    if not data_root.exists():
        raise SystemExit(f"data_path does not exist: {data_root}")

    if args.worker_manifest is None and args.workers > 1:
        run_parallel_workers(args, data_root)
    else:
        run_single_worker(args, data_root)


if __name__ == "__main__":
    main()
