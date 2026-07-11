#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "This script must be run with Blender, for example:\n"
        "  blender -b -P ./datasets/fbx2bvh_smal33.py --\n"
        "    --data_path ./datasets/Planet_Zoo_FBX-smal2/train_char\n"
    ) from exc


def parse_args():
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(
        description=(
            "Convert Planet Zoo FBX files to BVH while preserving the original frame length. "
            "Static/rest-pose FBX files (no animation) are supported automatically."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("./Planet_Zoo_FBX-smal2/train_char"),
        help="Root directory containing per-character subdirectories of .fbx files.",
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
        "--cleanup_mode",
        type=str,
        choices=["purge", "factory"],
        default="purge",
        help="How to clean the Blender scene between files. 'purge' is usually much faster.",
    )
    parser.add_argument(
        "--blender_binary",
        type=Path,
        default=None,
        help="Optional explicit Blender binary path for parallel worker mode.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="If set, re-export BVH files even when the target already exists.",
    )
    parser.add_argument(
        "--force_rest_pose",
        action="store_true",
        help=(
            "Always export a duplicated rest-pose BVH instead of using imported actions. "
            "Useful for batch2_dogs static FBX assets."
        ),
    )
    parser.add_argument(
        "--min_rest_frames",
        type=int,
        default=2,
        help=(
            "Minimum number of frames for static/rest exports. "
            "SMAL33 inference requires >=2 frames (default: 2)."
        ),
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print progress every N files in worker mode. Use 1 for per-file logs.",
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
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mocap_y_up",
        action="store_true",
        help=(
            "Import FBX with mocap-style axes (-Z forward, Y up). "
            "Recommended for non-Planet-Zoo assets (e.g. smal@shepherd). "
            "Default off preserves legacy Planet Zoo export behavior."
        ),
    )
    parser.add_argument(
        "--fbx_axis_forward",
        type=str,
        default="",
        help="Optional FBX import axis_forward override (e.g. -Z). Ignored when empty.",
    )
    parser.add_argument(
        "--fbx_axis_up",
        type=str,
        default="",
        help="Optional FBX import axis_up override (e.g. Y). Ignored when empty.",
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


def safe_relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def list_fbx_jobs(data_root):
    jobs = []
    directories = sorted([path for path in data_root.iterdir() if path.is_dir() and not path.name.startswith(".")])
    for directory in directories:
        files = sorted([path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".fbx"])
        for sourcepath in files:
            dumppath = sourcepath.with_suffix(".bvh")
            jobs.append((sourcepath, dumppath))
    return jobs


def iter_jobs(data_root, worker_manifest=None, worker_id=0, worker_count=1):
    if worker_manifest is None:
        jobs = list_fbx_jobs(data_root)
    else:
        lines = worker_manifest.read_text(encoding="utf-8").splitlines()
        jobs = []
        for line in lines:
            if not line.strip():
                continue
            sourcepath = Path(line.strip())
            jobs.append((sourcepath, sourcepath.with_suffix(".bvh")))

    if worker_count <= 1:
        return jobs
    return [job for index, job in enumerate(jobs) if index % worker_count == worker_id]


def find_armature_objects():
    return sorted(
        (obj for obj in bpy.data.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )


def animated_frame_range():
    if not bpy.data.actions:
        return None

    frame_start = int(9999)
    frame_end = int(-9999)
    for action in bpy.data.actions:
        frame_start = min(frame_start, int(action.frame_range[0]))
        frame_end = max(frame_end, int(action.frame_range[1]))

    if frame_end < frame_start:
        return None
    return frame_start, frame_end


def ensure_object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def prepare_rest_pose_export(min_rest_frames):
    """Bake identical rest-pose keys so static FBX can export >= min_rest_frames."""
    min_rest_frames = max(2, int(min_rest_frames))
    frame_start = 1
    frame_end = frame_start + min_rest_frames - 1

    armatures = find_armature_objects()
    if not armatures:
        raise RuntimeError("No armature found after FBX import.")

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.fps = 30

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")

    for arm_obj in armatures:
        arm_obj.data.pose_position = "REST"
        arm_obj.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj

        bpy.ops.object.mode_set(mode="POSE")
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            for pose_bone in arm_obj.pose.bones:
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                if pose_bone.parent is None:
                    pose_bone.keyframe_insert(data_path="location", frame=frame)
        bpy.ops.object.mode_set(mode="OBJECT")

    return frame_start, frame_end, "rest_pose"


def resolve_export_frames(force_rest_pose, min_rest_frames):
    if force_rest_pose:
        return prepare_rest_pose_export(min_rest_frames)

    frame_range = animated_frame_range()
    if frame_range is None:
        return prepare_rest_pose_export(min_rest_frames)

    frame_start, frame_end = frame_range
    frame_count = frame_end - frame_start + 1
    min_rest_frames = max(2, int(min_rest_frames))
    if frame_count < min_rest_frames:
        return prepare_rest_pose_export(min_rest_frames)

    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    return frame_start, frame_end, "animated"


def resolve_fbx_import_axes(args):
    if args.fbx_axis_forward or args.fbx_axis_up:
        return args.fbx_axis_forward or "-Z", args.fbx_axis_up or "Y"
    if args.mocap_y_up:
        return "-Z", "Y"
    return None, None


def import_fbx(sourcepath, axis_forward=None, axis_up=None):
    kwargs = {"filepath": str(sourcepath), "use_anim": True}
    if axis_forward and axis_up:
        kwargs["axis_forward"] = axis_forward
        kwargs["axis_up"] = axis_up
    bpy.ops.import_scene.fbx(**kwargs)


def export_bvh(filepath, frame_start, frame_end):
    ensure_object_mode()
    armatures = find_armature_objects()
    bpy.ops.object.select_all(action="DESELECT")
    for arm_obj in armatures:
        arm_obj.select_set(True)
    if armatures:
        bpy.context.view_layer.objects.active = armatures[0]

    bpy.ops.export_anim.bvh(
        filepath=str(filepath),
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        root_transform_only=True,
    )


def cleanup_imported_actions():
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def convert_one(sourcepath, dumppath, cleanup_mode, force_rest_pose=False, min_rest_frames=2, fbx_axes=None):
    cleanup_blender_scene(cleanup_mode)
    axis_forward, axis_up = fbx_axes if fbx_axes is not None else (None, None)
    import_fbx(sourcepath, axis_forward=axis_forward, axis_up=axis_up)

    frame_start, frame_end, export_mode = resolve_export_frames(
        force_rest_pose=force_rest_pose,
        min_rest_frames=min_rest_frames,
    )
    export_bvh(dumppath, frame_start, frame_end)
    cleanup_imported_actions()

    return {
        "path": str(sourcepath),
        "status": "processed",
        "export_mode": export_mode,
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "frame_count": int(frame_end - frame_start + 1),
    }


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


def save_worker_output(args, data_root, results):
    if args.json_out is None:
        return

    payload = {
        "data_root": str(data_root),
        "worker_id": int(args.worker_id),
        "worker_count": int(args.worker_count),
        "results": results,
    }
    json_out = args.json_out.resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_single_worker(args, data_root):
    jobs = iter_jobs(
        data_root,
        worker_manifest=args.worker_manifest,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
    )
    if not jobs:
        raise SystemExit(f"No .fbx files found under {data_root}")

    fbx_axes = resolve_fbx_import_axes(args)
    if fbx_axes != (None, None):
        print(f"FBX import axes: forward={fbx_axes[0]!r}, up={fbx_axes[1]!r}")

    worker_label = ""
    if args.worker_count > 1:
        worker_label = f" [worker {args.worker_id + 1}/{args.worker_count}]"
    print(f"Converting {len(jobs)} FBX files under: {data_root}{worker_label}")

    results = []
    processed = 0
    skipped = 0
    errors = 0
    log_every = max(1, args.log_every)

    for index, (sourcepath, dumppath) in enumerate(jobs, start=1):
        relpath = safe_relative(sourcepath, data_root)
        if dumppath.exists() and not args.overwrite_existing:
            skipped += 1
            results.append({"path": str(sourcepath), "status": "skipped_existing"})
        else:
            try:
                result = convert_one(
                    sourcepath,
                    dumppath,
                    args.cleanup_mode,
                    force_rest_pose=args.force_rest_pose,
                    min_rest_frames=args.min_rest_frames,
                    fbx_axes=fbx_axes,
                )
                processed += 1
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                results.append({"path": str(sourcepath), "status": "error", "error": repr(exc)})

        if index % log_every == 0 or index == len(jobs):
            print(
                f"[{index}/{len(jobs)}]{worker_label} "
                f"processed={processed} skipped={skipped} errors={errors} last={relpath}"
            )

    print(
        f"Worker done{worker_label}: total={len(jobs)} "
        f"processed={processed} skipped={skipped} errors={errors}"
    )
    save_worker_output(args, data_root, results)


def run_parallel_workers(args, data_root):
    blender_binary = find_blender_binary(args)
    script_path = Path(__file__).resolve()
    jobs = list_fbx_jobs(data_root)
    if not jobs:
        raise SystemExit(f"No .fbx files found under {data_root}")

    workers, threads_per_worker = resolve_worker_settings(args)
    if workers <= 1:
        return run_single_worker(args, data_root)

    output_root = data_root
    if args.json_out is not None:
        output_root = args.json_out.parent.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fbx2bvh_smal33_", dir=str(output_root)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "all_fbx_files.txt"
        manifest_path.write_text(
            "\n".join(str(sourcepath.resolve()) for sourcepath, _ in jobs) + "\n",
            encoding="utf-8",
        )

        print(
            f"Launching {workers} Blender workers with threads_per_worker={threads_per_worker} "
            f"for {len(jobs)} FBX files."
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
                    "--workers",
                    "1",
                    "--threads_per_worker",
                    str(threads_per_worker),
                    "--cleanup_mode",
                    args.cleanup_mode,
                    "--worker_manifest",
                    str(manifest_path),
                    "--worker_id",
                    str(worker_id),
                    "--worker_count",
                    str(workers),
                    "--json_out",
                    str(worker_json),
                    "--log_every",
                    str(args.log_every),
                ]
            )
            if args.overwrite_existing:
                cmd.append("--overwrite_existing")
            if args.force_rest_pose:
                cmd.append("--force_rest_pose")
            cmd.extend(["--min_rest_frames", str(args.min_rest_frames)])
            if args.mocap_y_up:
                cmd.append("--mocap_y_up")
            if args.fbx_axis_forward:
                cmd.extend(["--fbx_axis_forward", args.fbx_axis_forward])
            if args.fbx_axis_up:
                cmd.extend(["--fbx_axis_up", args.fbx_axis_up])

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

        total_processed = 0
        total_skipped = 0
        total_errors = 0
        for worker_json in worker_json_paths:
            if not worker_json.exists():
                raise RuntimeError(f"Worker output JSON was not created: {worker_json}")
            payload = json.loads(worker_json.read_text(encoding="utf-8"))
            for result in payload["results"]:
                status = result["status"]
                if status == "processed":
                    total_processed += 1
                elif status == "skipped_existing":
                    total_skipped += 1
                elif status == "error":
                    total_errors += 1

    print(
        f"All workers done: total={len(jobs)} "
        f"processed={total_processed} skipped={total_skipped} errors={total_errors}"
    )


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
