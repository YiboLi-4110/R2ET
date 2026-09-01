#!/usr/bin/env python3
"""
将指定子目录中的 FBX 动画渲染为预览视频（MP4）。

设计目标：清晰看到完整动作，不追求画质。默认用 Cycles 低采样 + GPU，
支持多 Blender 进程并行，并可将 worker 分配到不同 GPU。

用法示例见文件末尾或 README 注释。

必须通过 Blender 启动（或由本脚本在 workers>1 时自动拉起多个 Blender）:

  blender -b -P ./render_fbx_to_video.py -- \\
      --data_path ./Planet_Zoo_FBX-cleaned/train_char \\
      --subdirs clouded_leopard_male red_fox_female \\
      --output_dir ./Planet_Zoo_FBX-cleaned/train_char_videos \\
      --workers 4 --gpu_ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:
    raise SystemExit(
        "This script must be run with Blender, for example:\n"
        "  blender -b -P ./datasets/render_fbx_to_video.py -- \\\n"
        "      --data_path ./datasets/Planet_Zoo_FBX-cleaned/train_char \\\n"
        "      --subdirs clouded_leopard_male --workers 1\n"
    ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Render FBX animations under selected subdirs to MP4 preview videos."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("./Planet_Zoo_FBX-cleaned/train_char"),
        help="Root directory that contains animal subdirectories.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=None,
        help="One or more subdirectory names (or relative/absolute paths) to render. "
        "If omitted, render ALL first-level subdirs under data_path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output root for videos. Default: <data_path>_videos next to data_path.",
    )
    parser.add_argument(
        "--engine",
        choices=("cycles", "workbench", "eevee"),
        default="cycles",
        help="Render engine. 'cycles' is most reliable headless with GPU. "
        "'workbench'/'eevee' need a working display (DISPLAY=:0).",
    )
    parser.add_argument("--resolution", type=str, default="640x480", help="WxH, e.g. 640x480")
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Cycles samples per frame (1 is enough for motion preview).",
    )
    parser.add_argument("--fps", type=int, default=0, help="Override FPS. 0 = keep scene/FBX FPS.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Blender worker processes.",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Blender -t value per worker. Use 1 to avoid CPU oversubscription.",
    )
    parser.add_argument(
        "--max_cpu_cores",
        type=int,
        default=0,
        help="Cap total CPU cores across workers (workers * threads). 0 = no cap.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU indices for Cycles, e.g. '0,1,2,3'. "
        "Workers are assigned round-robin. Empty = let Blender use all visible GPUs.",
    )
    parser.add_argument(
        "--blender_binary",
        type=Path,
        default=None,
        help="Explicit Blender binary for launching parallel workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render even if the output MP4 already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N FBX files (after sorting). 0 = all.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        help="Skip clips shorter than this many frames.",
    )
    parser.add_argument(
        "--camera_azimuth_deg",
        type=float,
        default=40.0,
        help="Camera yaw around subject (degrees).",
    )
    parser.add_argument(
        "--camera_elevation_deg",
        type=float,
        default=20.0,
        help="Camera pitch above horizontal (degrees).",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=2.4,
        help="Camera distance as a multiple of bounding-sphere radius.",
    )
    parser.add_argument(
        "--display",
        type=str,
        default="",
        help="Optional DISPLAY for workbench/eevee, e.g. ':0'.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List FBX files that would be rendered, then exit.",
    )
    # Internal worker args
    parser.add_argument("--worker_manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker_count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker_gpu", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker_log_json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def purge_scene() -> None:
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for name in (
        "actions",
        "armatures",
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "curves",
        "node_groups",
        "worlds",
    ):
        collection = getattr(bpy.data, name, None)
        if collection is None:
            continue
        for datablock in list(collection):
            try:
                collection.remove(datablock, do_unlink=True)
            except Exception:
                pass

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass


def parse_resolution(text: str) -> Tuple[int, int]:
    parts = text.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise SystemExit(f"Invalid --resolution {text!r}, expected WxH like 640x480")
    return int(parts[0]), int(parts[1])


def parse_gpu_ids(text: str) -> List[str]:
    if not text.strip():
        return []
    return [p.strip() for p in text.split(",") if p.strip() != ""]


def world_bbox_corners(obj) -> List[Vector]:
    corners = []
    for corner in obj.bound_box:
        corners.append(obj.matrix_world @ Vector(corner))
    return corners


def compute_scene_bounds(objects: Iterable, sample_frames: Sequence[int]) -> Tuple[Vector, float]:
    """Return (center, radius) from sampled frames of mesh/armature objects."""
    scene = bpy.context.scene
    all_points: List[Vector] = []

    objs = [o for o in objects if o.type in {"MESH", "ARMATURE"}]
    if not objs:
        objs = [o for o in bpy.context.scene.objects if o.type in {"MESH", "ARMATURE"}]

    frames = list(sample_frames) if sample_frames else [scene.frame_current]
    for frame in frames:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        for obj in objs:
            if obj.type == "MESH":
                all_points.extend(world_bbox_corners(obj))
            elif obj.type == "ARMATURE":
                # Prefer pose-bone heads/tails when available
                if obj.pose is not None and obj.pose.bones:
                    for pb in obj.pose.bones:
                        all_points.append(obj.matrix_world @ pb.head)
                        all_points.append(obj.matrix_world @ pb.tail)
                else:
                    all_points.extend(world_bbox_corners(obj))

    if not all_points:
        return Vector((0.0, 0.0, 1.0)), 2.0

    xs = [p.x for p in all_points]
    ys = [p.y for p in all_points]
    zs = [p.z for p in all_points]
    min_v = Vector((min(xs), min(ys), min(zs)))
    max_v = Vector((max(xs), max(ys), max(zs)))
    center = (min_v + max_v) * 0.5
    radius = max((max_v - min_v).length * 0.5, 0.05)
    return center, radius


def look_at(obj, target: Vector, up: Vector = Vector((0.0, 0.0, 1.0))) -> None:
    direction = target - obj.location
    if direction.length < 1e-8:
        return
    quat = direction.to_track_quat("-Z", "Y")
    # Keep a stable up; Blender camera looks down -Z with Y up in camera space.
    obj.rotation_euler = quat.to_euler()
    _ = up  # reserved; track_quat already uses Y-up convention


def setup_camera_and_lights(
    center: Vector,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    distance_scale: float,
) -> None:
    scene = bpy.context.scene
    distance = max(radius * distance_scale, 0.5)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    cam_loc = center + Vector(
        (
            distance * math.cos(el) * math.cos(az),
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
        )
    )

    cam_data = bpy.data.cameras.new("PreviewCamera")
    cam_data.lens = 50.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = max(distance * 20.0, 1000.0)
    cam_obj = bpy.data.objects.new("PreviewCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc
    look_at(cam_obj, center)
    scene.camera = cam_obj

    # Key light
    light_data = bpy.data.lights.new("KeyLight", type="SUN")
    light_data.energy = 2.5
    light_obj = bpy.data.objects.new("KeyLight", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    # Soft fill
    fill_data = bpy.data.lights.new("FillLight", type="SUN")
    fill_data.energy = 0.6
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(20), math.radians(-40), math.radians(-50))

    world = bpy.data.worlds.new("PreviewWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.22, 0.24, 0.28, 1.0)
        bg.inputs[1].default_value = 1.0


def configure_render_settings(
    engine: str,
    resolution: Tuple[int, int],
    samples: int,
    fps: int,
    output_path: Path,
    worker_gpu: str,
) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.use_placeholder = False
    # Blender may still append frame-range to the stem for FFMPEG; we rename afterward.
    scene.render.filepath = str(output_path.with_suffix(""))

    if fps > 0:
        scene.render.fps = int(fps)
        scene.render.fps_base = 1.0

    # FFmpeg MP4
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = max(1, samples)
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 1
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 1
        _configure_cycles_device(worker_gpu)
    elif engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_cavity = False
    elif engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        # Keep defaults; motion preview only.
    else:
        raise SystemExit(f"Unsupported engine: {engine}")


def _configure_cycles_device(worker_gpu: str) -> None:
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences

    # Prefer a single backend (OPTIX > CUDA). Enabling both duplicates each physical GPU.
    selected_type = None
    for device_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
            selected_type = device_type
            break
        except Exception:
            continue

    devices = list(getattr(prefs, "devices", []))
    for d in devices:
        d.use = False

    gpu_devices = [d for d in devices if d.type == selected_type] if selected_type else []
    cpu_devices = [d for d in devices if d.type == "CPU"]

    if worker_gpu != "" and gpu_devices:
        try:
            idx = int(worker_gpu)
        except ValueError:
            idx = 0
        chosen = gpu_devices[idx % len(gpu_devices)]
        chosen.use = True
        scene.cycles.device = "GPU"
        print(f"[cycles] using GPU device: {chosen.name} ({chosen.type})")
        return

    if gpu_devices:
        # Default single-process: use only the first visible GPU to avoid multi-GPU sync overhead
        # for lightweight preview renders. Parallel mode isolates GPUs via CUDA_VISIBLE_DEVICES.
        gpu_devices[0].use = True
        scene.cycles.device = "GPU"
        print(f"[cycles] using GPU device: {gpu_devices[0].name} ({gpu_devices[0].type})")
        return

    if cpu_devices:
        for d in cpu_devices:
            d.use = True
        scene.cycles.device = "CPU"
        print("[cycles] no GPU found, falling back to CPU")
        return

    scene.cycles.device = "CPU"
    print("[cycles] no devices enumerated, using default CPU")


def _find_rendered_movie(out_mp4: Path, frame_start: int, frame_end: int) -> Optional[Path]:
    """Blender often writes '<stem><start>-<end>.mp4' for FFMPEG animation output."""
    candidates = [
        out_mp4,
        Path(str(out_mp4.with_suffix("")) + ".mp4"),
        Path(str(out_mp4) + ".mp4"),
        out_mp4.parent / f"{out_mp4.stem}{frame_start:04d}-{frame_end:04d}.mp4",
    ]
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p

    # Fallback: newest mp4 in the folder that starts with the stem.
    if out_mp4.parent.is_dir():
        matches = sorted(
            out_mp4.parent.glob(f"{out_mp4.stem}*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in matches:
            if p.is_file() and p.stat().st_size > 0:
                return p
    return None


def frame_range_from_scene() -> Tuple[int, int]:
    scene = bpy.context.scene
    f0, f1 = int(scene.frame_start), int(scene.frame_end)

    # Prefer action ranges on armatures / objects.
    ranges = []
    for obj in scene.objects:
        ad = obj.animation_data
        if ad and ad.action:
            fr = ad.action.frame_range
            ranges.append((int(fr[0]), int(fr[1])))
    for action in bpy.data.actions:
        fr = action.frame_range
        ranges.append((int(fr[0]), int(fr[1])))

    if ranges:
        f0 = min(r[0] for r in ranges)
        f1 = max(r[1] for r in ranges)

    if f1 < f0:
        f0, f1 = f1, f0
    return f0, f1


def ensure_meshes_visible() -> int:
    """Make sure mesh objects render; return mesh count."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for obj in meshes:
        obj.hide_render = False
        obj.hide_viewport = False
    # Armatures usually don't appear in Cycles render; that's fine if meshes exist.
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.show_in_front = True
            if hasattr(obj.data, "display_type"):
                obj.data.display_type = "OCTAHEDRAL"
    return len(meshes)


# ---------------------------------------------------------------------------
# Per-file render
# ---------------------------------------------------------------------------


def render_one_fbx(
    fbx_path: Path,
    out_mp4: Path,
    args,
    resolution: Tuple[int, int],
) -> dict:
    t0 = time.time()
    purge_scene()

    try:
        bpy.ops.import_scene.fbx(
            filepath=str(fbx_path),
            use_anim=True,
            automatic_bone_orientation=True,
            ignore_leaf_bones=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(fbx_path),
            "output": str(out_mp4),
            "status": "error",
            "error": f"import_failed: {exc}",
            "seconds": time.time() - t0,
        }

    mesh_count = ensure_meshes_visible()
    f0, f1 = frame_range_from_scene()
    num_frames = f1 - f0 + 1
    if num_frames < args.min_frames:
        return {
            "path": str(fbx_path),
            "output": str(out_mp4),
            "status": "skipped",
            "error": f"too_short:{num_frames}",
            "num_frames": num_frames,
            "seconds": time.time() - t0,
        }

    scene = bpy.context.scene
    scene.frame_start = f0
    scene.frame_end = f1

    # Sample a few frames to frame the camera.
    if num_frames <= 3:
        sample_frames = list(range(f0, f1 + 1))
    else:
        sample_frames = [
            f0,
            f0 + (num_frames - 1) // 2,
            f1,
        ]

    center, radius = compute_scene_bounds(scene.objects, sample_frames)
    setup_camera_and_lights(
        center,
        radius,
        args.camera_azimuth_deg,
        args.camera_elevation_deg,
        args.camera_distance_scale,
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale output so Blender overwrite is clean.
    if out_mp4.exists():
        out_mp4.unlink()

    configure_render_settings(
        engine=args.engine,
        resolution=resolution,
        samples=args.samples,
        fps=args.fps,
        output_path=out_mp4,
        worker_gpu=args.worker_gpu,
    )

    if mesh_count == 0:
        # Cycles cannot render armatures. Fall back tip: still attempt; user will see empty.
        # Prefer workbench if user chose cycles with armature-only — warn clearly.
        print(
            f"[warn] no mesh in {fbx_path.name}; Cycles may produce an empty video. "
            "Consider --engine workbench with DISPLAY=:0 for armature preview."
        )

    try:
        bpy.ops.render.render(animation=True, write_still=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "path": str(fbx_path),
            "output": str(out_mp4),
            "status": "error",
            "error": f"render_failed: {exc}",
            "num_frames": num_frames,
            "mesh_count": mesh_count,
            "seconds": time.time() - t0,
        }

    produced = _find_rendered_movie(out_mp4, f0, f1)
    if produced is None:
        return {
            "path": str(fbx_path),
            "output": str(out_mp4),
            "status": "error",
            "error": "output_missing_after_render",
            "num_frames": num_frames,
            "mesh_count": mesh_count,
            "seconds": time.time() - t0,
        }

    if produced.resolve() != out_mp4.resolve():
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        if out_mp4.exists():
            out_mp4.unlink()
        shutil.move(str(produced), str(out_mp4))

    return {
        "path": str(fbx_path),
        "output": str(out_mp4),
        "status": "ok",
        "num_frames": num_frames,
        "mesh_count": mesh_count,
        "frame_start": f0,
        "frame_end": f1,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# File discovery / output paths
# ---------------------------------------------------------------------------


def resolve_subdirs(data_root: Path, subdirs: Optional[Sequence[str]]) -> List[Path]:
    if not subdirs:
        dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
        if not dirs:
            raise SystemExit(f"No subdirectories found under {data_root}")
        return dirs

    resolved: List[Path] = []
    for name in subdirs:
        p = Path(name)
        if p.is_absolute():
            cand = p
        else:
            cand = data_root / name
            if not cand.exists() and p.exists():
                cand = p.resolve()
        if not cand.exists():
            raise SystemExit(f"Subdir not found: {name} (looked at {cand})")
        if not cand.is_dir():
            raise SystemExit(f"Not a directory: {cand}")
        resolved.append(cand.resolve())
    return resolved


def collect_fbx_files(subdirs: Sequence[Path], limit: int = 0) -> List[Path]:
    files: List[Path] = []
    for d in subdirs:
        files.extend(sorted(d.rglob("*.fbx")))
        files.extend(sorted(d.rglob("*.FBX")))
    # de-dup while preserving order
    seen = set()
    unique = []
    for f in files:
        rp = f.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rp)
    if limit > 0:
        unique = unique[:limit]
    return unique


def output_path_for(fbx_path: Path, data_root: Path, output_root: Path) -> Path:
    try:
        rel = fbx_path.relative_to(data_root)
    except ValueError:
        # FBX outside data_root: keep parent folder name
        rel = Path(fbx_path.parent.name) / fbx_path.name
    return (output_root / rel).with_suffix(".mp4")


def shard_for_worker(paths: Sequence[Path], worker_id: int, worker_count: int) -> List[Path]:
    if worker_count <= 1:
        return list(paths)
    return [p for i, p in enumerate(paths) if i % worker_count == worker_id]


# ---------------------------------------------------------------------------
# Parallel launcher
# ---------------------------------------------------------------------------


def resolve_worker_settings(args) -> Tuple[int, int]:
    workers = max(1, args.workers)
    threads = max(0, args.threads_per_worker)
    if args.max_cpu_cores > 0:
        if threads == 0:
            threads = 1
        max_workers = max(1, args.max_cpu_cores // max(1, threads))
        if workers > max_workers:
            print(
                f"Reducing workers from {workers} to {max_workers} "
                f"to respect max_cpu_cores={args.max_cpu_cores}"
            )
            workers = max_workers
    return workers, threads


def find_blender_binary(args) -> str:
    candidates = []
    if args.blender_binary is not None:
        candidates.append(Path(args.blender_binary))
    if getattr(bpy.app, "binary_path", None):
        candidates.append(Path(bpy.app.binary_path))
    if sys.argv:
        candidates.append(Path(sys.argv[0]))
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    # Project convention
    home_blender = Path.home() / "Softwares/blender-4.5.3-linux-x64/blender"
    candidates.append(home_blender)

    for c in candidates:
        if c and c.exists():
            return str(c.resolve())
    raise RuntimeError(
        "Could not find Blender binary. Pass --blender_binary /path/to/blender"
    )


def run_parallel_workers(args, data_root: Path, output_root: Path, fbx_paths: List[Path]) -> None:
    blender_binary = find_blender_binary(args)
    script_path = Path(__file__).resolve()
    workers, threads = resolve_worker_settings(args)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if workers <= 1:
        return run_single_worker(args, data_root, output_root, fbx_paths)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fbx_render_", dir=str(output_root)) as tmp:
        tmpdir = Path(tmp)
        manifest = tmpdir / "fbx_list.txt"
        manifest.write_text("\n".join(str(p) for p in fbx_paths) + "\n", encoding="utf-8")

        print(
            f"Launching {workers} Blender workers for {len(fbx_paths)} FBX files "
            f"(threads_per_worker={threads}, gpu_ids={gpu_ids or 'auto'})"
        )

        processes = []
        log_paths = []
        json_paths = []
        for worker_id in range(workers):
            worker_log = tmpdir / f"worker_{worker_id:02d}.log"
            worker_json = tmpdir / f"worker_{worker_id:02d}.json"
            log_paths.append(worker_log)
            json_paths.append(worker_json)

            cmd = [blender_binary, "--background", "--factory-startup"]
            if threads > 0:
                cmd.extend(["-t", str(threads)])
            cmd.extend(
                [
                    "--python",
                    str(script_path),
                    "--",
                    "--data_path",
                    str(data_root),
                    "--output_dir",
                    str(output_root),
                    "--engine",
                    args.engine,
                    "--resolution",
                    args.resolution,
                    "--samples",
                    str(args.samples),
                    "--fps",
                    str(args.fps),
                    "--min_frames",
                    str(args.min_frames),
                    "--camera_azimuth_deg",
                    str(args.camera_azimuth_deg),
                    "--camera_elevation_deg",
                    str(args.camera_elevation_deg),
                    "--camera_distance_scale",
                    str(args.camera_distance_scale),
                    "--workers",
                    "1",
                    "--worker_manifest",
                    str(manifest),
                    "--worker_id",
                    str(worker_id),
                    "--worker_count",
                    str(workers),
                    "--worker_log_json",
                    str(worker_json),
                ]
            )
            if args.overwrite:
                cmd.append("--overwrite")
            if args.subdirs:
                cmd.append("--subdirs")
                cmd.extend(args.subdirs)

            env = os.environ.copy()
            if args.display:
                env["DISPLAY"] = args.display
            elif args.engine in {"workbench", "eevee"} and "DISPLAY" not in env:
                env["DISPLAY"] = ":0"

            if gpu_ids:
                assigned = gpu_ids[worker_id % len(gpu_ids)]
                env["CUDA_VISIBLE_DEVICES"] = assigned
                # With a single visible GPU, worker_gpu=0 selects it.
                cmd.extend(["--worker_gpu", "0"])
                print(
                    f"  worker {worker_id + 1}/{workers}: CUDA_VISIBLE_DEVICES={assigned} "
                    f"-> {worker_log.name}"
                )
            else:
                print(f"  worker {worker_id + 1}/{workers}: -> {worker_log.name}")

            with worker_log.open("w", encoding="utf-8") as logf:
                proc = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                    env=env,
                )
            processes.append(proc)

        failed = []
        for worker_id, proc in enumerate(processes):
            code = proc.wait()
            if code != 0:
                failed.append((worker_id, code, log_paths[worker_id]))
            else:
                print(f"Worker {worker_id + 1}/{workers} finished OK")

        # Merge logs/json for summary
        records = []
        for jp in json_paths:
            if jp.exists():
                records.extend(json.loads(jp.read_text(encoding="utf-8")).get("records", []))

        _print_summary(records)

        # Persist merged summary next to outputs
        summary_path = output_root / "render_summary.json"
        summary_path.write_text(
            json.dumps({"records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote summary: {summary_path}")

        if failed:
            details = "\n".join(
                f"worker {wid + 1}: exit={code}, log={log}" for wid, code, log in failed
            )
            # Copy failed logs next to output for inspection
            for wid, _code, log in failed:
                dst = output_root / f"worker_{wid:02d}_FAILED.log"
                try:
                    shutil.copy2(log, dst)
                except Exception:
                    pass
            raise RuntimeError(f"Some workers failed:\n{details}")


def _print_summary(records: List[dict]) -> None:
    ok = sum(1 for r in records if r.get("status") == "ok")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    err = sum(1 for r in records if r.get("status") == "error")
    total_sec = sum(float(r.get("seconds", 0.0)) for r in records)
    print(
        f"Summary: ok={ok}, skipped={skipped}, error={err}, "
        f"files={len(records)}, sum_worker_seconds={total_sec:.1f}"
    )
    for r in records:
        if r.get("status") == "error":
            print(f"  ERROR {r.get('path')}: {r.get('error')}")


def run_single_worker(
    args,
    data_root: Path,
    output_root: Path,
    fbx_paths: Optional[List[Path]] = None,
) -> None:
    if args.worker_manifest is not None:
        lines = args.worker_manifest.read_text(encoding="utf-8").splitlines()
        fbx_paths = [Path(line) for line in lines if line.strip()]
        fbx_paths = shard_for_worker(fbx_paths, args.worker_id, args.worker_count)
    elif fbx_paths is None:
        subdirs = resolve_subdirs(data_root, args.subdirs)
        fbx_paths = collect_fbx_files(subdirs, limit=args.limit)

    if not fbx_paths:
        print("No FBX files to process.")
        records = []
    else:
        label = ""
        if args.worker_count > 1:
            label = f" [worker {args.worker_id + 1}/{args.worker_count}]"
        print(f"Rendering {len(fbx_paths)} FBX files{label}")
        print(f"  data_path = {data_root}")
        print(f"  output_dir = {output_root}")
        print(f"  engine={args.engine}, resolution={args.resolution}, samples={args.samples}")

        resolution = parse_resolution(args.resolution)
        records = []
        for i, fbx in enumerate(fbx_paths, 1):
            out_mp4 = output_path_for(fbx, data_root, output_root)
            if out_mp4.exists() and not args.overwrite:
                print(f"[{i}/{len(fbx_paths)}]{label} skip existing {out_mp4.name}")
                records.append(
                    {
                        "path": str(fbx),
                        "output": str(out_mp4),
                        "status": "skipped",
                        "error": "exists",
                        "seconds": 0.0,
                    }
                )
                continue

            print(f"[{i}/{len(fbx_paths)}]{label} {fbx.name} -> {out_mp4}")
            try:
                reset_scene()
                record = render_one_fbx(fbx, out_mp4, args, resolution)
            except Exception as exc:  # noqa: BLE001
                record = {
                    "path": str(fbx),
                    "output": str(out_mp4),
                    "status": "error",
                    "error": repr(exc),
                    "seconds": 0.0,
                }
            records.append(record)
            status = record.get("status")
            print(
                f"    -> {status}"
                + (
                    f" frames={record.get('num_frames')}"
                    if record.get("num_frames") is not None
                    else ""
                )
                + f" ({record.get('seconds', 0):.1f}s)"
                + (f" err={record.get('error')}" if status == "error" else "")
            )

    if args.worker_log_json is not None:
        args.worker_log_json.parent.mkdir(parents=True, exist_ok=True)
        args.worker_log_json.write_text(
            json.dumps({"records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        _print_summary(records)
        summary_path = output_root / "render_summary.json"
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"records": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_root = args.data_path.resolve()
    if not data_root.exists():
        raise SystemExit(f"data_path does not exist: {data_root}")

    if args.output_dir is None:
        output_root = data_root.parent / f"{data_root.name}_videos"
    else:
        output_root = args.output_dir.resolve()

    # If user passed --gpu_ids with a single worker and worker_gpu not set,
    # bind to the first listed GPU index among Cycles devices.
    if not args.worker_gpu and args.gpu_ids:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if gpu_ids and args.workers <= 1 and args.worker_manifest is None:
            args.worker_gpu = gpu_ids[0]

    # Discovery (also for dry-run / parallel)
    if args.worker_manifest is None:
        subdirs = resolve_subdirs(data_root, args.subdirs)
        fbx_paths = collect_fbx_files(subdirs, limit=args.limit)
        print("Selected subdirs:")
        for d in subdirs:
            try:
                rel = d.relative_to(data_root)
            except ValueError:
                rel = d
            n = sum(1 for _ in d.rglob("*.fbx")) + sum(1 for _ in d.rglob("*.FBX"))
            print(f"  - {rel} ({n} fbx)")
        print(f"Total FBX to consider: {len(fbx_paths)}")
        if args.dry_run:
            for p in fbx_paths:
                print(p)
            return
    else:
        fbx_paths = None  # loaded inside worker

    if args.worker_manifest is None and args.workers > 1:
        run_parallel_workers(args, data_root, output_root, fbx_paths)
    else:
        # For single-process GPU isolation, re-exec under CUDA_VISIBLE_DEVICES when needed.
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if (
            args.worker_manifest is None
            and gpu_ids
            and "CUDA_VISIBLE_DEVICES" not in os.environ
            and args.engine == "cycles"
        ):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            blender_binary = find_blender_binary(args)
            script_path = Path(__file__).resolve()
            cmd = [
                blender_binary,
                "--background",
                "--factory-startup",
                "--python",
                str(script_path),
                "--",
                "--data_path",
                str(data_root),
                "--output_dir",
                str(output_root),
                "--engine",
                args.engine,
                "--resolution",
                args.resolution,
                "--samples",
                str(args.samples),
                "--fps",
                str(args.fps),
                "--min_frames",
                str(args.min_frames),
                "--camera_azimuth_deg",
                str(args.camera_azimuth_deg),
                "--camera_elevation_deg",
                str(args.camera_elevation_deg),
                "--camera_distance_scale",
                str(args.camera_distance_scale),
                "--workers",
                "1",
                "--worker_gpu",
                "0",
            ]
            if args.subdirs:
                cmd.append("--subdirs")
                cmd.extend(args.subdirs)
            if args.overwrite:
                cmd.append("--overwrite")
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            print(f"Re-launching with CUDA_VISIBLE_DEVICES={gpu_ids[0]}")
            raise SystemExit(subprocess.call(cmd, env=env, cwd=str(Path.cwd())))

        run_single_worker(args, data_root, output_root, fbx_paths)


if __name__ == "__main__":
    main()
