#!/usr/bin/env python3
"""
Batch: for each dog_id, retarget ALL shepherd train_char actions and pack one
textured .blend (editable FBX armature bake by default, or LBS shape-keys).

retarget_mode:
  r2et   -> R2ET stage1/stage2/blend
  direct -> CopyQuat (same-skeleton; fourway-compatible)
  arp    -> Auto-Rig Pro in Blender (cross-skeleton; same pack layout)

Multi-GPU / multi-worker:
  - Multiple dogs: one dog per GPU worker in parallel
  - Single dog + multiple GPUs: shard actions across GPUs, merge, then pack
  - arp ignores CUDA; --gpus still controls parallel Blender workers

Example:
  python visualization/batch_r2et_dog_blend.py \\
    --config config/visualization_blend_per_dog_smal33.yaml \\
    --dog_ids 博美_3 \\
    --limit_actions 3 \\
    --gpus 0 \\
    --keep_intermediates

  # CopyQuat mode:
  python visualization/batch_r2et_dog_blend.py \\
    --config config/visualization_blend_per_dog_smal33.yaml \\
    --retarget_mode direct \\
    --dog_ids 泰迪_1 \\
    --gpus 0

  # ARP mode:
  python visualization/batch_r2et_dog_blend.py \\
    --config config/visualization_blend_per_dog_smal33.yaml \\
    --retarget_mode arp \\
    --blender blender \\
    --arp_addon_modules auto_rig_pro-master \\
    --gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from export_r2et_dog_actions_smal33 import (  # noqa: E402
    merge_shard_manifests,
    normalize_action_specs,
    select_source_actions,
)

DEFAULT_CONFIG = _REPO_ROOT / "config/visualization_blend_per_dog_smal33.yaml"
_print_lock = threading.Lock()


def log(message: str):
    with _print_lock:
        print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-worker batch export of per-dog R2ET/CopyQuat/ARP .blend files."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blend_output_dir", type=Path, default=None)
    parser.add_argument("--work_dir", type=Path, default=None)
    parser.add_argument("--dog_ids", type=str, nargs="+", default=None)
    default_blender = os.environ.get("BLENDER", "blender")
    parser.add_argument("--blender", type=str, default=default_blender)
    parser.add_argument("--device", type=int, default=None, help="Deprecated single-GPU override.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1,2,3. Default: config device or 0.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Parallel workers. Default: number of GPUs.",
    )
    parser.add_argument(
        "--retarget_mode",
        type=str,
        default=None,
        choices=["r2et", "direct", "arp"],
        help="Override retarget_mode (r2et | direct/CopyQuat | arp).",
    )
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
        help="ARP addon module names (retarget_mode=arp).",
    )
    parser.add_argument(
        "--stage2_mode",
        type=str,
        default=None,
        choices=["blend", "stage1", "stage2"],
        help="R2ET only: override stage2.mode.",
    )
    parser.add_argument("--gate_scale", type=float, default=None)
    parser.add_argument(
        "--stage1_weights",
        type=Path,
        default=None,
        help="Override model.stage1_weights for mode=stage1.",
    )
    parser.add_argument(
        "--stage2_weights",
        type=Path,
        default=None,
        help="Override model.stage2_weights for mode=stage2|blend.",
    )
    parser.add_argument("--limit_actions", type=int, default=None)
    parser.add_argument(
        "--min_frames",
        type=int,
        default=None,
        help=(
            "Skip source BVH clips with fewer than this many frames "
            "(overrides source.min_frames; 0 = no filter). "
            "Ignored when --actions / source.actions is set."
        ),
    )
    parser.add_argument(
        "--actions",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Explicit source clips (overrides train_char scan + min_frames). "
            "BVH path / Folder/stem / stem. Overrides source.actions when set."
        ),
    )
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=None,
        help=(
            "CPU thread cap for each worker subprocess "
            "(OMP/MKL/OpenBLAS/NUMEXPR + torch + export --cpu_threads). "
            "Example: --gpus 1,2 --cpu_threads_per_worker 8"
        ),
    )
    parser.add_argument(
        "--total_cpu_threads",
        type=int,
        default=None,
        help=(
            "Total CPU thread budget across concurrent workers; "
            "per-worker = total // concurrent_workers when "
            "--cpu_threads_per_worker is unset."
        ),
    )
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=None,
        help=(
            "Blender --threads for pack step. "
            "Default: same as --cpu_threads_per_worker when that is set."
        ),
    )
    parser.add_argument(
        "--skip_bvh_export",
        action="store_true",
        default=False,
        help="Skip Ours BVH export (faster; LBS npz is enough for packing).",
    )
    parser.add_argument("--keep_intermediates", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_gpu_list(raw: str | None, fallback: int = 0) -> list[int]:
    if raw is None or not str(raw).strip():
        return [int(fallback)]
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        return [int(fallback)]
    return [int(v) for v in values]


def resolve_cpu_threads_per_worker(args, concurrent_workers: int) -> int | None:
    """Unified per-worker CPU thread cap, or None to keep process defaults."""
    if args.cpu_threads_per_worker is not None:
        return max(int(args.cpu_threads_per_worker), 1)
    if args.total_cpu_threads is not None:
        return max(int(args.total_cpu_threads) // max(int(concurrent_workers), 1), 1)
    return None


def resolve_blender_threads(args, cpu_threads_per_worker: int | None) -> int | None:
    if args.blender_threads is not None:
        return max(int(args.blender_threads), 0) or None
    if cpu_threads_per_worker is not None:
        return int(cpu_threads_per_worker)
    return None


def subprocess_env(cpu_threads_per_worker: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if cpu_threads_per_worker is None:
        return env
    thread_value = str(cpu_threads_per_worker)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[key] = thread_value
    return env


def run_cmd(cmd: list[str], *, cwd: Path, step: str, env: dict[str, str] | None = None):
    log(f"[dog-blend][run] {step}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def resolve_explicit_actions(args, cfg: dict[str, Any]) -> list[str]:
    if args.actions is not None:
        return list(args.actions)
    source_cfg = cfg.get("source", {}) or {}
    return normalize_action_specs(source_cfg.get("actions"))


def discover_action_count(
    cfg: dict[str, Any],
    min_frames: int = 0,
    *,
    explicit_actions: list[str] | None = None,
    require_shape: bool = True,
) -> tuple[int, Path, str]:
    source_cfg = cfg.get("source", {}) or {}
    train_char = Path(source_cfg.get("train_char", "./datasets/shepherd/smal@shepherd/train_char"))
    train_shape = Path(source_cfg.get("train_shape", "./datasets/shepherd/smal@shepherd/train_shape"))
    if not train_char.is_absolute():
        train_char = (_REPO_ROOT / train_char).resolve()
    if not train_shape.is_absolute():
        train_shape = (_REPO_ROOT / train_shape).resolve()

    specs = normalize_action_specs(explicit_actions)
    if specs or train_shape.is_dir() or not require_shape:
        try:
            actions, mode = select_source_actions(
                train_char,
                train_shape,
                min_frames=min_frames,
                explicit_actions=specs,
                repo_root=_REPO_ROOT,
                require_shape=require_shape,
            )
            return len(actions), train_char, mode
        except FileNotFoundError:
            if specs:
                raise

    if not train_char.is_dir():
        return 0, train_char, "scan"
    count = 0
    for folder in train_char.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        bvhs = list(folder.glob("*.bvh"))
        if bvhs:
            count += len(bvhs)
        else:
            count += len(list(folder.glob("*.fbx")))
    return count, train_char, "scan"


def resolve_min_frames(args, cfg: dict[str, Any]) -> int:
    if args.min_frames is not None:
        return max(int(args.min_frames), 0)
    source_cfg = cfg.get("source", {}) or {}
    return max(int(source_cfg.get("min_frames", 0) or 0), 0)


def export_cmd_base(
    args,
    cfg_path: Path,
    work_dir: Path,
    *,
    cpu_threads_per_worker: int | None = None,
    retarget_mode: str | None = None,
    blender_threads: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "export_r2et_dog_actions_smal33.py"),
        "--config",
        str(cfg_path),
        "--output_root",
        str(work_dir),
    ]
    if args.retarget_mode is not None:
        cmd.extend(["--retarget_mode", args.retarget_mode])
    if args.stage2_mode is not None:
        cmd.extend(["--stage2_mode", args.stage2_mode])
    if args.gate_scale is not None:
        cmd.extend(["--gate_scale", str(args.gate_scale)])
    if args.stage1_weights is not None:
        cmd.extend(["--stage1_weights", str(args.stage1_weights)])
    if args.stage2_weights is not None:
        cmd.extend(["--stage2_weights", str(args.stage2_weights)])
    if args.limit_actions is not None:
        cmd.extend(["--limit_actions", str(args.limit_actions)])
    if args.min_frames is not None:
        cmd.extend(["--min_frames", str(args.min_frames)])
    if args.actions is not None:
        cmd.append("--actions")
        cmd.extend(list(args.actions))
    if args.skip_bvh_export:
        cmd.append("--skip_bvh_export")
    if cpu_threads_per_worker is not None:
        cmd.extend(["--cpu_threads", str(cpu_threads_per_worker)])
    mode = str(retarget_mode or args.retarget_mode or "").strip().lower()
    if mode == "arp":
        cmd.extend(["--blender", str(args.blender)])
        cmd.extend(["--arp_addon_modules", *list(args.arp_addon_modules)])
        if blender_threads is not None and int(blender_threads) > 0:
            cmd.extend(["--blender_threads", str(int(blender_threads))])
    return cmd


def pack_one_manifest(
    args,
    manifest_path: Path,
    blend_output_dir: Path,
    dog_id: str,
    *,
    cpu_threads_per_worker: int | None = None,
    blender_threads: int | None = None,
):
    blender_cmd = [args.blender]
    if blender_threads is not None and blender_threads > 0:
        blender_cmd.extend(["--threads", str(blender_threads)])
    blender_cmd.extend(
        [
            "--background",
            "--python",
            str(_SCRIPT_DIR / "pack_r2et_dog_blend_blender.py"),
            "--",
            "--manifest",
            str(manifest_path),
            "--blend_output_dir",
            str(blend_output_dir),
            "--dog_ids",
            dog_id,
        ]
    )
    run_cmd(
        blender_cmd,
        cwd=_REPO_ROOT,
        step=f"pack-blend[{dog_id}]",
        env=subprocess_env(cpu_threads_per_worker),
    )


def process_dog_on_gpu(
    dog_id: str,
    gpu_id: int,
    args,
    cfg_path: Path,
    work_dir: Path,
    blend_output_dir: Path,
    *,
    cpu_threads_per_worker: int | None = None,
    blender_threads: int | None = None,
) -> tuple[str, bool, str | None]:
    try:
        cmd = export_cmd_base(
            args,
            cfg_path,
            work_dir,
            cpu_threads_per_worker=cpu_threads_per_worker,
            retarget_mode=str(getattr(args, "retarget_mode", None) or ""),
            blender_threads=blender_threads,
        )
        cmd.extend(["--device", str(gpu_id), "--dog_ids", dog_id])
        run_cmd(
            cmd,
            cwd=_REPO_ROOT,
            step=f"export[{dog_id} gpu={gpu_id}]",
            env=subprocess_env(cpu_threads_per_worker),
        )

        manifest_path = work_dir / dog_id / "dog_actions_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        pack_one_manifest(
            args,
            manifest_path,
            blend_output_dir,
            dog_id,
            cpu_threads_per_worker=cpu_threads_per_worker,
            blender_threads=blender_threads,
        )

        out_blend = blend_output_dir / f"{dog_id}.blend"
        if not out_blend.exists():
            return dog_id, False, f"missing blend: {out_blend}"
        log(f"[dog-blend][done] {dog_id} -> {out_blend} (gpu={gpu_id})")
        return dog_id, True, None
    except Exception as exc:
        log(f"[dog-blend][fail] {dog_id} gpu={gpu_id}: {exc}")
        return dog_id, False, str(exc)


def process_single_dog_sharded(
    dog_id: str,
    gpu_ids: list[int],
    args,
    cfg_path: Path,
    work_dir: Path,
    blend_output_dir: Path,
    *,
    cpu_threads_per_worker: int | None = None,
    blender_threads: int | None = None,
) -> tuple[str, bool, str | None]:
    """Shard one dog's actions across GPUs, merge manifests, then pack once."""
    num_shards = len(gpu_ids)
    try:
        def _run_shard(shard_index: int, gpu_id: int):
            cmd = export_cmd_base(
                args,
                cfg_path,
                work_dir,
                cpu_threads_per_worker=cpu_threads_per_worker,
                retarget_mode=str(getattr(args, "retarget_mode", None) or ""),
                blender_threads=blender_threads,
            )
            cmd.extend(
                [
                    "--device",
                    str(gpu_id),
                    "--dog_ids",
                    dog_id,
                    "--shard_index",
                    str(shard_index),
                    "--num_shards",
                    str(num_shards),
                ]
            )
            run_cmd(
                cmd,
                cwd=_REPO_ROOT,
                step=f"export[{dog_id} shard={shard_index}/{num_shards} gpu={gpu_id}]",
                env=subprocess_env(cpu_threads_per_worker),
            )

        with ThreadPoolExecutor(max_workers=num_shards) as executor:
            futures = [
                executor.submit(_run_shard, shard_i, gpu_id)
                for shard_i, gpu_id in enumerate(gpu_ids)
            ]
            for fut in as_completed(futures):
                fut.result()

        dog_out = work_dir / dog_id
        merge_shard_manifests(dog_out, dog_id, num_shards)
        manifest_path = dog_out / "dog_actions_manifest.json"
        pack_one_manifest(
            args,
            manifest_path,
            blend_output_dir,
            dog_id,
            cpu_threads_per_worker=cpu_threads_per_worker,
            blender_threads=blender_threads,
        )
        out_blend = blend_output_dir / f"{dog_id}.blend"
        if not out_blend.exists():
            return dog_id, False, f"missing blend: {out_blend}"
        log(f"[dog-blend][done] {dog_id} (sharded x{num_shards}) -> {out_blend}")
        return dog_id, True, None
    except Exception as exc:
        log(f"[dog-blend][fail] {dog_id} sharded: {exc}")
        return dog_id, False, str(exc)


def run_parallel_dogs(
    dog_ids: list[str],
    gpu_ids: list[int],
    num_workers: int,
    args,
    cfg_path: Path,
    work_dir: Path,
    blend_output_dir: Path,
    *,
    cpu_threads_per_worker: int | None = None,
    blender_threads: int | None = None,
) -> tuple[list[str], list[str]]:
    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    produced: list[str] = []
    failed: list[str] = []

    def _run(dog_id: str):
        gpu_id = gpu_queue.get()
        try:
            return process_dog_on_gpu(
                dog_id,
                gpu_id,
                args,
                cfg_path,
                work_dir,
                blend_output_dir,
                cpu_threads_per_worker=cpu_threads_per_worker,
                blender_threads=blender_threads,
            )
        finally:
            gpu_queue.put(gpu_id)

    log(
        f"[dog-blend] parallel dogs={len(dog_ids)} workers={num_workers} gpus={gpu_ids}"
        + (
            f" cpu_threads_per_worker={cpu_threads_per_worker}"
            if cpu_threads_per_worker is not None
            else ""
        )
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run, dog_id): dog_id for dog_id in dog_ids}
        for fut in as_completed(futures):
            dog_id, ok, _err = fut.result()
            if ok:
                produced.append(dog_id)
            else:
                failed.append(dog_id)
    return produced, failed


def main():
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = load_yaml(cfg_path)

    if args.blend_output_dir is not None:
        cfg["blend_output_dir"] = str(args.blend_output_dir)
    if args.retarget_mode is not None:
        cfg["retarget_mode"] = args.retarget_mode
    if args.stage2_mode is not None:
        cfg.setdefault("stage2", {})
        cfg["stage2"]["mode"] = args.stage2_mode
    if args.gate_scale is not None:
        cfg.setdefault("stage2", {})
        cfg["stage2"]["gate_scale"] = float(args.gate_scale)
    if args.stage1_weights is not None:
        cfg.setdefault("model", {})
        cfg["model"]["stage1_weights"] = str(args.stage1_weights)
    if args.stage2_weights is not None:
        cfg.setdefault("model", {})
        cfg["model"]["stage2_weights"] = str(args.stage2_weights)

    retarget_mode = str(cfg.get("retarget_mode", "r2et")).strip().lower()
    if retarget_mode in ("copyquat", "copy_quat", "quat"):
        retarget_mode = "direct"
    if retarget_mode not in ("r2et", "direct", "arp"):
        raise SystemExit(
            f"Unknown retarget_mode={cfg.get('retarget_mode')!r}; expected r2et|direct|arp."
        )
    args.retarget_mode = retarget_mode

    dog_ids = args.dog_ids or list((cfg.get("targets", {}) or {}).get("dog_ids") or [])
    if not dog_ids:
        raise SystemExit("No dog_ids in config or CLI.")

    min_frames = resolve_min_frames(args, cfg)
    explicit_actions = resolve_explicit_actions(args, cfg)

    fallback_device = int(cfg.get("device", 0))
    if args.device is not None:
        gpu_ids = [int(args.device)]
    else:
        gpu_ids = parse_gpu_list(args.gpus, fallback=fallback_device)
    num_workers = args.num_workers if args.num_workers is not None else len(gpu_ids)
    num_workers = max(int(num_workers), 1)

    action_count, train_char, source_mode = discover_action_count(
        cfg,
        min_frames=min_frames,
        explicit_actions=explicit_actions,
        require_shape=(retarget_mode != "arp"),
    )
    stage2 = cfg.get("stage2", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    blend_output_dir = Path(cfg.get("blend_output_dir", "./visualization/blend"))
    if not blend_output_dir.is_absolute():
        blend_output_dir = (_REPO_ROOT / blend_output_dir).resolve()

    # Single dog + multiple GPUs -> shard actions; else parallelize by dog.
    use_action_sharding = len(dog_ids) == 1 and len(gpu_ids) > 1
    concurrent_workers = (
        len(gpu_ids)
        if use_action_sharding
        else min(num_workers, len(dog_ids), len(gpu_ids))
    )
    cpu_threads_per_worker = resolve_cpu_threads_per_worker(args, concurrent_workers)
    blender_threads = resolve_blender_threads(args, cpu_threads_per_worker)

    if retarget_mode in ("direct", "arp"):
        weights_preview = None
    else:
        mode = str(stage2.get("mode", "blend")).strip().lower()
        if mode in ("stage1", "skel", "skeleton", "base"):
            weights_preview = model_cfg.get("stage1_weights")
        else:
            weights_preview = model_cfg.get("stage2_weights")

    print(
        json.dumps(
            {
                "dog_ids": dog_ids,
                "retarget_mode": retarget_mode,
                "source_mode": source_mode,
                "approx_source_clips": action_count,
                "explicit_actions": explicit_actions if source_mode == "explicit" else None,
                "train_char": str(train_char),
                "min_frames": min_frames if source_mode == "scan" else None,
                "stage2_mode": (
                    None
                    if retarget_mode in ("direct", "arp")
                    else stage2.get("mode", "blend")
                ),
                "gate_scale": (
                    None
                    if retarget_mode in ("direct", "arp")
                    else stage2.get("gate_scale", 1.0)
                ),
                "weights": weights_preview,
                "gpus": gpu_ids,
                "num_workers": num_workers,
                "concurrent_workers": concurrent_workers,
                "cpu_threads_per_worker": cpu_threads_per_worker,
                "total_cpu_threads_budget": (
                    None
                    if cpu_threads_per_worker is None
                    else cpu_threads_per_worker * concurrent_workers
                ),
                "blender_threads": blender_threads,
                "parallel_mode": "action_shards" if use_action_sharding else "per_dog",
                "blender": args.blender,
                "arp_addon_modules": (
                    list(args.arp_addon_modules) if retarget_mode == "arp" else None
                ),
                "pack_mode": (cfg.get("blend", {}) or {}).get("pack_mode", "armature"),
                "skip_bvh_export": bool(args.skip_bvh_export),
                "blend_output_dir": str(blend_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return

    work_dir = args.work_dir
    temp_dir = None
    if work_dir is None:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="r2et_dog_blend_", dir=str(_REPO_ROOT / "temp"))
        )
        work_dir = temp_dir / "work"
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    blend_output_dir.mkdir(parents=True, exist_ok=True)

    if use_action_sharding:
        produced_ids: list[str] = []
        failed_ids: list[str] = []
        dog_id, ok, _err = process_single_dog_sharded(
            dog_ids[0],
            gpu_ids,
            args,
            cfg_path,
            work_dir,
            blend_output_dir,
            cpu_threads_per_worker=cpu_threads_per_worker,
            blender_threads=blender_threads,
        )
        if ok:
            produced_ids.append(dog_id)
        else:
            failed_ids.append(dog_id)
    else:
        produced_ids, failed_ids = run_parallel_dogs(
            dog_ids,
            gpu_ids,
            min(num_workers, len(dog_ids), len(gpu_ids)),
            args,
            cfg_path,
            work_dir,
            blend_output_dir,
            cpu_threads_per_worker=cpu_threads_per_worker,
            blender_threads=blender_threads,
        )

    # Write combined index for convenience.
    manifests = []
    for dog_id in dog_ids:
        path = work_dir / dog_id / "dog_actions_manifest.json"
        if path.exists():
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
    if manifests and args.keep_intermediates:
        index_path = work_dir / "dog_actions_manifest_index.json"
        index_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[dog-blend] done: {len(produced_ids)} blends -> {blend_output_dir}")
    for dog_id in dog_ids:
        path = blend_output_dir / f"{dog_id}.blend"
        print(f"  {'OK' if path.exists() else 'MISSING'}  {path}")

    if not args.keep_intermediates and temp_dir is not None and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    if failed_ids:
        log(f"[dog-blend][warn] failed: {', '.join(failed_ids)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
