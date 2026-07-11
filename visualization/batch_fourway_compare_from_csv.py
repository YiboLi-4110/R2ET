#!/usr/bin/env python3
"""
Batch four-way compare videos from a CSV of (action, dog_id) pairs.

Reads rows like temp/clipping_bad_dogs.csv, resolves shepherd/batch2_dogs assets,
runs ARP -> export -> LBS render in parallel (multi-thread + multi-GPU), and keeps
only final MP4 outputs.

Example:
  python visualization/batch_fourway_compare_from_csv.py \
    --csv temp/clipping_bad_dogs.csv \
    --config config/visualization_compare_smal33.yaml \
    --video_output_dir visualization/videos/compare/demos \
    --gpus 0,1,2,3 \
    --num_workers 4 \
    --cpu_threads_per_worker 8

  python visualization/batch_fourway_compare_from_csv.py --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_assets import resolve_fbx_from_bvh

DEFAULT_CONFIG = _REPO_ROOT / "config/visualization_compare_smal33.yaml"
DEFAULT_CSV = _REPO_ROOT / "temp/clipping_bad_dogs.csv"
DEFAULT_VIDEO_DIR = _REPO_ROOT / "visualization/videos/compare/demos"
FINAL_VIDEO_NAME = "fourway_compare_lbs.mp4"

_print_lock = threading.Lock()


@dataclass
class ResolvedCase:
    case_id: str
    action: str
    dog_id: str
    inp_bvh_path: str
    tgt_bvh_path: str
    inp_shape_path: str
    tgt_shape_path: str

    def as_case_cfg(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "inp_bvh_path": self.inp_bvh_path,
            "tgt_bvh_path": self.tgt_bvh_path,
            "inp_shape_path": self.inp_shape_path,
            "tgt_shape_path": self.tgt_shape_path,
            "arp_bvh_path": None,
        }


@dataclass
class BatchRuntime:
    repo_root: Path
    script_dir: Path
    blender: str
    arp_addon_modules: list[str]
    render_engine: str
    keep_intermediates: bool
    cpu_threads_per_worker: int | None
    blender_threads: int | None


def log(message: str):
    with _print_lock:
        print(message, flush=True)


def parse_gpu_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("GPU list must not be empty.")
    return [int(v) for v in values]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch four-way compare videos from action/dog_id CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video_output_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--work_dir", type=Path, default=None)
    default_blender = os.environ.get("BLENDER", "blender")
    parser.add_argument("--blender", type=str, default=default_blender)
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
    )
    parser.add_argument("--render_engine", type=str, default="eevee", choices=["eevee", "cycles"])
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated physical GPU ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Parallel case workers. Default: number of GPUs.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Deprecated single-GPU override. Use --gpus instead.",
    )
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=None,
        help="CPU threads for each worker subprocess (OMP/MKL/OpenBLAS/NUMEXPR).",
    )
    parser.add_argument(
        "--total_cpu_threads",
        type=int,
        default=None,
        help="Total CPU thread budget; per-worker = total // num_workers when "
        "cpu_threads_per_worker is unset.",
    )
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=None,
        help="Optional Blender --threads value for ARP/render subprocesses.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N valid cases.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Keep npz/manifest/bvh/work dirs after rendering.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_csv_rows(csv_path: Path) -> list[tuple[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "action" not in reader.fieldnames or "dog_id" not in reader.fieldnames:
            raise ValueError(f"CSV must contain action,dog_id columns: {csv_path}")
        for line_no, row in enumerate(reader, start=2):
            action = (row.get("action") or "").strip()
            dog_id = (row.get("dog_id") or "").strip()
            if not action or not dog_id:
                log(f"[batch][skip] line {line_no}: empty action or dog_id")
                continue
            key = (action, dog_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    return rows


def shepherd_roots(shepherd_root: Path) -> dict[str, Path]:
    return {
        "train_char": shepherd_root / "smal@shepherd/train_char",
        "train_shape": shepherd_root / "smal@shepherd/train_shape",
        "batch2_char": shepherd_root / "batch2_dogs/batch2_dogs_char",
        "batch2_shape": shepherd_root / "batch2_dogs/batch2_dogs_shape",
    }


def action_shape_name(action: str) -> str | None:
    match = re.match(r"^(shepherd@[^_]+)", action)
    return match.group(1) if match else None


def find_inp_bvh(action: str, train_char: Path) -> Path | None:
    if not train_char.is_dir():
        return None
    matches = sorted(train_char.rglob(f"{action}.bvh"))
    if not matches:
        return None
    if len(matches) > 1:
        log(f"[batch][warn] action '{action}' matched multiple BVH files; using {matches[0]}")
    return matches[0]


def find_tgt_bvh(dog_id: str, batch2_char: Path) -> Path | None:
    if "_" not in dog_id:
        return None
    breed = dog_id.rsplit("_", 1)[0]
    for candidate in (f"{dog_id}-0.bvh", f"{dog_id}.bvh"):
        bvh_path = batch2_char / breed / candidate
        if bvh_path.exists():
            return bvh_path
    return None


def find_tgt_shape(dog_id: str, batch2_shape: Path) -> Path | None:
    for candidate in (f"{dog_id}-0.npz", f"{dog_id}.npz"):
        shape_path = batch2_shape / candidate
        if shape_path.exists():
            return shape_path
    return None


def make_case_id(action: str, dog_id: str) -> str:
    action_part = action
    if action_part.startswith("shepherd@"):
        action_part = action_part[len("shepherd@") :]
    action_part = action_part.replace("_smal_dog-foot_on_ground", "")
    raw = f"{action_part}_to_{dog_id}"
    slug = re.sub(r'[\\/:*?"<>|]+', "_", raw)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug or "case"


def resolve_case(
    action: str,
    dog_id: str,
    roots: dict[str, Path],
    assets_cfg: dict[str, Any],
) -> tuple[ResolvedCase | None, str | None]:
    train_char = roots["train_char"]
    train_shape = roots["train_shape"]
    batch2_char = roots["batch2_char"]
    batch2_shape = roots["batch2_shape"]

    if not train_char.is_dir():
        return None, f"action '{action}': train_char not found under datasets/shepherd ({train_char})"

    inp_bvh = find_inp_bvh(action, train_char)
    if inp_bvh is None:
        return None, f"action '{action}': BVH not found under {train_char}"

    shape_name = action_shape_name(action)
    if shape_name is None:
        return None, f"action '{action}': cannot infer shape npz name from action"
    inp_shape = train_shape / f"{shape_name}.npz"
    if not inp_shape.exists():
        return None, f"action '{action}': shape npz not found ({inp_shape})"

    if not batch2_char.is_dir():
        return None, f"dog_id '{dog_id}': batch2_dogs_char not found under datasets/shepherd ({batch2_char})"

    tgt_bvh = find_tgt_bvh(dog_id, batch2_char)
    if tgt_bvh is None:
        breed = dog_id.rsplit("_", 1)[0] if "_" in dog_id else dog_id
        return None, f"dog_id '{dog_id}': target BVH not found under {batch2_char / breed}"

    tgt_shape = find_tgt_shape(dog_id, batch2_shape)
    if tgt_shape is None:
        return None, f"dog_id '{dog_id}': target shape npz not found under {batch2_shape}"

    recursive = bool(assets_cfg.get("recursive_fbx_search", False))
    try:
        resolve_fbx_from_bvh(inp_bvh, recursive_search=recursive)
        resolve_fbx_from_bvh(tgt_bvh, recursive_search=recursive)
    except FileNotFoundError as exc:
        return None, f"{action} + {dog_id}: {exc}"

    case_id = make_case_id(action, dog_id)
    return (
        ResolvedCase(
            case_id=case_id,
            action=action,
            dog_id=dog_id,
            inp_bvh_path=str(inp_bvh.resolve()),
            tgt_bvh_path=str(tgt_bvh.resolve()),
            inp_shape_path=str(inp_shape.resolve()),
            tgt_shape_path=str(tgt_shape.resolve()),
        ),
        None,
    )


def build_cases(
    rows: list[tuple[str, str]],
    shepherd_root: Path,
    assets_cfg: dict[str, Any],
    limit: int | None,
) -> tuple[list[ResolvedCase], list[str]]:
    roots = shepherd_roots(shepherd_root)
    resolved: list[ResolvedCase] = []
    skipped: list[str] = []

    for action, dog_id in rows:
        case, err = resolve_case(action, dog_id, roots, assets_cfg)
        if case is None:
            skipped.append(err or f"{action} + {dog_id}: unknown error")
            log(f"[batch][skip] {err}")
            continue
        resolved.append(case)
        log(f"[batch][ok] {case.case_id}")
        if limit is not None and len(resolved) >= limit:
            break

    return resolved, skipped


def resolve_cpu_threads_per_worker(args, num_workers: int) -> int | None:
    if args.cpu_threads_per_worker is not None:
        return max(int(args.cpu_threads_per_worker), 1)
    if args.total_cpu_threads is not None:
        return max(int(args.total_cpu_threads) // max(num_workers, 1), 1)
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
    ):
        env[key] = thread_value
    return env


def blender_prefix(blender: str, blender_threads: int | None) -> list[str]:
    cmd = [blender]
    if blender_threads is not None and blender_threads > 0:
        cmd.extend(["--threads", str(blender_threads)])
    return cmd


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    step: str,
    case_id: str,
    gpu_id: int | None,
    env: dict[str, str],
):
    gpu_tag = f" gpu={gpu_id}" if gpu_id is not None else ""
    log(f"[batch][run][{case_id}]{gpu_tag} {step}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def build_case_config(
    base_cfg: dict[str, Any],
    case: ResolvedCase,
    work_dir: Path,
    arp_dir: Path,
    arp_mesh_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["output_root"] = str(work_dir)
    cfg["cases"] = [case.as_case_cfg()]
    cfg["device"] = int(gpu_id)
    cfg["arp"] = dict(cfg.get("arp", {}) or {})
    cfg["arp"]["output_dir"] = str(arp_dir)
    cfg["arp"]["mesh_output_dir"] = str(arp_mesh_dir)
    return cfg


def collect_and_cleanup_case(
    case: ResolvedCase,
    work_dir: Path,
    arp_dir: Path,
    arp_mesh_dir: Path,
    arp_suffix: str,
    video_output_dir: Path,
    keep_intermediates: bool,
) -> Path | None:
    case_dir = work_dir / case.case_id
    src_video = case_dir / FINAL_VIDEO_NAME
    if not src_video.exists():
        log(f"[batch][error][{case.case_id}] missing final video: {src_video}")
        return None

    video_output_dir.mkdir(parents=True, exist_ok=True)
    dest_video = video_output_dir / f"{case.case_id}.mp4"
    shutil.copy2(src_video, dest_video)
    log(f"[batch][video][{case.case_id}] {dest_video}")

    if keep_intermediates:
        return dest_video

    arp_bvh = arp_dir / f"{case.case_id}{arp_suffix}"
    arp_mesh = arp_mesh_dir / f"{case.case_id}_arp_mesh.npz"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    if arp_bvh.exists():
        arp_bvh.unlink()
    if arp_mesh.exists():
        arp_mesh.unlink()
    return dest_video


def process_one_case(
    case: ResolvedCase,
    gpu_id: int,
    worker_id: int,
    base_cfg: dict[str, Any],
    work_dir: Path,
    video_output_dir: Path,
    runtime: BatchRuntime,
    arp_suffix: str,
) -> tuple[str, bool, str | None]:
    case_cfg_dir = work_dir / "_configs"
    case_cfg_dir.mkdir(parents=True, exist_ok=True)
    arp_dir = work_dir / "arp_outputs"
    arp_dir.mkdir(parents=True, exist_ok=True)
    arp_mesh_dir = work_dir / "arp_mesh_outputs"
    arp_mesh_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = case_cfg_dir / f"{case.case_id}.yaml"
    case_cfg = build_case_config(base_cfg, case, work_dir, arp_dir, arp_mesh_dir, gpu_id)
    write_yaml(cfg_path, case_cfg)

    env = subprocess_env(runtime.cpu_threads_per_worker)
    manifest_index = work_dir / f"{case.case_id}_manifest_index.json"

    try:
        run_command(
            [
                *blender_prefix(runtime.blender, runtime.blender_threads),
                "--background",
                "--python",
                str(runtime.script_dir / "arp_export_mesh_blender.py"),
                "--",
                "--config",
                str(cfg_path),
                "--case_ids",
                case.case_id,
                "--arp_addon_modules",
                *runtime.arp_addon_modules,
            ],
            cwd=runtime.repo_root,
            step="arp",
            case_id=case.case_id,
            gpu_id=gpu_id,
            env=env,
        )
        run_command(
            [
                sys.executable,
                str(runtime.script_dir / "export_fourway_compare_smal33.py"),
                "--config",
                str(cfg_path),
                "--case_ids",
                case.case_id,
                "--device",
                str(gpu_id),
            ],
            cwd=runtime.repo_root,
            step="export",
            case_id=case.case_id,
            gpu_id=gpu_id,
            env=env,
        )

        case_manifest = work_dir / case.case_id / "fourway_manifest.json"
        if not case_manifest.exists():
            raise FileNotFoundError(f"missing manifest: {case_manifest}")
        manifest_index.write_text(
            json.dumps([json.loads(case_manifest.read_text(encoding="utf-8"))], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        run_command(
            [
                *blender_prefix(runtime.blender, runtime.blender_threads),
                "--background",
                "--python",
                str(runtime.script_dir / "render_fourway_lbs_blender.py"),
                "--",
                "--manifest_index",
                str(manifest_index),
                "--render_engine",
                runtime.render_engine,
            ],
            cwd=runtime.repo_root,
            step="render-lbs",
            case_id=case.case_id,
            gpu_id=gpu_id,
            env=env,
        )

        dest = collect_and_cleanup_case(
            case,
            work_dir,
            arp_dir,
            arp_mesh_dir,
            arp_suffix,
            video_output_dir,
            runtime.keep_intermediates,
        )
        if dest is None:
            return case.case_id, False, "final video missing after render"

        if not runtime.keep_intermediates and manifest_index.exists():
            manifest_index.unlink()
        if not runtime.keep_intermediates and cfg_path.exists():
            cfg_path.unlink()

        log(f"[batch][done][worker={worker_id} gpu={gpu_id}] {case.case_id}")
        return case.case_id, True, None
    except Exception as exc:
        log(f"[batch][fail][worker={worker_id} gpu={gpu_id}] {case.case_id}: {exc}")
        return case.case_id, False, str(exc)


def run_parallel_batch(
    cases: list[ResolvedCase],
    base_cfg: dict[str, Any],
    work_dir: Path,
    video_output_dir: Path,
    runtime: BatchRuntime,
    gpu_ids: list[int],
    num_workers: int,
) -> tuple[list[str], list[str]]:
    arp_suffix = (base_cfg.get("arp", {}) or {}).get("output_suffix", "_arp_retarget.bvh")
    gpu_queue: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    produced: list[str] = []
    failed: list[str] = []

    def _run_case(case: ResolvedCase) -> tuple[str, bool, str | None]:
        gpu_id = gpu_queue.get()
        worker_id = gpu_id
        try:
            return process_one_case(
                case,
                gpu_id,
                worker_id,
                base_cfg,
                work_dir,
                video_output_dir,
                runtime,
                arp_suffix,
            )
        finally:
            gpu_queue.put(gpu_id)

    log(
        f"[batch] parallel run: cases={len(cases)} workers={num_workers} "
        f"gpus={gpu_ids} cpu_threads_per_worker={runtime.cpu_threads_per_worker} "
        f"blender_threads={runtime.blender_threads}"
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_run_case, case): case for case in cases}
        for future in as_completed(futures):
            case_id, ok, _err = future.result()
            if ok:
                produced.append(case_id)
            else:
                failed.append(case_id)

    return produced, failed


def cleanup_batch_workspace(
    work_dir: Path,
    temp_cfg_dir: Path | None,
    keep_intermediates: bool,
):
    if keep_intermediates:
        return

    cfg_dir = work_dir / "_configs"
    if cfg_dir.exists():
        shutil.rmtree(cfg_dir, ignore_errors=True)

    arp_dir = work_dir / "arp_outputs"
    if arp_dir.exists() and not any(arp_dir.iterdir()):
        arp_dir.rmdir()

    for path in work_dir.glob("*_manifest_index.json"):
        path.unlink(missing_ok=True)

    if temp_cfg_dir is not None and temp_cfg_dir.exists():
        shutil.rmtree(temp_cfg_dir, ignore_errors=True)

    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()


def main():
    args = parse_args()
    csv_path = args.csv.resolve()
    base_cfg = load_yaml(args.config.resolve())
    shepherd_root = (_REPO_ROOT / "datasets/shepherd").resolve()
    assets_cfg = base_cfg.get("assets", {}) or {}

    gpu_ids = parse_gpu_list(args.gpus)
    if args.device is not None:
        gpu_ids = [int(args.device)]

    num_workers = args.num_workers if args.num_workers is not None else len(gpu_ids)
    num_workers = max(int(num_workers), 1)
    cpu_threads_per_worker = resolve_cpu_threads_per_worker(args, num_workers)

    rows = read_csv_rows(csv_path)
    cases, skipped = build_cases(rows, shepherd_root, assets_cfg, args.limit)
    if not cases:
        raise SystemExit(f"No valid cases resolved from CSV. Skipped {len(skipped)} rows.")

    print(f"[batch] valid cases: {len(cases)}, skipped: {len(skipped)}")
    if args.dry_run:
        summary = {
            "valid_cases": [c.case_id for c in cases],
            "skipped": skipped,
            "num_workers": num_workers,
            "gpus": gpu_ids,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "blender_threads": args.blender_threads,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    work_dir = args.work_dir
    temp_cfg_dir: Path | None = None
    if work_dir is None:
        temp_cfg_dir = Path(tempfile.mkdtemp(prefix="fourway_batch_", dir=str(_REPO_ROOT / "temp")))
        work_dir = temp_cfg_dir / "work"
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    runtime = BatchRuntime(
        repo_root=_REPO_ROOT,
        script_dir=_SCRIPT_DIR,
        blender=args.blender,
        arp_addon_modules=args.arp_addon_modules,
        render_engine=args.render_engine,
        keep_intermediates=args.keep_intermediates,
        cpu_threads_per_worker=cpu_threads_per_worker,
        blender_threads=args.blender_threads,
    )
    video_output_dir = args.video_output_dir.resolve()

    produced, failed = run_parallel_batch(
        cases,
        base_cfg,
        work_dir,
        video_output_dir,
        runtime,
        gpu_ids,
        num_workers,
    )

    cleanup_batch_workspace(work_dir, temp_cfg_dir, args.keep_intermediates)

    log(f"[batch] done: {len(produced)} videos -> {video_output_dir}")
    if failed:
        log(f"[batch][warn] failed cases ({len(failed)}): {', '.join(sorted(failed))}")
    if skipped:
        log(f"[batch][warn] skipped rows: {len(skipped)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
