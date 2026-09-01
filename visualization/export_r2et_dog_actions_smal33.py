#!/usr/bin/env python3
"""
Export retarget results for ALL shepherd train_char actions onto one dog.

Supports:
  retarget_mode=r2et   -> stage1 / stage2 / blend RetNet inference
  retarget_mode=direct -> CopyQuat (same-skeleton; fourway-compatible)
  retarget_mode=arp    -> Auto-Rig Pro in Blender (cross-skeleton; same pack layout)

Primary pack inputs:
  armature: <action_id>_ours_fbx_bake.npz (+ native BVH for debug)
  lbs:      <action_id>_ours_lbs.npz
  manifest: <output_root>/<dog_id>/dog_actions_manifest.json

Optional model-space BVH (disabled by default via blend.skip_bvh_export).

Example:
  python visualization/export_r2et_dog_actions_smal33.py \\
    --config config/visualization_blend_per_dog_smal33.yaml \\
    --dog_ids 博美_3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_assets import resolve_fbx_from_bvh
from direct_copy_mesh import compute_copyquat_outputs
from fbx_bake_cache import write_fbx_bake_cache_from_model
from arp_sequence_common import blender_z_up_to_lbs_y_up

from datasets.lbs_runtime import skin_mesh_sequence
from datasets.smal33_motion_io import (
    SMAL33_PARENTS,
    build_model_inputs,
    get_inp_from_bvh,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
    load_retnet,
    load_shape_retnet,
    load_shape_vector,
    load_stats,
    report_lbs_rest_skel_mismatch,
    retarget_to_bvh,
    retarget_to_native_bvh,
    setup_cuda_device,
)

REPO_ROOT = _PROJECT_ROOT
VALID_RETARGET_MODES = ("r2et", "direct", "arp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export all shepherd actions retargeted onto target dog_id(s)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/visualization_blend_per_dog_smal33.yaml",
    )
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--dog_ids",
        type=str,
        nargs="+",
        default=None,
        help="Override config targets.dog_ids.",
    )
    parser.add_argument(
        "--stage2_mode",
        type=str,
        default=None,
        choices=["blend", "stage1", "stage2"],
    )
    parser.add_argument("--gate_scale", type=float, default=None)
    parser.add_argument(
        "--stage1_weights",
        type=Path,
        default=None,
        help="Override model.stage1_weights (skeleton-aware RetNet).",
    )
    parser.add_argument(
        "--stage2_weights",
        type=Path,
        default=None,
        help="Override model.stage2_weights (shape-aware RetNet).",
    )
    parser.add_argument("--limit_actions", type=int, default=None)
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Action shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split source actions across this many shards (multi-GPU).",
    )
    parser.add_argument(
        "--skip_bvh_export",
        action="store_true",
        default=False,
        help="Skip writing Ours BVH (LBS npz is enough for blend packing).",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=None,
        help=(
            "Skip source BVH clips with fewer than this many frames "
            "(overrides source.min_frames in config; 0 = no filter). "
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
            "Each entry: BVH path (absolute / repo-relative / relative to "
            "train_char), or Folder/stem, or stem looked up under train_char. "
            "Overrides source.actions in config when provided."
        ),
    )
    parser.add_argument(
        "--retarget_mode",
        type=str,
        default=None,
        choices=list(VALID_RETARGET_MODES),
        help="Override retarget_mode (r2et | direct/CopyQuat | arp).",
    )
    parser.add_argument(
        "--blender",
        type=str,
        default=os.environ.get("BLENDER", "blender"),
        help="Blender binary for retarget_mode=arp.",
    )
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
        help="ARP addon module names (retarget_mode=arp).",
    )
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=None,
        help="Optional Blender --threads for retarget_mode=arp.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help=(
            "Cap intra-op CPU threads for this process "
            "(OMP/MKL/OpenBLAS/NUMEXPR + torch). "
            "Usually set by batch_r2et_dog_blend.py via --cpu_threads_per_worker."
        ),
    )
    return parser.parse_args()


def apply_cpu_thread_limit(cpu_threads: int | None) -> None:
    """Limit BLAS/OpenMP/torch thread pools for this process."""
    if cpu_threads is None:
        return
    n = max(int(cpu_threads), 1)
    thread_value = str(n)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[key] = thread_value
    try:
        import torch

        torch.set_num_threads(n)
        if hasattr(torch, "set_num_interop_threads"):
            # Interop threads must be set before parallel work starts; ignore late failures.
            try:
                torch.set_num_interop_threads(max(min(n, 2), 1))
            except RuntimeError:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"[export-dog][warn] failed to set torch CPU threads: {exc}", flush=True)
    print(f"[export-dog] cpu_threads={n}", flush=True)


def resolve_retarget_mode(cfg: dict[str, Any], override: str | None = None) -> str:
    raw = override if override is not None else cfg.get("retarget_mode", "r2et")
    mode = str(raw).strip().lower()
    if mode in ("copyquat", "copy_quat", "quat"):
        mode = "direct"
    if mode not in VALID_RETARGET_MODES:
        raise SystemExit(
            f"Unknown retarget_mode={raw!r}; expected one of {VALID_RETARGET_MODES}."
        )
    return mode


def normalize_local_for_stats(local_phys: np.ndarray, stats: dict) -> np.ndarray:
    """Map physical-space local joints to stats-normalized space (R2ET convention)."""
    local_mean = np.asarray(stats["local_mean"], dtype=np.float32)
    local_std = np.asarray(stats["local_std"], dtype=np.float32)
    return (np.asarray(local_phys, dtype=np.float32) - local_mean) / np.maximum(
        local_std, 1e-8
    )


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def motion_parse_options(motion_cfg: dict[str, Any], prefix: str):
    return {
        "axis_transform": motion_cfg.get(f"{prefix}_axis_transform", "none"),
        "forward_mode": motion_cfg.get(f"{prefix}_forward_mode", "body"),
        "post_axis_yaw_deg": float(motion_cfg.get(f"{prefix}_post_axis_yaw_deg", 0.0) or 0.0),
        "canonicalize_bind_pose": bool(
            motion_cfg.get(f"{prefix}_canonicalize_bind_pose", True)
        ),
    }


def mesh_load_options(motion_cfg: dict[str, Any], prefix: str):
    return {
        "canonicalize_bind_pose": bool(
            motion_cfg.get(f"{prefix}_canonicalize_bind_pose", True)
        ),
        "forward_mode": motion_cfg.get(f"{prefix}_forward_mode", "body"),
    }


def resolve_stage2_inference_knobs(stage2_cfg: dict[str, Any] | None):
    """Map stage2 config to (gate_scale, force_gate_ones, mode_label).

    Modes:
      stage1 -> pure skeleton RetNet (weights chosen by loader; gate unused)
      stage2 -> shape RetNet with force_gate_ones, k=gate_scale
      blend  -> shape RetNet with learned gate * gate_scale
    """
    stage2_cfg = stage2_cfg or {}
    mode = str(stage2_cfg.get("mode", "blend")).strip().lower()
    gate_scale = float(stage2_cfg.get("gate_scale", 1.0))
    if mode in ("stage1", "skel", "skeleton", "base"):
        return 0.0, False, "stage1"
    if mode in ("stage2", "shape", "full", "hat"):
        # Full shape residual path; gate_scale still scales qB_hat vs qB_base.
        return gate_scale, True, "stage2"
    if mode not in ("blend", "balanced", "auto", ""):
        print(f"[export-dog][warn] unknown stage2.mode={mode!r}, fallback to blend")
    return gate_scale, False, "blend"


@torch.no_grad()
def run_stage1_inference(
    model,
    inp_motion,
    tgt_motion,
    inp_shape,
    tgt_shape,
    stats,
    device,
):
    """Pure skeleton-aware RetNet (stage-1)."""
    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)
    local_b, global_b, quat_b, _ = model(
        inp_batch["seq"],
        None,
        inp_batch["skel"],
        tgt_batch["skel"],
        inp_batch["shape"],
        tgt_batch["shape"],
        inp_batch["quat"],
        inp_batch["height"],
        tgt_batch["height"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        SMAL33_PARENTS,
    )
    return (
        local_b[0].cpu().numpy(),
        global_b[0].cpu().numpy(),
        quat_b[0].cpu().numpy(),
    )


@torch.no_grad()
def run_stage2_inference(
    model,
    inp_motion,
    tgt_motion,
    inp_shape,
    tgt_shape,
    stats,
    device,
    gate_scale=1.0,
    force_gate_ones=False,
):
    """Shape-aware RetNet (stage-2 / blend)."""
    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)
    local_b, global_b, quat_b, _, _ = model(
        inp_batch["seq"],
        tgt_batch["seq"],
        inp_batch["skel"],
        tgt_batch["skel"],
        inp_batch["shape"],
        tgt_batch["shape"],
        inp_batch["quat"],
        inp_batch["height"],
        tgt_batch["height"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        SMAL33_PARENTS,
        k=float(gate_scale),
        phase="test",
        force_gate_ones=bool(force_gate_ones),
    )
    return (
        local_b[0].cpu().numpy(),
        global_b[0].cpu().numpy(),
        quat_b[0].cpu().numpy(),
    )


def run_retarget_inference(
    model,
    *,
    mode: str,
    inp_motion,
    tgt_motion,
    inp_shape,
    tgt_shape,
    stats,
    device,
    gate_scale: float = 1.0,
    force_gate_ones: bool = False,
):
    if mode == "stage1":
        return run_stage1_inference(
            model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device
        )
    return run_stage2_inference(
        model,
        inp_motion,
        tgt_motion,
        inp_shape,
        tgt_shape,
        stats,
        device,
        gate_scale=gate_scale,
        force_gate_ones=force_gate_ones,
    )


def load_inference_model(mode: str, model_cfg: dict[str, Any], ret_model_args, device):
    """Load pure stage-1 skeleton RetNet or stage-2 shape RetNet by mode."""
    stats_path = model_cfg.get("stats_path")
    if not stats_path:
        raise SystemExit("Missing model.stats_path.")

    if mode == "stage1":
        weights = model_cfg.get("stage1_weights")
        if not weights:
            raise SystemExit(
                "stage2.mode=stage1 requires model.stage1_weights "
                "(pure skeleton-aware checkpoint)."
            )
        print(f"[export-dog] loading stage1 RetNet: {weights}")
        model = load_retnet(weights, ret_model_args, device)
        return model, str(weights)

    weights = model_cfg.get("stage2_weights")
    if not weights:
        raise SystemExit(
            f"stage2.mode={mode} requires model.stage2_weights "
            "(shape-aware checkpoint)."
        )
    print(f"[export-dog] loading stage2 RetNet: {weights}")
    model = load_shape_retnet(weights, ret_model_args, device)
    return model, str(weights)


def align_frames(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if len(arr) == num_frames:
        return arr
    if len(arr) <= 1:
        return np.repeat(arr, num_frames, axis=0)
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(np.int64)
    return arr[idx]


def shard_actions(actions: list[dict[str, Any]], shard_index: int, num_shards: int):
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    if num_shards == 1:
        return actions
    return [a for i, a in enumerate(actions) if i % num_shards == shard_index]


def action_shape_name(action_stem: str) -> str | None:
    """Legacy shepherd stem helper: shepherd@Attack_F_RM... -> shepherd@Attack."""
    match = re.match(r"^(shepherd@[^_]+)", action_stem)
    return match.group(1) if match else None


def resolve_inp_shape_path(
    action_stem: str,
    folder_name: str,
    train_shape: Path,
) -> Path | None:
    """
    Resolve source shape .npz for a clip.

    Mirrors datasets/train_feeder_r2et_smal33.Feeder._resolve_shape_key:
    try sequence name, then folder name. Also keeps shepherd@Xxx stem mapping
    and common species prefixes (e.g. sand_cat_juvenile@... -> sand_cat.npz).
    """
    candidates: list[str] = []
    legacy = action_shape_name(action_stem)
    if legacy:
        candidates.append(legacy)
    candidates.append(action_stem)
    candidates.append(folder_name)
    if "@" in action_stem:
        prefix = action_stem.split("@", 1)[0]
        candidates.append(prefix)
        for suffix in ("_juvenile", "_adult", "_female", "_male"):
            if prefix.endswith(suffix):
                candidates.append(prefix[: -len(suffix)])
                break

    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = train_shape / f"{name}.npz"
        if path.is_file():
            return path
    return None


def make_action_id(action_stem: str) -> str:
    name = action_stem
    if name.startswith("shepherd@"):
        name = name[len("shepherd@") :]
    name = name.replace("@", "_")
    name = name.replace("_smal_dog-foot_on_ground", "")
    slug = re.sub(r'[\\/:*?"<>|]+', "_", name)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug or "action"


def find_tgt_bvh(dog_id: str, batch2_char: Path) -> Path | None:
    if "_" not in dog_id:
        return None
    breed = dog_id.rsplit("_")[0]
    for candidate in (f"{dog_id}-0.bvh", f"{dog_id}.bvh"):
        path = batch2_char / breed / candidate
        if path.exists():
            return path
    return None


def find_tgt_shape(dog_id: str, batch2_shape: Path) -> Path | None:
    for candidate in (f"{dog_id}-0.npz", f"{dog_id}.npz"):
        path = batch2_shape / candidate
        if path.exists():
            return path
    return None


def read_bvh_frame_count(bvh_path: Path) -> int:
    """Parse the Frames: line from a BVH header without loading motion data."""
    frames_re = re.compile(r"^\s*Frames:\s+(\d+)\s*$")
    with bvh_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = frames_re.match(line)
            if match:
                return int(match.group(1))
    raise ValueError(f"Frames: header not found in {bvh_path}")


def normalize_action_specs(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        text = str(raw).strip()
        return [text] if text else []
    if not isinstance(raw, (list, tuple)):
        raise TypeError(
            f"source.actions must be a list of paths/stems, got {type(raw).__name__}"
        )
    specs: list[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            specs.append(text)
    return specs


def resolve_explicit_bvh_path(
    spec: str,
    train_char: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """
    Resolve one explicit source clip to a BVH path.

    Accepted forms:
      - absolute path to .bvh / .fbx
      - path relative to repo root
      - path relative to train_char (e.g. Attack/Attack_F_RM.bvh)
      - Folder/stem or stem (looked up under train_char)
    """
    repo_root = repo_root or REPO_ROOT
    raw = str(spec).strip()
    if not raw:
        raise FileNotFoundError("empty action spec")

    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((repo_root / path).resolve())
        candidates.append((train_char / path).resolve())

    # Stem / Folder/stem without suffix.
    if path.suffix.lower() not in {".bvh", ".fbx"}:
        rel = Path(raw)
        if len(rel.parts) == 1:
            # Unique stem search under train_char.
            matches = sorted(train_char.rglob(f"{rel.name}.bvh"))
            if not matches:
                matches = sorted(train_char.rglob(f"{rel.name}.fbx"))
            if len(matches) == 1:
                candidates.append(matches[0].resolve())
            elif len(matches) > 1:
                preview = ", ".join(str(m.relative_to(train_char)) for m in matches[:5])
                raise FileNotFoundError(
                    f"ambiguous action stem '{raw}' under {train_char}: {preview}"
                )
        else:
            candidates.append((train_char / rel.with_suffix(".bvh")).resolve())
            candidates.append((train_char / rel.with_suffix(".fbx")).resolve())

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            if cand.suffix.lower() == ".fbx":
                bvh = cand.with_suffix(".bvh")
                if not bvh.is_file():
                    raise FileNotFoundError(
                        f"found FBX for '{raw}' but missing sibling BVH: {bvh}"
                    )
                return bvh
            if cand.suffix.lower() == ".bvh":
                return cand
            raise FileNotFoundError(f"action spec is not a BVH/FBX file: {cand}")

    raise FileNotFoundError(
        f"cannot resolve action '{raw}' (tried under repo and {train_char})"
    )


def build_action_entry(
    bvh_path: Path,
    train_shape: Path,
    *,
    folder_name: str | None = None,
    num_frames: int | None = None,
    require_shape: bool = True,
) -> dict[str, Any] | None:
    bvh_path = Path(bvh_path).resolve()
    stem = bvh_path.stem
    folder = folder_name or bvh_path.parent.name
    shape_path = None
    if train_shape.is_dir():
        shape_path = resolve_inp_shape_path(stem, folder, train_shape)
    if shape_path is None and require_shape:
        print(
            f"[export-dog][skip] cannot resolve shape for {stem} "
            f"(folder={folder}, train_shape={train_shape})"
        )
        return None
    entry: dict[str, Any] = {
        "action_stem": stem,
        "action_id": make_action_id(stem),
        "folder": folder,
        "inp_bvh_path": bvh_path,
        "inp_shape_path": shape_path,
    }
    if num_frames is not None:
        entry["num_frames"] = int(num_frames)
    return entry


def load_explicit_source_actions(
    specs: list[str],
    train_char: Path,
    train_shape: Path,
    *,
    repo_root: Path | None = None,
    require_shape: bool = True,
) -> list[dict[str, Any]]:
    """Build action entries from an explicit list of clips (no min_frames filter)."""
    if not specs:
        return []
    if require_shape and not train_shape.is_dir():
        raise FileNotFoundError(f"train_shape not found: {train_shape}")

    actions: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for spec in specs:
        bvh_path = resolve_explicit_bvh_path(spec, train_char, repo_root=repo_root)
        if bvh_path in seen:
            print(f"[export-dog][skip] duplicate action: {bvh_path}")
            continue
        seen.add(bvh_path)
        folder_name = bvh_path.parent.name
        try:
            rel = bvh_path.resolve().relative_to(train_char.resolve())
            if len(rel.parts) >= 2:
                folder_name = rel.parts[0]
        except ValueError:
            pass
        entry = build_action_entry(
            bvh_path,
            train_shape,
            folder_name=folder_name,
            require_shape=require_shape,
        )
        if entry is None:
            continue
        actions.append(entry)

    print(
        f"[export-dog] explicit actions: requested={len(specs)}, "
        f"resolved={len(actions)}"
    )
    return actions


def discover_source_actions(
    train_char: Path,
    train_shape: Path,
    *,
    min_frames: int = 0,
    require_shape: bool = True,
) -> list[dict[str, Any]]:
    if not train_char.is_dir():
        raise FileNotFoundError(f"train_char not found: {train_char}")

    min_frames = max(int(min_frames), 0)
    actions: list[dict[str, Any]] = []
    skipped_short = 0
    for folder in sorted(p for p in train_char.iterdir() if p.is_dir() and not p.name.startswith(".")):
        bvhs = sorted(folder.glob("*.bvh"))
        if bvhs:
            stems_paths = [(p.stem, p) for p in bvhs]
        else:
            # Local machines may only have FBX; require matching BVH on disk for inference.
            stems_paths = [(p.stem, p.with_suffix(".bvh")) for p in sorted(folder.glob("*.fbx"))]

        for stem, bvh_path in stems_paths:
            num_frames = None
            if min_frames > 0:
                if not bvh_path.exists():
                    print(f"[export-dog][skip] missing BVH for frame check: {bvh_path}")
                    continue
                try:
                    num_frames = read_bvh_frame_count(bvh_path)
                except ValueError as exc:
                    print(f"[export-dog][skip] {exc}")
                    continue
                if num_frames < min_frames:
                    skipped_short += 1
                    print(
                        f"[export-dog][skip] too short ({num_frames}<{min_frames}): "
                        f"{bvh_path.name}"
                    )
                    continue

            entry = build_action_entry(
                bvh_path,
                train_shape,
                folder_name=folder.name,
                num_frames=num_frames,
                require_shape=require_shape,
            )
            if entry is None:
                continue
            actions.append(entry)

    if min_frames > 0:
        print(
            f"[export-dog] min_frames={min_frames}: kept {len(actions)}, "
            f"skipped_short={skipped_short}"
        )
    return actions


def select_source_actions(
    train_char: Path,
    train_shape: Path,
    *,
    min_frames: int = 0,
    explicit_actions: list[str] | None = None,
    repo_root: Path | None = None,
    require_shape: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """
    Choose source clips.

    If explicit_actions is non-empty -> use that list (scan/min_frames ignored).
    Else -> scan train_char with optional min_frames filter.
    Returns (actions, mode) where mode is 'explicit' or 'scan'.
    """
    specs = normalize_action_specs(explicit_actions)
    if specs:
        actions = load_explicit_source_actions(
            specs,
            train_char,
            train_shape,
            repo_root=repo_root,
            require_shape=require_shape,
        )
        return actions, "explicit"
    actions = discover_source_actions(
        train_char, train_shape, min_frames=min_frames, require_shape=require_shape
    )
    return actions, "scan"


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _blender_prefix(blender: str, blender_threads: int | None) -> list[str]:
    cmd = [blender]
    if blender_threads is not None and int(blender_threads) > 0:
        cmd.extend(["--threads", str(int(blender_threads))])
    return cmd


def _convert_arp_mesh_to_ours_lbs(raw_path: Path, out_path: Path) -> tuple[int, int]:
    """Blender Z-up ARP mesh -> pack-compatible ours_lbs.npz (LBS Y-up)."""
    payload = np.load(str(raw_path), allow_pickle=True)
    if "vertices" not in payload.files:
        raise KeyError(f"ARP mesh cache missing 'vertices': {raw_path}")
    verts = np.asarray(payload["vertices"], dtype=np.float32)
    faces = (
        np.asarray(payload["faces"], dtype=np.int32)
        if "faces" in payload.files
        else np.zeros((0, 3), dtype=np.int32)
    )
    space = str(payload["space"]) if "space" in payload.files else "blender_world_z_up"
    if space != "lbs_y_up":
        verts = blender_z_up_to_lbs_y_up(verts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ours_vertices=verts.astype(np.float32),
        target_faces=faces.astype(np.int32),
        num_frames=np.int32(len(verts)),
    )
    return int(len(verts)), int(verts.shape[1]) if verts.ndim == 3 else 0


def _run_arp_blender_export(
    *,
    runtime_cfg_path: Path,
    blender: str,
    blender_threads: int | None,
    arp_addon_modules: list[str],
) -> None:
    cmd = [
        *_blender_prefix(blender, blender_threads),
        "--background",
        "--python",
        str(_SCRIPT_DIR / "arp_export_mesh_blender.py"),
        "--",
        "--config",
        str(runtime_cfg_path),
        "--arp_addon_modules",
        *list(arp_addon_modules or ["auto_rig_pro-master", "auto_rig_pro"]),
    ]
    print(f"[export-dog][run] arp-blender: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def resolve_pack_mode(cfg: dict[str, Any]) -> str:
    pack_mode = str((cfg.get("blend", {}) or {}).get("pack_mode", "armature")).strip().lower()
    if pack_mode not in ("armature", "lbs", "both"):
        print(f"[export-dog][warn] unknown pack_mode={pack_mode!r}, fallback to armature")
        return "armature"
    return pack_mode


def export_one_dog_arp(
    dog_id: str,
    cfg: dict[str, Any],
    source_actions: list[dict[str, Any]],
    output_root: Path,
    *,
    skip_bvh_export: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    blender: str = "blender",
    blender_threads: int | None = None,
    arp_addon_modules: list[str] | None = None,
) -> dict[str, Any]:
    """ARP-retarget all source actions onto one dog; write the dog-blend manifest."""
    targets_cfg = cfg.get("targets", {}) or {}
    assets_cfg = dict(cfg.get("assets", {}) or {})
    arp_cfg = dict(cfg.get("arp", {}) or {})
    char_root = Path(
        targets_cfg.get("char_root", "./datasets/shepherd/batch2_dogs/batch2_dogs_char")
    )
    if not char_root.is_absolute():
        char_root = (REPO_ROOT / char_root).resolve()

    tgt_bvh = find_tgt_bvh(dog_id, char_root)
    if tgt_bvh is None:
        raise FileNotFoundError(f"target BVH not found for dog_id={dog_id} under {char_root}")
    tgt_fbx = resolve_fbx_from_bvh(
        tgt_bvh,
        recursive_search=bool(assets_cfg.get("recursive_fbx_search", False)),
    )
    shape_root = Path(
        targets_cfg.get("shape_root", "./datasets/shepherd/batch2_dogs/batch2_dogs_shape")
    )
    if not shape_root.is_absolute():
        shape_root = (REPO_ROOT / shape_root).resolve()
    tgt_shape_path = find_tgt_shape(dog_id, shape_root)

    pack_mode = resolve_pack_mode(cfg)
    want_lbs = pack_mode in ("lbs", "both")
    want_bake = pack_mode in ("armature", "both")
    skip_model_bvh = bool(skip_bvh_export)

    print(
        f"[export-dog] dog={dog_id} retarget_mode=arp "
        f"actions={len(source_actions)} shard={shard_index}/{num_shards} "
        f"pack_mode={pack_mode} want_lbs={want_lbs} want_bake={want_bake}",
        flush=True,
    )

    dog_out = output_root / dog_id
    dog_out.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    failed: list[str] = []
    recursive_fbx = bool(assets_cfg.get("recursive_fbx_search", False))
    for action in source_actions:
        action_id = action["action_id"]
        inp_bvh = Path(action["inp_bvh_path"])
        if not inp_bvh.exists():
            msg = f"missing source BVH: {inp_bvh}"
            print(f"[export-dog][fail] {msg}")
            failed.append(msg)
            continue
        try:
            inp_fbx = resolve_fbx_from_bvh(inp_bvh, recursive_search=recursive_fbx)
        except FileNotFoundError as exc:
            if bool(arp_cfg.get("use_source_fbx", True)):
                msg = f"{action_id}: {exc}"
                print(f"[export-dog][fail] {msg}")
                failed.append(msg)
                continue
            inp_fbx = None
        action_out = dog_out / action_id
        action_out.mkdir(parents=True, exist_ok=True)
        mesh_raw = action_out / f"{action_id}_arp_mesh.npz"
        lbs_path = action_out / f"{action_id}_ours_lbs.npz"
        bake_path = action_out / f"{action_id}_ours_fbx_bake.npz"
        bvh_path = action_out / f"{action_id}_ours_native.bvh"
        case = {
            "case_id": action_id,
            "inp_bvh_path": str(inp_bvh),
            "tgt_bvh_path": str(tgt_bvh),
            "tgt_fbx_path": str(tgt_fbx),
            "mesh_path": str(mesh_raw),
            "bake_path": str(bake_path),
            "arp_bvh_path": str(bvh_path),
        }
        if inp_fbx is not None:
            case["inp_fbx_path"] = str(inp_fbx)
        cases.append(case)
        planned.append(
            {
                "action": action,
                "action_id": action_id,
                "action_out": action_out,
                "mesh_raw": mesh_raw,
                "lbs_path": lbs_path,
                "bake_path": bake_path,
                "bvh_path": bvh_path,
                "inp_fbx": str(inp_fbx) if inp_fbx is not None else None,
            }
        )

    if cases:
        runtime_arp = dict(arp_cfg)
        runtime_arp["export_mesh"] = bool(want_lbs)
        runtime_arp["export_bake"] = bool(want_bake)
        runtime_arp["export_bvh"] = not skip_model_bvh
        # Keep authored root travel for pack (jumps / strides). Sequence videos
        # often lock XY; dog-blend armature/LBS packing should not.
        runtime_arp.setdefault("mesh_export_space", "blender_world_z_up")
        runtime_arp["output_dir"] = str(dog_out / "_arp_bvh")
        runtime_arp["mesh_output_dir"] = str(dog_out / "_arp_mesh")
        runtime_arp["bake_output_dir"] = str(dog_out / "_arp_bake")
        runtime_cfg = {
            "arp": runtime_arp,
            "assets": assets_cfg,
            "cases": cases,
        }
        shard_tag = f".shard{shard_index}" if num_shards > 1 else ""
        runtime_cfg_path = dog_out / f"_arp_batch_config{shard_tag}.yaml"
        _write_yaml(runtime_cfg_path, runtime_cfg)
        _run_arp_blender_export(
            runtime_cfg_path=runtime_cfg_path,
            blender=blender,
            blender_threads=blender_threads,
            arp_addon_modules=list(arp_addon_modules or ["auto_rig_pro-master", "auto_rig_pro"]),
        )

    exported: list[dict[str, Any]] = []
    for item in planned:
        action = item["action"]
        action_id = item["action_id"]
        try:
            lbs_path = None
            num_frames = None
            num_vertices = None
            if want_lbs:
                raw = Path(item["mesh_raw"])
                if not raw.is_file():
                    raise FileNotFoundError(f"ARP mesh cache missing: {raw}")
                num_frames, num_vertices = _convert_arp_mesh_to_ours_lbs(
                    raw, Path(item["lbs_path"])
                )
                lbs_path = Path(item["lbs_path"])
            bake_out = None
            if want_bake:
                bake_out = Path(item["bake_path"])
                if not bake_out.is_file():
                    raise FileNotFoundError(f"ARP bake cache missing: {bake_out}")
                if num_frames is None:
                    bake = np.load(str(bake_out), allow_pickle=True)
                    num_frames = int(np.asarray(bake["ours_globals"]).shape[0])
            native_bvh = None
            if not skip_model_bvh:
                native_bvh = Path(item["bvh_path"])
                if not native_bvh.is_file():
                    native_bvh = None
            if num_frames is None:
                try:
                    num_frames = read_bvh_frame_count(Path(action["inp_bvh_path"]))
                except Exception:
                    num_frames = 0
            shape_path = action.get("inp_shape_path")
            exported.append(
                {
                    "action_id": action_id,
                    "action_stem": action["action_stem"],
                    "folder": action["folder"],
                    "inp_bvh_path": str(action["inp_bvh_path"]),
                    "inp_shape_path": str(shape_path) if shape_path else None,
                    "inp_fbx_path": item.get("inp_fbx"),
                    "ours_bvh_path": None,
                    "ours_native_bvh_path": str(native_bvh) if native_bvh else None,
                    "ours_fbx_bake_path": str(bake_out) if bake_out else None,
                    "ours_lbs_path": str(lbs_path) if lbs_path else None,
                    "num_frames": int(num_frames or 0),
                    "num_vertices": num_vertices,
                    "collection_name": action_id,
                    "retarget_mode": "arp",
                }
            )
        except Exception as exc:
            msg = f"{action_id}: {exc}"
            print(f"[export-dog][fail] {msg}")
            failed.append(msg)

    blend_cfg = cfg.get("blend", {}) or {}
    manifest = {
        "dog_id": dog_id,
        "target_fbx_path": str(tgt_fbx),
        "target_bvh_path": str(tgt_bvh),
        "tgt_shape_path": str(tgt_shape_path) if tgt_shape_path else None,
        "retarget_mode": "arp",
        "stage2_mode": "arp",
        "gate_scale": 0.0,
        "force_gate_ones": False,
        "weights_loaded": "arp",
        "pack_mode": pack_mode,
        "inference_tgt_axis_transform": None,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "blend": {
            "layout": blend_cfg.get("layout", "grid"),
            "spacing": float(blend_cfg.get("spacing", 2.5)),
            "grid_cols": int(blend_cfg.get("grid_cols", 10)),
            "hide_all_but_first": bool(blend_cfg.get("hide_all_but_first", True)),
            "rot_z_deg": float(blend_cfg.get("rot_z_deg", 0.0)),
            "fps": int(blend_cfg.get("fps", 24)),
            "compress": bool(blend_cfg.get("compress", True)),
            "pack_mode": pack_mode,
            "lbs_axis_transform": blend_cfg.get("lbs_axis_transform", "y_up_to_z_up"),
            "lbs_global_yaw_deg": float(blend_cfg.get("lbs_global_yaw_deg", 0.0)),
            "animation_mode": blend_cfg.get("animation_mode", "shapekeys"),
        },
        "actions": exported,
        "failed": failed,
    }
    if num_shards > 1:
        manifest_path = dog_out / f"dog_actions_manifest.shard{shard_index}.json"
    else:
        manifest_path = dog_out / "dog_actions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[export-dog][{dog_id}] done: ok={len(exported)} fail={len(failed)} "
        f"manifest={manifest_path}"
    )
    return manifest


def export_one_dog(
    dog_id: str,
    cfg: dict[str, Any],
    source_actions: list[dict[str, Any]],
    model,
    stats,
    device,
    output_root: Path,
    *,
    skip_bvh_export: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    retarget_mode: str = "r2et",
) -> dict[str, Any]:
    retarget_mode = resolve_retarget_mode({"retarget_mode": retarget_mode})
    targets_cfg = cfg.get("targets", {}) or {}
    char_root = Path(targets_cfg.get("char_root", "./datasets/shepherd/batch2_dogs/batch2_dogs_char"))
    shape_root = Path(targets_cfg.get("shape_root", "./datasets/shepherd/batch2_dogs/batch2_dogs_shape"))
    if not char_root.is_absolute():
        char_root = (REPO_ROOT / char_root).resolve()
    if not shape_root.is_absolute():
        shape_root = (REPO_ROOT / shape_root).resolve()

    tgt_bvh = find_tgt_bvh(dog_id, char_root)
    if tgt_bvh is None:
        raise FileNotFoundError(f"target BVH not found for dog_id={dog_id} under {char_root}")
    tgt_shape_path = find_tgt_shape(dog_id, shape_root)
    if tgt_shape_path is None:
        raise FileNotFoundError(f"target shape npz not found for dog_id={dog_id} under {shape_root}")
    tgt_fbx = resolve_fbx_from_bvh(tgt_bvh)

    motion_cfg = cfg.get("motion", {}) or {}
    tgt_motion = get_inp_from_bvh(str(tgt_bvh), **motion_parse_options(motion_cfg, "tgt"))
    if tgt_motion is None:
        raise RuntimeError(f"failed to parse target BVH: {tgt_bvh}")
    tgt_shape = None
    if retarget_mode == "r2et":
        tgt_shape = load_shape_vector(str(tgt_shape_path))
    tgt_mesh = load_mesh_from_npz(
        str(tgt_shape_path), **mesh_load_options(motion_cfg, "tgt")
    )
    tgt_rest_skel = lbs_rest_skel_from_mesh(
        tgt_mesh, fallback_skel=tgt_motion["skel"][0]
    )
    report_lbs_rest_skel_mismatch(
        tgt_rest_skel,
        tgt_motion["skel"][0],
        label=dog_id,
        mesh_canonicalized=tgt_mesh.get("bind_pose_canonicalized"),
    )

    if retarget_mode == "r2et":
        gate_scale, force_gate_ones, stage2_mode = resolve_stage2_inference_knobs(
            cfg.get("stage2", {})
        )
    else:
        gate_scale, force_gate_ones, stage2_mode = 0.0, False, "direct"

    blend_cfg = cfg.get("blend", {}) or {}
    pack_mode = str(blend_cfg.get("pack_mode", "armature")).strip().lower()
    if pack_mode not in ("armature", "lbs", "both"):
        print(f"[export-dog][warn] unknown pack_mode={pack_mode!r}, fallback to armature")
        pack_mode = "armature"
    want_lbs = pack_mode in ("lbs", "both")
    want_native_bvh = pack_mode in ("armature", "both")
    # Legacy flag: only skips model-space retarget_to_bvh, never skips native BVH.
    skip_model_bvh = bool(skip_bvh_export)

    tgt_axis = motion_cfg.get("tgt_axis_transform", "none")
    print(
        f"[export-dog] dog={dog_id} retarget_mode={retarget_mode} "
        f"stage2_mode={stage2_mode} gate_scale={gate_scale} "
        f"force_gate_ones={force_gate_ones} "
        f"actions={len(source_actions)} shard={shard_index}/{num_shards} "
        f"pack_mode={pack_mode} want_lbs={want_lbs} "
        f"want_native_bvh={want_native_bvh} want_fbx_bake={want_native_bvh}"
    )

    dog_out = output_root / dog_id
    dog_out.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    failed: list[str] = []
    for idx, action in enumerate(source_actions):
        action_id = action["action_id"]
        inp_bvh = Path(action["inp_bvh_path"])
        inp_shape_path = Path(action["inp_shape_path"])
        print(f"[export-dog][{dog_id}] ({idx + 1}/{len(source_actions)}) {action_id}")

        if not inp_bvh.exists():
            msg = f"missing source BVH: {inp_bvh}"
            print(f"[export-dog][fail] {msg}")
            failed.append(msg)
            continue
        if retarget_mode == "r2et" and not inp_shape_path.exists():
            msg = f"missing source shape: {inp_shape_path}"
            print(f"[export-dog][fail] {msg}")
            failed.append(msg)
            continue
        try:
            inp_motion = get_inp_from_bvh(
                str(inp_bvh), **motion_parse_options(motion_cfg, "inp")
            )
            if inp_motion is None:
                raise RuntimeError("get_inp_from_bvh returned None")
            num_frames = len(inp_motion["quat"])

            if retarget_mode == "direct":
                # CopyQuat: copy source local quats onto target mesh rest skel.
                # rest_skel_tgt MUST be lbs_rest_skel_from_mesh (not BVH alone)
                # to keep bind facing aligned and avoid "拉皮".
                copy_local, ours_global, ours_quat, _height_scale = compute_copyquat_outputs(
                    inp_motion,
                    tgt_motion,
                    device,
                    rest_skel_tgt=tgt_rest_skel,
                )
                # Bake / world_joints_from_motion expect stats-normalized local.
                ours_local = normalize_local_for_stats(copy_local, stats)
            else:
                inp_shape = load_shape_vector(str(inp_shape_path))
                ours_local, ours_global, ours_quat = run_retarget_inference(
                    model,
                    mode=stage2_mode,
                    inp_motion=inp_motion,
                    tgt_motion=tgt_motion,
                    inp_shape=inp_shape,
                    tgt_shape=tgt_shape,
                    stats=stats,
                    device=device,
                    gate_scale=gate_scale,
                    force_gate_ones=force_gate_ones,
                )
            ours_local = align_frames(ours_local.astype(np.float32), num_frames)
            ours_global = align_frames(ours_global.astype(np.float32), num_frames)
            ours_quat = align_frames(ours_quat.astype(np.float32), num_frames)

            action_out = dog_out / action_id
            action_out.mkdir(parents=True, exist_ok=True)

            lbs_path = None
            num_vertices = None
            if want_lbs:
                ours_verts = skin_mesh_sequence(
                    ours_quat,
                    tgt_rest_skel,
                    tgt_mesh,
                    device,
                )
                lbs_path = action_out / f"{action_id}_ours_lbs.npz"
                np.savez_compressed(
                    lbs_path,
                    ours_vertices=ours_verts.astype(np.float32),
                    target_faces=tgt_mesh["faces"].astype(np.int32),
                    num_frames=np.int32(num_frames),
                )
                num_vertices = int(ours_verts.shape[1])

            ours_bvh = None
            if not skip_model_bvh:
                _, _, ours_bvh = retarget_to_bvh(
                    inp_motion,
                    tgt_motion,
                    ours_local,
                    ours_global,
                    ours_quat,
                    stats,
                    action_out,
                    f"{action_id}_ours",
                    inp_bvh_path=str(inp_bvh),
                    tgt_bvh_path=str(tgt_bvh),
                )

            ours_native_bvh = None
            ours_fbx_bake = None
            if want_native_bvh:
                # Optional debug BVH (incomplete facing inverse — not used for pack bake).
                _, ours_native_bvh = retarget_to_native_bvh(
                    inp_motion,
                    tgt_motion,
                    ours_local,
                    ours_global,
                    ours_quat,
                    stats,
                    action_out,
                    f"{action_id}_ours",
                    tgt_bvh_path=str(tgt_bvh),
                    axis_transform=tgt_axis,
                )
                # Pack bakes from model-space FK (same space as LBS / inspect).
                ours_fbx_bake = write_fbx_bake_cache_from_model(
                    ours_quat=ours_quat,
                    ours_local=ours_local,
                    ours_global=ours_global,
                    stats=stats,
                    rest_skel=tgt_rest_skel,
                    parents=SMAL33_PARENTS,
                    save_path=action_out / f"{action_id}_ours_fbx_bake.npz",
                )

            exported.append(
                {
                    "action_id": action_id,
                    "action_stem": action["action_stem"],
                    "folder": action["folder"],
                    "inp_bvh_path": str(inp_bvh),
                    "inp_shape_path": str(inp_shape_path),
                    "ours_bvh_path": str(ours_bvh) if ours_bvh is not None else None,
                    "ours_native_bvh_path": (
                        str(ours_native_bvh) if ours_native_bvh is not None else None
                    ),
                    "ours_fbx_bake_path": (
                        str(ours_fbx_bake) if ours_fbx_bake is not None else None
                    ),
                    "ours_lbs_path": str(lbs_path) if lbs_path is not None else None,
                    "num_frames": int(num_frames),
                    "num_vertices": num_vertices,
                    "collection_name": action_id,
                    "retarget_mode": retarget_mode,
                }
            )
        except Exception as exc:
            msg = f"{action_id}: {exc}"
            print(f"[export-dog][fail] {msg}")
            failed.append(msg)

    manifest = {
        "dog_id": dog_id,
        "target_fbx_path": str(tgt_fbx),
        "target_bvh_path": str(tgt_bvh),
        "tgt_shape_path": str(tgt_shape_path),
        "retarget_mode": retarget_mode,
        "stage2_mode": stage2_mode,
        "gate_scale": float(gate_scale),
        "force_gate_ones": bool(force_gate_ones),
        "weights_loaded": (cfg.get("model", {}) or {}).get("_weights_loaded"),
        "pack_mode": pack_mode,
        "inference_tgt_axis_transform": tgt_axis,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "blend": {
            "layout": blend_cfg.get("layout", "grid"),
            "spacing": float(blend_cfg.get("spacing", 2.5)),
            "grid_cols": int(blend_cfg.get("grid_cols", 10)),
            "hide_all_but_first": bool(blend_cfg.get("hide_all_but_first", True)),
            "rot_z_deg": float(blend_cfg.get("rot_z_deg", 0.0)),
            "fps": int(blend_cfg.get("fps", 24)),
            "compress": bool(blend_cfg.get("compress", True)),
            "pack_mode": pack_mode,
            "lbs_axis_transform": blend_cfg.get("lbs_axis_transform", "y_up_to_z_up"),
            "lbs_global_yaw_deg": float(blend_cfg.get("lbs_global_yaw_deg", 0.0)),
            "animation_mode": blend_cfg.get("animation_mode", "shapekeys"),
        },
        "actions": exported,
        "failed": failed,
    }
    if num_shards > 1:
        manifest_path = dog_out / f"dog_actions_manifest.shard{shard_index}.json"
    else:
        manifest_path = dog_out / "dog_actions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[export-dog][{dog_id}] done: ok={len(exported)} fail={len(failed)} "
        f"manifest={manifest_path}"
    )
    return manifest


def merge_shard_manifests(dog_out: Path, dog_id: str, num_shards: int) -> dict[str, Any]:
    shards = []
    for i in range(num_shards):
        path = dog_out / f"dog_actions_manifest.shard{i}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing shard manifest: {path}")
        shards.append(json.loads(path.read_text(encoding="utf-8")))

    # Prefer a non-empty shard as the metadata base (paths / blend knobs).
    base = None
    for shard in shards:
        if shard.get("target_fbx_path") and (shard.get("actions") or shard.get("tgt_shape_path")):
            base = dict(shard)
            break
    if base is None:
        base = dict(shards[0])

    actions = []
    failed = []
    seen = set()
    for shard in shards:
        failed.extend(shard.get("failed") or [])
        for action in shard.get("actions") or []:
            aid = action.get("action_id")
            if aid in seen:
                continue
            seen.add(aid)
            actions.append(action)
    actions.sort(key=lambda a: (a.get("folder", ""), a.get("action_id", "")))
    base["actions"] = actions
    base["failed"] = failed
    base["shard_index"] = 0
    base["num_shards"] = 1
    base["dog_id"] = dog_id
    out_path = dog_out / "dog_actions_manifest.json"
    out_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export-dog][{dog_id}] merged {num_shards} shards -> {len(actions)} actions")
    return base


def main():
    args = parse_args()
    apply_cpu_thread_limit(args.cpu_threads)
    cfg = load_config(args.config.resolve())

    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root)
    if args.device is not None:
        cfg["device"] = int(args.device)
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
    if args.retarget_mode is not None:
        cfg["retarget_mode"] = args.retarget_mode

    retarget_mode = resolve_retarget_mode(cfg)

    source_cfg = cfg.get("source", {}) or {}
    train_char = Path(source_cfg.get("train_char", "./datasets/shepherd/smal@shepherd/train_char"))
    train_shape = Path(source_cfg.get("train_shape", "./datasets/shepherd/smal@shepherd/train_shape"))
    if not train_char.is_absolute():
        train_char = (REPO_ROOT / train_char).resolve()
    if not train_shape.is_absolute():
        train_shape = (REPO_ROOT / train_shape).resolve()

    if args.min_frames is not None:
        min_frames = int(args.min_frames)
    else:
        min_frames = int(source_cfg.get("min_frames", 0) or 0)

    dog_ids = args.dog_ids
    if not dog_ids:
        dog_ids = list((cfg.get("targets", {}) or {}).get("dog_ids") or [])
    if not dog_ids:
        raise SystemExit("No dog_ids specified in config or CLI.")

    if args.actions is not None:
        explicit_actions = list(args.actions)
    else:
        explicit_actions = normalize_action_specs(source_cfg.get("actions"))

    source_actions, source_mode = select_source_actions(
        train_char,
        train_shape,
        min_frames=min_frames,
        explicit_actions=explicit_actions,
        repo_root=REPO_ROOT,
        require_shape=(retarget_mode != "arp"),
    )
    if args.limit_actions is not None:
        source_actions = source_actions[: max(int(args.limit_actions), 0)]
    total_before_shard = len(source_actions)
    source_actions = shard_actions(source_actions, args.shard_index, args.num_shards)
    if total_before_shard == 0:
        if source_mode == "explicit":
            raise SystemExit(
                "No source actions resolved from --actions / source.actions"
            )
        raise SystemExit(f"No source actions discovered under {train_char}")
    if not source_actions:
        # Valid for uneven shard splits (e.g. 3 actions / 4 GPUs).
        print(
            f"[export-dog] shard={args.shard_index}/{args.num_shards} has 0 actions; "
            "writing empty shard manifest."
        )
        output_root = Path(cfg.get("output_root", "./visualization/blend/_work"))
        if not output_root.is_absolute():
            output_root = (REPO_ROOT / output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        for dog_id in dog_ids:
            dog_out = output_root / dog_id
            dog_out.mkdir(parents=True, exist_ok=True)
            empty = {
                "dog_id": dog_id,
                "actions": [],
                "failed": [],
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "retarget_mode": retarget_mode,
                "pack_mode": str((cfg.get("blend", {}) or {}).get("pack_mode", "armature")),
            }
            # Copy static fields from a minimal stub; merge will take non-empty shard as base.
            path = dog_out / f"dog_actions_manifest.shard{args.shard_index}.json"
            # Prefer writing a fuller stub if another shard already wrote one.
            existing = sorted(dog_out.glob("dog_actions_manifest.shard*.json"))
            if existing:
                base = json.loads(existing[0].read_text(encoding="utf-8"))
                base["actions"] = []
                base["failed"] = []
                base["shard_index"] = int(args.shard_index)
                base["num_shards"] = int(args.num_shards)
                path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                path.write_text(json.dumps(empty, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[export-dog] wrote empty shard manifest: {path}")
        return

    print(
        f"[export-dog] retarget_mode={retarget_mode} mode={source_mode} "
        f"shard={args.shard_index}/{args.num_shards} "
        f"actions={len(source_actions)}/{total_before_shard} under {train_char}"
    )

    output_root = Path(cfg.get("output_root", "./visualization/blend/_work"))
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    skip_bvh = bool(args.skip_bvh_export or (cfg.get("blend", {}) or {}).get("skip_bvh_export", False))

    if retarget_mode == "arp":
        manifests = []
        for dog_id in dog_ids:
            manifest = export_one_dog_arp(
                dog_id,
                cfg,
                source_actions,
                output_root,
                skip_bvh_export=skip_bvh,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                blender=str(args.blender),
                blender_threads=args.blender_threads,
                arp_addon_modules=list(args.arp_addon_modules),
            )
            manifests.append(manifest)
        if args.num_shards == 1:
            index_path = output_root / "dog_actions_manifest_index.json"
            index_path.write_text(
                json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[export-dog] wrote index: {index_path}")
        else:
            print(
                f"[export-dog] shard {args.shard_index}/{args.num_shards} finished; "
                "batch driver will merge shard manifests."
            )
        return

    device = setup_cuda_device(int(cfg.get("device", 0)))
    model_cfg = cfg.get("model", {}) or {}
    stats_path = model_cfg.get("stats_path")
    if not stats_path:
        raise SystemExit("Missing model.stats_path.")
    stats = load_stats(stats_path)

    model = None
    weights_used = None
    if retarget_mode == "r2et":
        _, _, stage2_mode = resolve_stage2_inference_knobs(cfg.get("stage2", {}))
        model, weights_used = load_inference_model(
            stage2_mode, model_cfg, cfg["ret_model_args"], device
        )
        cfg.setdefault("model", {})
        cfg["model"]["_weights_loaded"] = weights_used
    else:
        cfg.setdefault("model", {})
        cfg["model"]["_weights_loaded"] = "copyquat"

    manifests = []
    for dog_id in dog_ids:
        manifest = export_one_dog(
            dog_id,
            cfg,
            source_actions,
            model,
            stats,
            device,
            output_root,
            skip_bvh_export=skip_bvh,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            retarget_mode=retarget_mode,
        )
        manifests.append(manifest)

    if args.num_shards == 1:
        index_path = output_root / "dog_actions_manifest_index.json"
        index_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[export-dog] wrote index: {index_path}")
    else:
        print(
            f"[export-dog] shard {args.shard_index}/{args.num_shards} finished; "
            "batch driver will merge shard manifests."
        )


if __name__ == "__main__":
    main()
