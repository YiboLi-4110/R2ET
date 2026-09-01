#!/usr/bin/env python3
"""
R2ET retarget mesh cache export for ARP-sequence pipeline.

Mirrors ``direct_copy_mesh.py`` output format (space=lbs_y_up) but replaces
CopyQuat with R2ET stage1 / stage2 / blend inference from
``export_r2et_dog_actions_smal33.py``.

Run (usually via batch_arp_sequence_smal33.py with sequence.retarget_mode=r2et):
  python visualization/r2et_mesh.py --config /path/to/_r2et_batch_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_assets import enrich_case_assets  # noqa: E402
from datasets.smal33_motion_io import (  # noqa: E402
    get_inp_from_bvh,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
    load_shape_vector,
    load_stats,
    report_lbs_rest_skel_mismatch,
    setup_cuda_device,
)
from export_r2et_dog_actions_smal33 import (  # noqa: E402
    align_frames,
    load_inference_model,
    mesh_load_options,
    motion_parse_options,
    resolve_inp_shape_path,
    resolve_stage2_inference_knobs,
    run_retarget_inference,
    skin_mesh_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="R2ET mesh cache export for sequence.retarget_mode=r2et."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--stage2_mode",
        type=str,
        default=None,
        choices=["stage1", "stage2", "blend"],
        help="Override stage2.mode (stage1 | stage2 | blend).",
    )
    parser.add_argument("--gate_scale", type=float, default=None)
    parser.add_argument(
        "--stage1_weights",
        type=Path,
        default=None,
        help="Override model.stage1_weights.",
    )
    parser.add_argument(
        "--stage2_weights",
        type=Path,
        default=None,
        help="Override model.stage2_weights.",
    )
    return parser.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abs_path(path_like: str | Path, *, base: Path = _PROJECT_ROOT) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_source_shape_root(cfg: dict[str, Any]) -> Path:
    """Resolve the SOURCE action shape directory (not the target dog shape root).

    ``r2et.shape_root`` / ``direct.shape_root`` are target-side and must not be
    used here; otherwise batch runtime configs that copy direct.shape_root into
    r2et.shape_root would steal the source lookup path.
    """
    r2et_cfg = cfg.get("r2et", {}) or {}
    source_cfg = cfg.get("source", {}) or {}
    raw = (
        r2et_cfg.get("train_shape")
        or source_cfg.get("train_shape")
        or source_cfg.get("shape_root")
    )
    if not raw:
        raise SystemExit(
            "R2ET mode requires source.train_shape (or r2et.train_shape) "
            "to resolve per-clip source shape npz files."
        )
    path = _abs_path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"source shape root not found: {path}")
    return path


def resolve_target_shape_path(case_cfg: dict[str, Any], cfg: dict[str, Any]) -> Path:
    explicit = case_cfg.get("tgt_shape_path") or (cfg.get("target", {}) or {}).get(
        "tgt_shape_path"
    )
    if explicit:
        path = _abs_path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"tgt_shape_path not found: {path}")

    shape_root = (
        (cfg.get("r2et", {}) or {}).get("shape_root")
        or (cfg.get("direct", {}) or {}).get("shape_root")
        or (cfg.get("target", {}) or {}).get("shape_root")
        or "./datasets/shepherd/batch2_dogs/batch2_dogs_shape"
    )
    shape_root = _abs_path(shape_root)
    stem = Path(case_cfg["tgt_bvh_path"]).stem
    for candidate in (f"{stem}.npz", f"{stem}-0.npz"):
        path = shape_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Target shape npz not found under {shape_root} for stem={stem!r}. "
        "Set target.tgt_shape_path or r2et/direct.shape_root."
    )


def resolve_inp_shape_for_case(
    case_cfg: dict[str, Any],
    cfg: dict[str, Any],
    train_shape: Path,
) -> Path:
    explicit = case_cfg.get("inp_shape_path")
    if explicit:
        path = _abs_path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"inp_shape_path not found: {path}")

    bvh_path = Path(case_cfg["inp_bvh_path"])
    folder_name = bvh_path.parent.name
    shape_path = resolve_inp_shape_path(bvh_path.stem, folder_name, train_shape)
    if shape_path is None:
        raise FileNotFoundError(
            f"cannot resolve source shape for {bvh_path.stem} "
            f"(folder={folder_name}, train_shape={train_shape}). "
            "Set sources[].inp_shape_path or source.train_shape."
        )
    return shape_path


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    centers = (verts.min(axis=1) + verts.max(axis=1)) * 0.5
    if anchor == "median":
        ref = np.median(centers, axis=0)
    else:
        ref = centers[0]
    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    return verts - drift[:, None, :]


def apply_cli_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    cfg = dict(cfg)
    if args.stage2_mode is not None:
        cfg.setdefault("stage2", {})
        cfg["stage2"] = dict(cfg.get("stage2", {}) or {})
        cfg["stage2"]["mode"] = args.stage2_mode
    if args.gate_scale is not None:
        cfg.setdefault("stage2", {})
        cfg["stage2"] = dict(cfg.get("stage2", {}) or {})
        cfg["stage2"]["gate_scale"] = float(args.gate_scale)
    if args.stage1_weights is not None:
        cfg.setdefault("model", {})
        cfg["model"] = dict(cfg.get("model", {}) or {})
        cfg["model"]["stage1_weights"] = str(args.stage1_weights)
    if args.stage2_weights is not None:
        cfg.setdefault("model", {})
        cfg["model"] = dict(cfg.get("model", {}) or {})
        cfg["model"]["stage2_weights"] = str(args.stage2_weights)
    return cfg


@torch.no_grad()
def run_case(
    case_cfg: dict[str, Any],
    cfg: dict[str, Any],
    *,
    model,
    stats,
    device,
    stage2_mode: str,
    gate_scale: float,
    force_gate_ones: bool,
    train_shape: Path,
) -> Path:
    case_id = case_cfg["case_id"]
    r2et_cfg = cfg.get("r2et", {}) or {}
    motion_cfg = cfg.get("motion", {}) or {}

    inp_bvh = Path(case_cfg["inp_bvh_path"])
    tgt_bvh = Path(case_cfg["tgt_bvh_path"])
    inp_shape_path = resolve_inp_shape_for_case(case_cfg, cfg, train_shape)
    tgt_shape_path = resolve_target_shape_path(case_cfg, cfg)

    print(f"[r2et-mesh][{case_id}] mode={stage2_mode} gate_scale={gate_scale}")
    print(f"[r2et-mesh][{case_id}] source BVH: {inp_bvh}")
    print(f"[r2et-mesh][{case_id}] source shape: {inp_shape_path}")
    print(f"[r2et-mesh][{case_id}] target BVH: {tgt_bvh}")
    print(f"[r2et-mesh][{case_id}] target shape: {tgt_shape_path}")

    inp_motion = get_inp_from_bvh(
        str(inp_bvh), **motion_parse_options(motion_cfg, "inp")
    )
    tgt_motion = get_inp_from_bvh(
        str(tgt_bvh), **motion_parse_options(motion_cfg, "tgt")
    )
    if inp_motion is None or tgt_motion is None:
        raise RuntimeError(f"[{case_id}] failed to parse source/target BVH.")

    inp_shape = load_shape_vector(str(inp_shape_path))
    tgt_shape = load_shape_vector(str(tgt_shape_path))
    tgt_mesh = load_mesh_from_npz(
        str(tgt_shape_path), **mesh_load_options(motion_cfg, "tgt")
    )
    rest_skel_lbs = lbs_rest_skel_from_mesh(
        tgt_mesh, fallback_skel=tgt_motion["skel"][0]
    )
    report_lbs_rest_skel_mismatch(
        rest_skel_lbs,
        tgt_motion["skel"][0],
        label=case_id,
        mesh_canonicalized=tgt_mesh.get("bind_pose_canonicalized"),
    )

    num_frames = len(inp_motion["quat"])
    _local, _global, ours_quat = run_retarget_inference(
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
    ours_quat = align_frames(ours_quat.astype(np.float32), num_frames)
    verts = skin_mesh_sequence(
        ours_quat,
        rest_skel_lbs,
        tgt_mesh,
        device,
    ).astype(np.float32)

    if r2et_cfg.get("mesh_lock_horizontal_translation", False):
        anchor = str(r2et_cfg.get("mesh_lock_reference", "first")).lower()
        if anchor not in ("first", "median"):
            anchor = "first"
        verts = remove_horizontal_drift(verts, vertical_axis=1, anchor=anchor)

    mesh_dir = Path(
        r2et_cfg.get(
            "mesh_output_dir",
            "./visualization/videos/arp_sequence/r2et_mesh_outputs",
        )
    )
    if not mesh_dir.is_absolute():
        mesh_dir = (_PROJECT_ROOT / mesh_dir).resolve()
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / f"{case_id}_r2et_mesh.npz"

    np.savez_compressed(
        mesh_path,
        vertices=verts.astype(np.float32),
        faces=tgt_mesh["faces"].astype(np.int32),
        frame_start=np.int32(0),
        frame_end=np.int32(max(len(verts) - 1, 0)),
        space=np.asarray("lbs_y_up"),
        retarget_mode=np.asarray(f"r2et_{stage2_mode}"),
        stage2_mode=np.asarray(stage2_mode),
        gate_scale=np.float32(gate_scale),
        force_gate_ones=np.bool_(force_gate_ones),
        inp_shape_path=np.asarray(str(inp_shape_path)),
        tgt_shape_path=np.asarray(str(tgt_shape_path)),
    )
    print(
        f"[r2et-mesh][{case_id}] mesh cache: {mesh_path} "
        f"(T={verts.shape[0]} V={verts.shape[1]})"
    )
    return mesh_path


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_cfg(args.config.resolve()), args)
    assets_cfg = cfg.get("assets", {}) or {}
    cases = list(cfg.get("cases") or [])
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c.get("case_id") in wanted]
    if not cases:
        raise SystemExit("No cases to process.")

    model_cfg = cfg.get("model", {}) or {}
    stats_path = model_cfg.get("stats_path")
    if not stats_path:
        raise SystemExit("Missing model.stats_path.")
    ret_model_args = cfg.get("ret_model_args") or {}
    if not ret_model_args:
        raise SystemExit("Missing ret_model_args.")

    gate_scale, force_gate_ones, stage2_mode = resolve_stage2_inference_knobs(
        cfg.get("stage2", {})
    )
    train_shape = resolve_source_shape_root(cfg)

    device_id = (
        int(args.device) if args.device is not None else int(cfg.get("device", 0))
    )
    device = setup_cuda_device(device_id)
    stats = load_stats(stats_path)
    model, weights_used = load_inference_model(
        stage2_mode, model_cfg, ret_model_args, device
    )
    print(
        f"[r2et-mesh] loaded mode={stage2_mode} weights={weights_used} "
        f"gate_scale={gate_scale} force_gate_ones={force_gate_ones}"
    )

    produced = []
    for case in cases:
        resolved = enrich_case_assets(case, assets_cfg)
        for key in ("inp_shape_path", "tgt_shape_path"):
            if case.get(key):
                resolved[key] = case[key]
        path = run_case(
            resolved,
            cfg,
            model=model,
            stats=stats,
            device=device,
            stage2_mode=stage2_mode,
            gate_scale=gate_scale,
            force_gate_ones=force_gate_ones,
            train_shape=train_shape,
        )
        produced.append(str(path))

    summary = {
        "num_cases": len(produced),
        "mesh_paths": produced,
        "stage2_mode": stage2_mode,
        "gate_scale": gate_scale,
        "force_gate_ones": force_gate_ones,
        "weights": weights_used,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[r2et-mesh] done: {len(produced)} case(s)")


if __name__ == "__main__":
    main()
