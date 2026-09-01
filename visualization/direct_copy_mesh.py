#!/usr/bin/env python3
"""
Same-skeleton direct retarget via CopyQuat (same logic as fourway compare).

Unlike the previous Blender local-pose copy, this path:
  1) parses source/target BVH into R2ET model space (Y-up, +Z forward)
  2) copies source local quaternions onto the target rest skeleton
  3) skins the target mesh with LBS using the shape.npz skeleton (same bind
     frame as rest_vertices; mesh bind yaw is canonicalized to match BVH)

This matches ``export_fourway_compare_smal33.compute_copyquat_outputs`` and avoids
Blender BVH/FBX bone-axis mismatches that swap limb directions.

Run (usually via batch_arp_sequence_smal33.py with sequence.retarget_mode=direct):
  python visualization/direct_copy_mesh.py --config /path/to/_direct_batch_config.yaml
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
_OUTSIDE = _PROJECT_ROOT / "outside-code"
for _p in (_PROJECT_ROOT, _SCRIPT_DIR, _OUTSIDE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Quaternions import Quaternions  # noqa: E402
from compare_assets import enrich_case_assets  # noqa: E402
from datasets.lbs_runtime import skin_mesh_sequence  # noqa: E402
from datasets.smal33_motion_io import (  # noqa: E402
    SMAL33_PARENTS,
    get_height_from_skel,
    get_inp_from_bvh,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
    report_lbs_rest_skel_mismatch,
    setup_cuda_device,
)
from src.forward_kinematics import FK  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="CopyQuat same-skeleton mesh cache export (fourway-compatible)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=int, default=None)
    return parser.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
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


def mesh_load_options(motion_cfg: dict[str, Any], prefix: str = "tgt"):
    """Keep mesh bind yaw aligned with the matching BVH parse options."""
    return {
        "canonicalize_bind_pose": bool(
            motion_cfg.get(f"{prefix}_canonicalize_bind_pose", True)
        ),
        "forward_mode": motion_cfg.get(f"{prefix}_forward_mode", "body"),
    }


def resolve_target_shape_path(case_cfg: dict[str, Any], cfg: dict[str, Any]) -> Path:
    explicit = case_cfg.get("tgt_shape_path") or (cfg.get("target", {}) or {}).get(
        "tgt_shape_path"
    )
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"tgt_shape_path not found: {path}")

    shape_root = (
        (cfg.get("direct", {}) or {}).get("shape_root")
        or (cfg.get("target", {}) or {}).get("shape_root")
        or "./datasets/shepherd/batch2_dogs/batch2_dogs_shape"
    )
    shape_root = Path(shape_root)
    if not shape_root.is_absolute():
        shape_root = (_PROJECT_ROOT / shape_root).resolve()

    stem = Path(case_cfg["tgt_bvh_path"]).stem  # e.g. 哈士奇_4-0
    for candidate in (f"{stem}.npz", f"{stem}-0.npz"):
        path = shape_root / candidate
        if path.exists():
            return path
    # Also try stem without trailing -0 duplication.
    raise FileNotFoundError(
        f"Target shape npz not found under {shape_root} for stem={stem!r}. "
        "Set target.tgt_shape_path or direct.shape_root."
    )


def apply_local_quat_basis_change(quat: np.ndarray, mode: str | None) -> np.ndarray:
    """
    Change of basis on local joint quaternions (conjugation), not a mesh yaw.

    cat_actions clip rest offsets often differ from batch3_dogs mesh bind by
    ~180° about Y (bone +Z vs -Z). Copying locals unchanged then inverts
    flexion/extension (divein up vs down) and swaps left/right. Conjugating
    by Ry180 maps X->-X, Z->-Z and keeps identity as rest.
    """
    mode = str(mode or "identity").strip().lower()
    quat = np.asarray(quat, dtype=np.float32)
    if mode in ("", "identity", "none"):
        return quat
    basis = {
        "yaw_180": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "rx_180": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "rz_180": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }.get(mode)
    if basis is None:
        raise ValueError(
            f"Unknown copyquat_quat_basis={mode!r}. "
            "Expected identity | yaw_180 | rx_180 | rz_180."
        )
    b = Quaternions(basis.reshape(1, 1, 4))
    return (-b * Quaternions(quat) * b).normalized().qs.astype(np.float32)


@torch.no_grad()
def compute_copyquat_outputs(inp_motion, tgt_motion, device, rest_skel_tgt=None):
    """Identical to export_fourway_compare_smal33.compute_copyquat_outputs."""
    source_quat = inp_motion["quat"].astype(np.float32)
    num_frames = len(source_quat)
    global_in = inp_motion["seq"][:, -8:-4].astype(np.float32)
    h_in = float(get_height_from_skel(inp_motion["skel"][0]))
    rest_skel_tgt = (
        np.asarray(rest_skel_tgt, dtype=np.float32)
        if rest_skel_tgt is not None
        else tgt_motion["skel"][0].astype(np.float32)
    )
    h_tgt = float(get_height_from_skel(rest_skel_tgt))
    scale = 1.0 if abs(h_in) < 1e-8 else (h_tgt / h_in)

    global_out = np.zeros_like(global_in, dtype=np.float32)
    global_out[:, :3] = global_in[:, :3] * scale
    global_out[:, 3] = global_in[:, 3]

    rest_repeat = np.repeat(rest_skel_tgt[None], num_frames, axis=0)
    parents = torch.as_tensor(SMAL33_PARENTS, dtype=torch.long, device=device)
    local_out = (
        FK.run(
            parents,
            torch.from_numpy(rest_repeat).float().to(device),
            torch.from_numpy(source_quat).float().to(device),
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return local_out, global_out.astype(np.float32), source_quat, float(scale)


def apply_root_motion_to_vertices(
    vertices: np.ndarray,
    global_vel: np.ndarray,
    *,
    axes: str | None = None,
    apply_yaw: bool = False,
) -> np.ndarray:
    """
    Bake processed root channels into LBS vertices (same integration as
    ``put_in_world_bvh``).

    ``global_vel`` is ``seq[:, -8:-4]``: (vx, vy, vz[, rvelocity]).
    Frame 0 stays at the posed origin; later frames accumulate translation
    (optionally heading-rotated) and/or yaw about +Y.

    Default is pose-only (axes=None, apply_yaw=False). Prefer axes='xz' so
    vertical root does not lift the mesh off the floor plane.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    g = np.asarray(global_vel, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != len(verts) or g.shape[1] < 3:
        raise ValueError(
            f"root channels shape {g.shape} incompatible with mesh frames {len(verts)}"
        )

    vel = np.zeros((len(verts), 3), dtype=np.float64)
    rvel = np.zeros(len(verts), dtype=np.float64)
    if axes is not None:
        vel[:, :3] = g[:, :3]
        mode = str(axes).strip().lower()
        if mode in ("xz", "horizontal"):
            vel[:, 1] = 0.0
        elif mode not in ("xyz", "true", "all"):
            raise ValueError(f"Unknown root-translation axes '{axes}'")
    if apply_yaw:
        if g.shape[1] < 4:
            raise ValueError("apply_root_yaw requires the rvelocity channel (4-vector).")
        rvel[:] = g[:, 3]

    if axes is None and not apply_yaw:
        return verts

    rotation = Quaternions.id(1)
    translation = np.zeros((1, 3), dtype=np.float64)
    out = np.empty_like(verts)
    for i in range(len(verts)):
        rotated = rotation * verts[i]
        out[i] = np.asarray(rotated, dtype=np.float32) + translation[0].astype(
            np.float32
        )
        if apply_yaw:
            rotation = (
                Quaternions.from_angle_axis(-rvel[i], np.array([0.0, 1.0, 0.0]))
                * rotation
            )
        translation = translation + np.asarray(
            rotation * vel[i], dtype=np.float64
        ).reshape(1, 3)
    return out


def apply_root_velocity_to_vertices(
    vertices: np.ndarray,
    global_vel: np.ndarray,
    *,
    axes: str = "xyz",
) -> np.ndarray:
    """Translation-only wrapper (no yaw). Kept for older callers."""
    return apply_root_motion_to_vertices(
        vertices, global_vel, axes=axes, apply_yaw=False
    )


def _parse_apply_root_translation(value) -> str | None:
    """Return axes mode, or None to skip. Default is off (pose-only LBS)."""
    if value is None or value is False:
        return None
    if value is True:
        return "xyz"
    mode = str(value).strip().lower()
    if mode in ("", "0", "false", "none", "off"):
        return None
    if mode in ("1", "true", "xyz", "all"):
        return "xyz"
    if mode in ("xz", "horizontal"):
        return "xz"
    raise ValueError(f"Unknown apply_root_translation={value!r}")


def _parse_apply_root_yaw(value) -> bool:
    """Default off. True bakes seq rvelocity as a +Y heading."""
    if value is None or value is False:
        return False
    if value is True:
        return True
    mode = str(value).strip().lower()
    if mode in ("", "0", "false", "none", "off"):
        return False
    if mode in ("1", "true", "on", "yes", "yaw"):
        return True
    raise ValueError(f"Unknown apply_root_yaw={value!r}")


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


def run_case(case_cfg: dict[str, Any], cfg: dict[str, Any], device) -> Path:
    case_id = case_cfg["case_id"]
    direct_cfg = cfg.get("direct", {}) or {}
    motion_cfg = cfg.get("motion", {}) or {}

    inp_bvh = Path(case_cfg["inp_bvh_path"])
    tgt_bvh = Path(case_cfg["tgt_bvh_path"])
    tgt_shape_path = resolve_target_shape_path(case_cfg, cfg)

    print(f"[direct-copy][{case_id}] source BVH: {inp_bvh}")
    print(f"[direct-copy][{case_id}] target BVH: {tgt_bvh}")
    print(f"[direct-copy][{case_id}] target shape: {tgt_shape_path}")

    inp_motion = get_inp_from_bvh(
        str(inp_bvh), **motion_parse_options(motion_cfg, "inp")
    )
    tgt_motion = get_inp_from_bvh(
        str(tgt_bvh), **motion_parse_options(motion_cfg, "tgt")
    )
    if inp_motion is None or tgt_motion is None:
        raise RuntimeError(f"[{case_id}] failed to parse source/target BVH.")

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
    print(
        f"[direct-copy][{case_id}] LBS rest skel=shape.npz "
        f"(canonicalized={bool(tgt_mesh.get('bind_pose_canonicalized'))})"
    )
    quat_basis = str(direct_cfg.get("copyquat_quat_basis", "identity") or "identity")
    if quat_basis not in ("identity", "none", ""):
        inp_motion = dict(inp_motion)
        inp_motion["quat"] = apply_local_quat_basis_change(
            inp_motion["quat"], quat_basis
        )
        print(f"[direct-copy][{case_id}] copyquat_quat_basis={quat_basis}")
    _local, copy_global, copy_quat, height_scale = compute_copyquat_outputs(
        inp_motion, tgt_motion, device, rest_skel_tgt=rest_skel_lbs
    )
    if not bool(direct_cfg.get("scale_root_translation", True)):
        copy_global = inp_motion["seq"][:, -8:-4].astype(np.float32)
        height_scale = 1.0

    verts = skin_mesh_sequence(
        copy_quat.astype(np.float32),
        rest_skel_lbs,
        tgt_mesh,
        device,
    ).astype(np.float32)

    # Default off (pose-only). Enable apply_root_translation=xz and/or
    # apply_root_yaw=true to bake locomotion / turning. Use xz (not xyz) so
    # floor alignment with the template dog is unchanged.
    root_axes = _parse_apply_root_translation(
        direct_cfg.get("apply_root_translation", False)
    )
    apply_yaw = _parse_apply_root_yaw(direct_cfg.get("apply_root_yaw", False))
    lock_cfg = direct_cfg.get("mesh_lock_horizontal_translation", False)
    if lock_cfg and (root_axes is not None or apply_yaw):
        print(
            f"[direct-copy][{case_id}][warn] mesh_lock_horizontal_translation "
            "will cancel baked XZ root motion; keep lock false to see travel."
        )
    if root_axes is not None or apply_yaw:
        before = verts[-1].mean(axis=0) - verts[0].mean(axis=0)
        verts = apply_root_motion_to_vertices(
            verts, copy_global, axes=root_axes, apply_yaw=apply_yaw
        )
        after = verts[-1].mean(axis=0) - verts[0].mean(axis=0)
        print(
            f"[direct-copy][{case_id}] applied root motion "
            f"axes={root_axes} yaw={apply_yaw} "
            f"(height_scale={height_scale:.4f}) "
            f"center_delta {before} -> {after}"
        )
    else:
        print(
            f"[direct-copy][{case_id}] pose-only LBS "
            f"(apply_root_translation=false apply_root_yaw=false "
            f"height_scale={height_scale:.4f})"
        )

    if lock_cfg:
        anchor = str(direct_cfg.get("mesh_lock_reference", "first")).lower()
        if anchor not in ("first", "median"):
            anchor = "first"
        verts = remove_horizontal_drift(verts, vertical_axis=1, anchor=anchor)

    mesh_dir = Path(
        direct_cfg.get(
            "mesh_output_dir",
            "./visualization/videos/arp_sequence/direct_mesh_outputs",
        )
    )
    if not mesh_dir.is_absolute():
        mesh_dir = (_PROJECT_ROOT / mesh_dir).resolve()
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / f"{case_id}_direct_mesh.npz"

    np.savez_compressed(
        mesh_path,
        vertices=verts.astype(np.float32),
        faces=tgt_mesh["faces"].astype(np.int32),
        frame_start=np.int32(0),
        frame_end=np.int32(max(len(verts) - 1, 0)),
        space=np.asarray("lbs_y_up"),
        retarget_mode=np.asarray("direct_copyquat"),
        height_scale=np.float32(height_scale),
        tgt_shape_path=np.asarray(str(tgt_shape_path)),
    )
    print(
        f"[direct-copy][{case_id}] mesh cache: {mesh_path} "
        f"(T={verts.shape[0]} V={verts.shape[1]} height_scale={height_scale:.4f})"
    )
    return mesh_path


def main():
    args = parse_args()
    cfg = load_cfg(args.config.resolve())
    assets_cfg = cfg.get("assets", {}) or {}
    cases = list(cfg.get("cases") or [])
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c.get("case_id") in wanted]
    if not cases:
        raise SystemExit("No cases to process.")

    device_id = (
        int(args.device)
        if args.device is not None
        else int(cfg.get("device", 0))
    )
    device = setup_cuda_device(device_id)

    produced = []
    for case in cases:
        resolved = enrich_case_assets(case, assets_cfg)
        # Preserve optional shape override from original case.
        if case.get("tgt_shape_path"):
            resolved["tgt_shape_path"] = case["tgt_shape_path"]
        path = run_case(resolved, cfg, device)
        produced.append(str(path))

    summary = {"num_cases": len(produced), "mesh_paths": produced}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[direct-copy] done: {len(produced)} case(s)")


if __name__ == "__main__":
    main()
