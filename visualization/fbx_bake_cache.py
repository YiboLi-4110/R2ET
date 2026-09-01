#!/usr/bin/env python3
"""
Bake-cache I/O for FBX armature packing.

Export (conda): build cache from *model-space* R2ET outputs (same space as LBS /
inspect: Y-up, +Z forward after get_inp_from_bvh). Blender pack Kabsch-aligns
this rest skeleton onto FBX bone heads, so jump direction matches LBS.

Blender pack: only load_fbx_bake_cache() — pure NumPy, no Animation import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Joint names matching SMAL33 remap order (Root + JOINTS_LIST).
SMAL33_JOINT_NAMES = [
    "Root",
    "Spine1",
    "Spine2",
    "Spine3",
    "Spine4",
    "Spine5",
    "Spine6",
    "LeftScapula",
    "LeftUpperArm",
    "LeftForeLeg",
    "LeftFrontPaw",
    "RightScapula",
    "RightUpperArm",
    "RightForeLeg",
    "RightFrontPaw",
    "Neck",
    "Head",
    "Jaw",
    "LeftThigh",
    "LeftShin",
    "LeftHock",
    "LeftHindPaw",
    "RightThigh",
    "RightShin",
    "RightHock",
    "RightHindPaw",
    "Tail1",
    "Tail2",
    "Tail3",
    "Tail4",
    "Tail5",
    "Tail6",
    "Tail7",
]


def _quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """(…, 4) wxyz -> (…, 3, 3)."""
    q = np.asarray(quats, dtype=np.float64)
    # Normalize
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)
    row0 = np.stack([m00, m01, m02], axis=-1)
    row1 = np.stack([m10, m11, m12], axis=-1)
    row2 = np.stack([m20, m21, m22], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def fk_globals_from_quat_rest(
    quat: np.ndarray,
    rest_skel: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    LBS-compatible FK: local translation = rest_skel, local rotation = quat.

    quat: (T, J, 4) or (J, 4)
    rest_skel: (J, 3)
    returns: (T, J, 4, 4)
    """
    quat = np.asarray(quat, dtype=np.float64)
    rest_skel = np.asarray(rest_skel, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    if quat.ndim == 2:
        quat = quat[None, ...]
    num_frames, num_joints = quat.shape[:2]
    if rest_skel.shape != (num_joints, 3):
        raise ValueError(
            f"rest_skel shape {rest_skel.shape} != ({num_joints}, 3)"
        )

    rot = _quat_to_rotmat(quat)
    local = np.zeros((num_frames, num_joints, 4, 4), dtype=np.float64)
    local[:, :, :3, :3] = rot
    local[:, :, :3, 3] = rest_skel[None, :, :]
    local[:, :, 3, 3] = 1.0

    global_tf = np.zeros_like(local)
    global_tf[:, 0] = local[:, 0]
    for j in range(1, num_joints):
        p = int(parents[j])
        if p < 0:
            global_tf[:, j] = local[:, j]
        else:
            global_tf[:, j] = global_tf[:, p] @ local[:, j]
    return global_tf


def build_fbx_bake_cache_from_model(
    *,
    ours_quat: np.ndarray,
    ours_local: np.ndarray,
    ours_global: np.ndarray,
    stats: dict,
    rest_skel: np.ndarray,
    parents: np.ndarray,
) -> dict[str, Any]:
    """
    Build bake cache in R2ET model space (same as LBS).

    rest_globals: (J, 4, 4) identity-quat FK at rest_skel
    ours_globals: (T, J, 4, 4) animated FK; root translation from put_in_world_bvh
    """
    from datasets.smal33_motion_io import NUM_JOINTS, world_joints_from_motion

    quat = np.asarray(ours_quat, dtype=np.float64)[:, :NUM_JOINTS]
    rest = np.asarray(rest_skel, dtype=np.float64).reshape(NUM_JOINTS, 3)
    parents = np.asarray(parents, dtype=np.int64)

    identity = np.zeros((NUM_JOINTS, 4), dtype=np.float64)
    identity[:, 0] = 1.0
    rest_globals = fk_globals_from_quat_rest(identity, rest, parents)[0]

    ours_globals = fk_globals_from_quat_rest(quat, rest, parents)
    # Root travel in model space (LBS keeps rest root; bake needs the jump path).
    world = world_joints_from_motion(
        np.asarray(ours_local, dtype=np.float32),
        np.asarray(ours_global, dtype=np.float32),
        stats,
    )
    world = np.asarray(world, dtype=np.float64)
    if world.shape[0] != ours_globals.shape[0]:
        n = min(world.shape[0], ours_globals.shape[0])
        world = world[:n]
        ours_globals = ours_globals[:n]
        quat = quat[:n]
    ours_globals = ours_globals.copy()
    ours_globals[:, 0, :3, 3] = world[:, 0, :]

    return {
        "rest_globals": rest_globals.astype(np.float64),
        "ours_globals": ours_globals.astype(np.float64),
        "joint_names": np.asarray(SMAL33_JOINT_NAMES[:NUM_JOINTS], dtype=object),
        "frametime": np.float64(1.0 / 24.0),
        "space": np.asarray("model"),
    }


def save_fbx_bake_cache(path: Path | str, cache: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        rest_globals=cache["rest_globals"],
        ours_globals=cache["ours_globals"],
        joint_names=cache["joint_names"],
        frametime=cache["frametime"],
        space=np.asarray(cache.get("space", "model")),
    )
    return path


def load_fbx_bake_cache(path: Path | str) -> dict[str, Any]:
    """Blender-safe: NumPy only."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)
    names = data["joint_names"]
    if getattr(names, "dtype", None) == object:
        joint_names = [str(x) for x in names.tolist()]
    else:
        joint_names = [str(x) for x in names]
    space = "model"
    if "space" in data.files:
        space = str(np.asarray(data["space"]).reshape(-1)[0])
    return {
        "rest_globals": np.asarray(data["rest_globals"], dtype=np.float64),
        "ours_globals": np.asarray(data["ours_globals"], dtype=np.float64),
        "joint_names": joint_names,
        "frametime": float(np.asarray(data["frametime"]).reshape(-1)[0]),
        "space": space,
    }


def write_fbx_bake_cache_from_model(
    *,
    ours_quat: np.ndarray,
    ours_local: np.ndarray,
    ours_global: np.ndarray,
    stats: dict,
    rest_skel: np.ndarray,
    parents: np.ndarray,
    save_path: Path | str,
) -> Path:
    cache = build_fbx_bake_cache_from_model(
        ours_quat=ours_quat,
        ours_local=ours_local,
        ours_global=ours_global,
        stats=stats,
        rest_skel=rest_skel,
        parents=parents,
    )
    return save_fbx_bake_cache(save_path, cache)


# --- Legacy BVH-based API (kept for optional debug; prefer model-space) ---


def write_fbx_bake_cache_for_pair(
    target_bvh_path: Path | str,
    native_bvh_path: Path | str,
    save_path: Path | str,
) -> Path:
    """Deprecated: native BVH lacks facing inverse; use write_fbx_bake_cache_from_model."""
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    outside = repo_root / "outside-code"
    if str(outside) not in sys.path:
        sys.path.insert(0, str(outside))
    import Animation  # noqa: WPS433
    import BVH  # noqa: WPS433

    def _globals(path: Path):
        anim, names, frametime = BVH.load(str(path))
        g = Animation.transforms_global(anim).astype(np.float64)
        return g, list(names), float(frametime)

    rest_all, rest_names, _ = _globals(Path(target_bvh_path))
    ours_g, ours_names, frametime = _globals(Path(native_bvh_path))
    if len(rest_names) != len(ours_names):
        raise RuntimeError("BVH joint count mismatch in legacy bake cache")
    cache = {
        "rest_globals": rest_all[0].astype(np.float64),
        "ours_globals": ours_g.astype(np.float64),
        "joint_names": np.asarray(ours_names, dtype=object),
        "frametime": np.float64(frametime),
        "space": np.asarray("native_bvh_legacy"),
    }
    return save_fbx_bake_cache(save_path, cache)
