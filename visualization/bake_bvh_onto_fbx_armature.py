#!/usr/bin/env python3
"""
Bake retarget motion onto an existing Blender FBX armature (no BVH re-import).

Uses a precomputed .npz cache built in *R2ET model space* (same as LBS/inspect:
Y-up, +Z forward). Kabsch-aligns model rest joints onto FBX bone heads, then
applies relative rotations + root translation only — so jump direction matches LBS.

Transfer rules (critical for bind-pose skinning):
  - Align model rest joint positions -> FBX bone heads (similarity).
  - Apply *relative* global rotations onto FBX rest bone orientations.
  - Root: location delta only.
  - Non-root: rotation only; location stays (0,0,0).
  - Rest smoke test must pass before animation keys are written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import bpy
import mathutils
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))

from fbx_bake_cache import load_fbx_bake_cache  # noqa: E402


def norm_bone_name(name: str) -> str:
    return str(name).split(":")[-1]


def kabsch_similarity(src: np.ndarray, dst: np.ndarray):
    """Similarity transform mapping src -> dst: x' = s R x + t (column vectors)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"Kabsch needs >=3 matched points, got {src.shape}")

    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_c = src - src_mu
    dst_c = dst - dst_mu
    cov = src_c.T @ dst_c
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    src_var = float(np.sum(src_c * src_c))
    scale = float(np.sum(s) / max(src_var, 1e-12))
    t = dst_mu - scale * (r @ src_mu)
    aligned = (scale * (src @ r.T)) + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return r.astype(np.float64), t.astype(np.float64), scale, rmse


def np_mat3_to_blender(mat3: np.ndarray) -> mathutils.Matrix:
    m = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(mat3[i, j])
    return m


def blender_mat3(mat: mathutils.Matrix) -> np.ndarray:
    return np.array(
        [
            [mat[0][0], mat[0][1], mat[0][2]],
            [mat[1][0], mat[1][1], mat[1][2]],
            [mat[2][0], mat[2][1], mat[2][2]],
        ],
        dtype=np.float64,
    )


def topological_pose_bones(armature) -> list:
    bones = list(armature.pose.bones)
    index = {b.name: i for i, b in enumerate(bones)}
    depth = {}

    def bone_depth(pb):
        name = pb.name
        if name in depth:
            return depth[name]
        if pb.parent is None:
            depth[name] = 0
        else:
            depth[name] = bone_depth(pb.parent) + 1
        return depth[name]

    return sorted(bones, key=lambda b: (bone_depth(b), index[b.name]))


def match_fbx_to_bvh_joints(armature, bvh_names: list[str]):
    fbx_by_norm = {}
    for pb in armature.pose.bones:
        key = norm_bone_name(pb.name)
        fbx_by_norm.setdefault(key, pb)

    pairs = []
    misses = []
    for j, name in enumerate(bvh_names):
        key = norm_bone_name(name)
        pb = fbx_by_norm.get(key)
        if pb is None:
            misses.append(name)
            continue
        pairs.append((pb, j))

    if not pairs:
        raise RuntimeError(
            "No FBX bone names matched BVH joints. "
            f"FBX sample={[b.name for b in list(armature.pose.bones)[:5]]} "
            f"BVH sample={bvh_names[:5]}"
        )

    fbx_heads = np.zeros((len(pairs), 3), dtype=np.float64)
    for i, (pb, _) in enumerate(pairs):
        head = pb.bone.head_local
        fbx_heads[i] = (float(head[0]), float(head[1]), float(head[2]))
    return pairs, fbx_heads, misses


def _homogenize_pos(mat4: np.ndarray) -> np.ndarray:
    w = float(mat4[3, 3]) if mat4.shape == (4, 4) else 1.0
    return mat4[:3, 3] / max(w, 1e-8)


def _reset_pose(armature):
    for pb in armature.pose.bones:
        pb.matrix_basis = mathutils.Matrix.Identity(4)
    bpy.context.view_layer.update()


def _mat3_to_quat(mat3: np.ndarray) -> mathutils.Quaternion:
    return np_mat3_to_blender(mat3).to_quaternion().normalized()


def _apply_rotation_pose(
    *,
    ordered,
    r_des_by_name: dict,
    root_delta: np.ndarray,
):
    """
    Root-to-leaf pose write:
      - non-root: rotation only via matrix_basis (location stays 0)
      - root: rotation + location delta in pose space
    """
    for pb in ordered:
        if pb.name not in r_des_by_name:
            continue
        r_des = r_des_by_name[pb.name]
        rest = pb.bone.matrix_local

        if pb.parent is None:
            # pose.matrix = rest @ basis  (no parent)
            # Want armature R = r_des, head = rest_head + delta
            head = np.array(rest.to_translation(), dtype=np.float64) + root_delta
            m_des = np_mat3_to_blender(r_des)
            m_des[0][3] = float(head[0])
            m_des[1][3] = float(head[1])
            m_des[2][3] = float(head[2])
            basis = rest.inverted() @ m_des
            pb.matrix_basis = basis
        else:
            # pose.matrix ~= parent.matrix @ inv(parent.rest) @ rest @ basis
            parent_pose = pb.parent.matrix
            parent_rest = pb.parent.bone.matrix_local
            # Local rest from parent: inv(parent_rest) @ rest
            local_rest = parent_rest.inverted() @ rest
            # With basis location=0: pose = parent_pose @ local_rest @ basis_rot
            # => R_pose = parent_R @ local_rest_R @ R_basis
            # => R_basis = inv(local_rest_R) @ inv(parent_R) @ r_des
            parent_r = blender_mat3(parent_pose)
            local_r = blender_mat3(local_rest)
            r_basis = np.linalg.inv(local_r) @ np.linalg.inv(parent_r) @ r_des
            pb.location = (0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
            pb.rotation_quaternion = _mat3_to_quat(r_basis)

        bpy.context.view_layer.update()


def _head_errors(pairs) -> tuple[float, float]:
    errs = []
    for pb, _ in pairs:
        posed = np.array(pb.matrix.to_translation(), dtype=np.float64)
        rest = np.array(pb.bone.head_local, dtype=np.float64)
        errs.append(float(np.linalg.norm(posed - rest)))
    return float(max(errs)), float(np.mean(errs))


def bake_cache_onto_fbx_armature(
    armature,
    *,
    bake_cache_path: Path,
    action_name: str,
    frame_start: int = 1,
    rest_smoke_frac: float = 1e-3,
    rest_smoke_abs: float = 1e-3,
) -> int:
    cache = load_fbx_bake_cache(bake_cache_path)
    rest_g = cache["rest_globals"]
    ours_g = cache["ours_globals"]
    joint_names = cache["joint_names"]
    space = cache.get("space", "unknown")
    print(f"[bake-fbx] cache space={space} path={bake_cache_path.name}")
    if space not in ("model", "unknown", "fbx_armature"):
        print(
            f"[bake-fbx][warn] cache space={space!r}; expected 'model' or 'fbx_armature' "
            "(re-export if jump direction looks wrong)"
        )

    if rest_g.ndim != 3 or rest_g.shape[-2:] != (4, 4):
        raise RuntimeError(f"Bad rest_globals shape: {rest_g.shape}")
    if ours_g.ndim != 4 or ours_g.shape[-2:] != (4, 4):
        raise RuntimeError(f"Bad ours_globals shape: {ours_g.shape}")
    if rest_g.shape[0] != ours_g.shape[1]:
        raise RuntimeError(
            f"Joint mismatch rest={rest_g.shape[0]} ours={ours_g.shape[1]}"
        )

    pairs, fbx_heads, misses = match_fbx_to_bvh_joints(armature, joint_names)
    if misses:
        print(
            f"[bake-fbx] unmatched BVH joints ({len(misses)}): "
            f"{misses[:8]}{'...' if len(misses) > 8 else ''}"
        )

    bvh_rest_pos = np.stack([_homogenize_pos(rest_g[j]) for _, j in pairs], axis=0)
    r_align, _t_align, scale, rmse = kabsch_similarity(bvh_rest_pos, fbx_heads)
    extent = float(np.linalg.norm(fbx_heads.max(0) - fbx_heads.min(0)))
    print(
        f"[bake-fbx] rest align: scale={scale:.6f} rmse={rmse:.6f} "
        f"matched={len(pairs)}/{len(joint_names)} extent={extent:.4f}"
    )
    if extent > 1e-6 and rmse > 0.05 * extent:
        raise RuntimeError(
            f"[bake-fbx] rest alignment RMSE too large ({rmse:.6f}); aborting"
        )

    r_bvh_rest = []
    r_fbx_rest = []
    root_pair_idx = None
    for i, (pb, j) in enumerate(pairs):
        r_bvh = rest_g[j, :3, :3].astype(np.float64)
        r_bvh_rest.append(r_align @ r_bvh)
        r_fbx_rest.append(blender_mat3(pb.bone.matrix_local))
        if pb.parent is None and root_pair_idx is None:
            root_pair_idx = i
    if root_pair_idx is None:
        root_pair_idx = 0
    r_bvh_rest = np.stack(r_bvh_rest, axis=0)
    r_fbx_rest = np.stack(r_fbx_rest, axis=0)
    r_bvh_rest_t = np.transpose(r_bvh_rest, (0, 2, 1))

    root_pb = pairs[root_pair_idx][0]
    root_bvh_j = pairs[root_pair_idx][1]
    p_bvh_root_rest = _homogenize_pos(rest_g[root_bvh_j])

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except RuntimeError:
        pass

    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"

    ordered = topological_pose_bones(armature)
    matched_names = {pb.name for pb, _ in pairs}

    # --- Rest smoke: identity relative motion must leave heads on rest ---
    _reset_pose(armature)
    r_des_rest = {pb.name: r_fbx_rest[i] for i, (pb, _) in enumerate(pairs)}
    _apply_rotation_pose(
        ordered=ordered,
        r_des_by_name=r_des_rest,
        root_delta=np.zeros(3, dtype=np.float64),
    )
    max_err, mean_err = _head_errors(pairs)
    smoke_tol = max(float(rest_smoke_abs), float(rest_smoke_frac) * max(extent, 1e-6))
    print(
        f"[bake-fbx] rest smoke: max_head_err={max_err:.6e} "
        f"mean_head_err={mean_err:.6e} tol={smoke_tol:.6e}"
    )
    if max_err > smoke_tol:
        raise RuntimeError(
            f"[bake-fbx] rest smoke FAILED (max_head_err={max_err:.6e} > {smoke_tol:.6e}). "
            "Refusing to write animation keys that would destroy bind-pose skinning."
        )
    _reset_pose(armature)

    if armature.animation_data is None:
        armature.animation_data_create()
    action = bpy.data.actions.new(name=action_name)
    armature.animation_data.action = action

    num_frames = int(ours_g.shape[0])
    for fi in range(num_frames):
        frame = int(frame_start + fi)
        r_des_by_name = {}
        for i, (pb, j) in enumerate(pairs):
            r_bvh_t = ours_g[fi, j, :3, :3].astype(np.float64)
            r_rel = (r_align @ r_bvh_t) @ r_bvh_rest_t[i]
            r_des_by_name[pb.name] = r_rel @ r_fbx_rest[i]

        p_bvh_root_t = _homogenize_pos(ours_g[fi, root_bvh_j])
        root_delta = scale * (r_align @ (p_bvh_root_t - p_bvh_root_rest))

        _apply_rotation_pose(
            ordered=ordered,
            r_des_by_name=r_des_by_name,
            root_delta=root_delta,
        )

        for pb in ordered:
            if pb.name not in matched_names:
                continue
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if pb.parent is None:
                pb.keyframe_insert(data_path="location", frame=frame)

    print(
        f"[bake-fbx] keyed action={action.name} frames={num_frames} "
        f"start={frame_start} cache={bake_cache_path.name} "
        f"root={root_pb.name} (location keys on root only)"
    )
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    return num_frames


def import_fbx_armature_and_meshes(fbx_path: Path):
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    arms = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not arms:
        raise RuntimeError(f"No armature in FBX: {fbx_path}")
    if not meshes:
        raise RuntimeError(f"No mesh in FBX: {fbx_path}")
    arm = arms[-1]
    if arm.animation_data is not None:
        arm.animation_data.action = None
    return arm, meshes, imported


def apply_root_object_transform(
    objects: Iterable,
    *,
    x_bias: float = 0.0,
    y_bias: float = 0.0,
    z_bias: float = 0.0,
    rot_z_deg: float = 0.0,
):
    rot_z_rad = math.radians(float(rot_z_deg))
    for obj in objects:
        if obj.parent is not None:
            continue
        obj.rotation_euler[2] += rot_z_rad
        obj.location[0] += float(x_bias)
        obj.location[1] += float(y_bias)
        obj.location[2] += float(z_bias)


def move_objects_to_collection(objects: Iterable, collection_name: str):
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)
    for obj in objects:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        if obj.name not in coll.objects:
            coll.objects.link(obj)
    return coll
