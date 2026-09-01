"""
Shared SMAL33 motion I/O utilities for eval / inference.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage.filters as filters
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTSIDE_CODE = REPO_ROOT / "outside-code"
if str(OUTSIDE_CODE) not in sys.path:
    sys.path.insert(0, str(OUTSIDE_CODE))

import Animation  # noqa: E402
import BVH  # noqa: E402
from Quaternions import Quaternions  # noqa: E402
from Pivots import Pivots  # noqa: E402

NUM_JOINTS = 33
GLOBAL_DIM = 4
SEQ_TAIL_DIM = 8

SMAL33_PARENTS = np.array(
    [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 16,
        0, 18, 19, 20, 0, 22, 23, 24, 0, 26, 27, 28, 29, 30, 31,
    ],
    dtype=np.int64,
)

JOINTS_LIST = [
    "Spine1", "Spine2", "Spine3", "Spine4", "Spine5", "Spine6",
    "LeftScapula", "LeftUpperArm", "LeftForeLeg", "LeftFrontPaw",
    "RightScapula", "RightUpperArm", "RightForeLeg", "RightFrontPaw",
    "Neck", "Head", "Jaw",
    "LeftThigh", "LeftShin", "LeftHock", "LeftHindPaw",
    "RightThigh", "RightShin", "RightHock", "RightHindPaw",
    "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6", "Tail7",
]

ATTENTION_JOINTS = [9, 10, 13, 14, 19, 20, 21, 23, 24, 25]
BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17]
LIMB_JOINTS = ATTENTION_JOINTS

STATS_FILES = {
    "local_mean": "smal33_local_motion_mean.npy",
    "local_std": "smal33_local_motion_std.npy",
    "global_mean": "smal33_global_motion_mean.npy",
    "global_std": "smal33_global_motion_std.npy",
    "quat_mean": "smal33_quat_mean.npy",
    "quat_std": "smal33_quat_std.npy",
}


AXIS_TRANSFORMS = {
    "none": np.eye(3, dtype=np.float64),
    # Shepherd bind-pose diagnostic result:
    #   new_x = old_y, new_y = -old_z, new_z = old_x
    "shepherd_y_negz_x": np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
    # A/B candidate when shepherd_y_negz_x appears vertically flipped:
    #   new_x = old_y, new_y = old_z, new_z = old_x
    "shepherd_y_z_x": np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
}


def axis_transform_matrix(name=None):
    key = "none" if name in (None, "", "none") else str(name)
    if key not in AXIS_TRANSFORMS:
        known = ", ".join(sorted(AXIS_TRANSFORMS))
        raise ValueError(f"Unknown axis transform '{name}'. Known: {known}")
    return AXIS_TRANSFORMS[key]


def apply_axis_transform_points(points, axis_transform=None):
    matrix = axis_transform_matrix(axis_transform)
    return np.einsum("ij,...j->...i", matrix, points)


def apply_axis_transform_quats(quats, axis_transform=None):
    matrix = axis_transform_matrix(axis_transform)
    if np.allclose(matrix, np.eye(3)):
        return quats.copy()
    rotations = Quaternions(quats).transforms()
    # Conjugate local rotations into the transformed coordinate basis.
    transformed = np.einsum("ij,...jk,lk->...il", matrix, rotations, matrix)
    return Quaternions.from_transforms(transformed).normalized().qs


def apply_axis_transform_anim(anim, axis_transform=None):
    matrix = axis_transform_matrix(axis_transform)
    if np.allclose(matrix, np.eye(3)):
        return anim
    anim.positions = apply_axis_transform_points(anim.positions, axis_transform)
    anim.offsets = apply_axis_transform_points(anim.offsets, axis_transform)
    anim.rotations.qs = apply_axis_transform_quats(anim.rotations.qs, axis_transform)
    anim.orients.qs = apply_axis_transform_quats(anim.orients.qs, axis_transform)
    return anim


def yaw_rotation_matrix_degrees(yaw_deg):
    """Right-handed rotation matrix about +Y (degrees)."""
    angle = np.deg2rad(float(yaw_deg))
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def yaw_quaternion_degrees(yaw_deg):
    """Unit quaternion for a right-handed yaw about +Y (degrees)."""
    angle = np.deg2rad(float(yaw_deg))
    qs = np.array(
        [np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0],
        dtype=np.float64,
    )
    return Quaternions(qs.reshape(1, 1, 4))


def apply_world_yaw_anim(anim, yaw_deg):
    """
    Apply a world yaw about +Y as a full basis change (same style as axis_transform).

    Rotates positions/offsets and conjugates local rotations/orients so that a
    clip which faces -X after ``shepherd_y_z_x`` can be aligned to +Z while
    keeping root local quaternions near identity (dog_actions convention).

    Default yaw_deg=0 is a no-op (preserves existing pipelines).
    """
    if yaw_deg is None or abs(float(yaw_deg)) < 1e-8:
        return anim
    matrix = yaw_rotation_matrix_degrees(yaw_deg)
    anim.positions = np.einsum("ij,...j->...i", matrix, np.asarray(anim.positions))
    anim.offsets = np.einsum("ij,...j->...i", matrix, np.asarray(anim.offsets))
    rotations = Quaternions(anim.rotations.qs).transforms()
    transformed = np.einsum("ij,...jk,lk->...il", matrix, rotations, matrix)
    anim.rotations.qs = Quaternions.from_transforms(transformed).normalized().qs
    orients = Quaternions(anim.orients.qs).transforms()
    transformed_o = np.einsum("ij,...jk,lk->...il", matrix, orients, matrix)
    anim.orients.qs = Quaternions.from_transforms(transformed_o).normalized().qs
    return anim


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def get_skel(joints, parents):
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else:
            c_offsets.append(joints[j, :])
    return np.stack(c_offsets, axis=0)


def estimate_forward(positions, mode="across", landmarks=None):
    if landmarks is None:
        sdr_l, sdr_r, hip_l, hip_r = 9, 13, 19, 23
    else:
        sdr_l = int(landmarks["sdr_l"])
        sdr_r = int(landmarks["sdr_r"])
        hip_l = int(landmarks["hip_l"])
        hip_r = int(landmarks["hip_r"])
    if mode == "across":
        across1 = positions[:, hip_l] - positions[:, hip_r]
        across0 = positions[:, sdr_l] - positions[:, sdr_r]
        across = across0 + across1
        across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]
        return np.cross(across, np.array([[0, 1, 0]]))
    if mode == "body":
        shoulder = 0.5 * (positions[:, sdr_l] + positions[:, sdr_r])
        hip = 0.5 * (positions[:, hip_l] + positions[:, hip_r])
        forward = shoulder - hip
        forward[:, 1] = 0.0
        return forward
    raise ValueError(f"Unknown forward mode '{mode}'. Expected 'across' or 'body'.")


def process_positions(positions, forward_mode="across", landmarks=None):
    if landmarks is None:
        fid_l, fid_r = np.array([10, 21]), np.array([14, 25])
        forward_landmarks = None
        contact_fid_l, contact_fid_r = fid_l, fid_r
    else:
        fid_l = np.asarray(landmarks["foot_l"], dtype=np.int64)
        fid_r = np.asarray(landmarks["foot_r"], dtype=np.int64)
        # Dummy root is inserted at index 0 below; shift facing landmarks.
        forward_landmarks = {
            "sdr_l": int(landmarks["sdr_l"]) + 1,
            "sdr_r": int(landmarks["sdr_r"]) + 1,
            "hip_l": int(landmarks["hip_l"]) + 1,
            "hip_r": int(landmarks["hip_r"]) + 1,
        }
        contact_fid_l = fid_l + 1
        contact_fid_r = fid_r + 1
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    positions = positions.copy()
    positions[:, :, 1] -= floor_height

    reference = positions[:, 0]
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])
    feet_l_x = (positions[1:, contact_fid_l, 0] - positions[:-1, contact_fid_l, 0]) ** 2
    feet_l_y = (positions[1:, contact_fid_l, 1] - positions[:-1, contact_fid_l, 1]) ** 2
    feet_l_z = (positions[1:, contact_fid_l, 2] - positions[:-1, contact_fid_l, 2]) ** 2
    feet_l_h = positions[:-1, contact_fid_l, 1]
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(np.float32)

    feet_r_x = (positions[1:, contact_fid_r, 0] - positions[:-1, contact_fid_r, 0]) ** 2
    feet_r_y = (positions[1:, contact_fid_r, 1] - positions[:-1, contact_fid_r, 1]) ** 2
    feet_r_z = (positions[1:, contact_fid_r, 2] - positions[:-1, contact_fid_r, 2]) ** 2
    feet_r_h = positions[:-1, contact_fid_r, 1]
    feet_r = (
        ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    ).astype(np.float32)

    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()
    positions[:, :, 0] = positions[:, :, 0] - positions[:, :1, 0]
    positions[1:, 1:, 1] = positions[1:, 1:, 1] - (
        positions[1:, :1, 1] - positions[:1, :1, 1]
    )
    positions[:, :, 2] = positions[:, :, 2] - positions[:, :1, 2]

    forward = estimate_forward(
        positions, mode=forward_mode, landmarks=forward_landmarks
    )
    forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 1]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    return positions, rotation


def remap_bvh_anim(anim, joint_names_in_file):
    to_keep = [0]
    for jname in JOINTS_LIST:
        for k, name in enumerate(joint_names_in_file):
            if jname == name[-len(jname) :]:
                to_keep.append(k + 1)
                break

    anim.parents = anim.parents[to_keep]
    for i in range(1, len(anim.parents)):
        if anim.parents[i] not in to_keep:
            anim.parents[i] = anim.parents[i] - 1
        anim.parents[i] = to_keep.index(anim.parents[i])

    anim.positions = anim.positions[:, to_keep, :]
    anim.rotations.qs = anim.rotations.qs[:, to_keep, :]
    anim.orients.qs = anim.orients.qs[to_keep, :]
    return anim, to_keep


def parse_bvh_joint_names(bvh_path):
    bvh_file = Path(bvh_path).read_text().split("JOINT")
    return [f.split("\n")[0].strip() for f in bvh_file[1:]]


def get_inp_from_bvh(
    bvh_path,
    axis_transform=None,
    forward_mode="across",
    canonicalize_bind_pose=True,
    bind_pose_dot_threshold=0.9,
    post_axis_yaw_deg=0.0,
    keep_joint_names=None,
    skeleton_mode="auto",
):
    """
    Load a BVH into the training/inference motion dict.

    Default ``skeleton_mode='auto'`` keeps the historical SMAL33 remap when the
    file (or ``keep_joint_names``) looks like SMAL33. Otherwise all joints are
    kept, or subset/reordered to ``keep_joint_names`` (shape.npz order).

    ``post_axis_yaw_deg`` (default 0) applies an optional world yaw about +Y
    *after* ``axis_transform`` and *before* facing canonicalization, as a full
    basis change (positions/offsets/quats). Use +90 for ARP-exported
    cat_actions / batch2_dogs so they face +Z with near-identity root quats
    like dog_actions. Leave 0 for dog_actions / smal@shepherd.
    """
    try:
        from .skeleton_io import (
            is_smal33_names,
            parse_bvh_hierarchy_names,
            remap_anim_to_names,
            resolve_landmark_indices,
        )
    except ImportError:
        from skeleton_io import (
            is_smal33_names,
            parse_bvh_hierarchy_names,
            remap_anim_to_names,
            resolve_landmark_indices,
        )

    anim, names, ftime = BVH.load(str(bvh_path))
    joint_names_no_root = parse_bvh_joint_names(bvh_path)
    hierarchy_names = parse_bvh_hierarchy_names(bvh_path)
    if len(hierarchy_names) == len(anim.parents):
        file_names = hierarchy_names
    else:
        file_names = [str(n) for n in names]
        if len(file_names) != len(anim.parents):
            raise RuntimeError(
                f"BVH joint count mismatch in {bvh_path}: "
                f"anim={len(anim.parents)} load_names={len(file_names)} "
                f"hierarchy={len(hierarchy_names)}"
            )

    keep_names = None
    if keep_joint_names is not None:
        keep_names = [str(n) for n in keep_joint_names]

    mode = str(skeleton_mode or "auto").strip().lower()
    if mode == "auto":
        probe = keep_names if keep_names else file_names
        mode = "smal33" if is_smal33_names(probe) else "generic"
    if mode not in ("smal33", "generic"):
        raise ValueError(f"Unknown skeleton_mode {skeleton_mode!r}")

    process_landmarks = None
    if mode == "smal33":
        anim, to_keep = remap_bvh_anim(anim, joint_names_no_root)
        kept_joint_names = ["Root"] + list(JOINTS_LIST)
    else:
        kept_joint_names = keep_names if keep_names else list(file_names)
        anim, to_keep = remap_anim_to_names(anim, file_names, kept_joint_names)
        process_landmarks = resolve_landmark_indices(kept_joint_names)

    if anim.positions.shape[0] <= 1:
        return None
    anim = apply_axis_transform_anim(anim, axis_transform)
    anim = apply_world_yaw_anim(anim, post_axis_yaw_deg)

    joints = Animation.positions_global(anim)
    joints = np.concatenate([joints, joints[-1:]], axis=0)
    new_joints, rotation = process_positions(
        joints, forward_mode=forward_mode, landmarks=process_landmarks
    )
    new_joints = new_joints[:, 3:]
    rotation = rotation[:-1]
    anim.rotations[:, 0, :] = rotation[:, 0, :] * anim.rotations[:, 0, :]
    angle = anim.rotations.qs.copy()

    anim.rotations.qs[...] = anim.orients.qs[None]
    tjoints = Animation.positions_global(anim)
    anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
    anim.positions[:, 0, :] = new_joints[:, :3]
    skel = anim.positions.copy()
    motion = {
        "quat": angle,
        "seq": new_joints,
        "skel": skel,
        "anim": anim,
        "names": names,
        "joint_names": kept_joint_names,
        "parents": np.asarray(anim.parents, dtype=np.int64),
        "ftime": ftime,
        "to_keep": to_keep,
        "_axis_transform": axis_transform,
        "_forward_mode": forward_mode,
        "_post_axis_yaw_deg": float(post_axis_yaw_deg or 0.0),
        "_skeleton_mode": mode,
    }
    if canonicalize_bind_pose:
        try:
            from .bind_pose_canonicalize_smal33 import canonicalize_motion_bind_pose
        except ImportError:
            from bind_pose_canonicalize_smal33 import canonicalize_motion_bind_pose

        motion, _ = canonicalize_motion_bind_pose(
            motion,
            forward_mode=forward_mode,
            dot_threshold=bind_pose_dot_threshold,
        )
    return motion


def load_shape_vector(shape_npz_path):
    payload = np.load(str(shape_npz_path))
    full_width = payload["full_width"].astype(np.single)
    joint_shape = payload["joint_shape"].astype(np.single)
    return np.divide(joint_shape, full_width[None, :]).reshape(-1)


def character_label_from_path(path):
    """Stable character label from an explicit .npz or BVH path."""
    return Path(path).stem


def resolve_shape_npz_path(shape_path=None, character=None, npz_path=None):
    """
    Resolve a shape .npz file.

    Priority:
      1. Explicit ``npz_path``
      2. ``shape_path / f"{character}.npz"`` with glob fallback
    """
    if npz_path is not None:
        resolved = Path(npz_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Shape npz not found: {resolved}")
        return resolved

    if shape_path is None or character is None:
        raise ValueError("Either npz_path or both shape_path and character are required.")

    root = Path(shape_path)
    direct = root / f"{character}.npz"
    if direct.exists():
        return direct

    candidates = sorted(root.glob(f"*{character}*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No shape npz for character '{character}' under {root}"
        )
    return candidates[0]


def load_shape_for_character(shape_path=None, character=None, npz_path=None):
    path = resolve_shape_npz_path(
        shape_path=shape_path,
        character=character,
        npz_path=npz_path,
    )
    return load_shape_vector(path)


def load_mesh_from_npz(
    npz_path,
    *,
    canonicalize_bind_pose=True,
    forward_mode="body",
    bind_pose_dot_threshold=0.9,
):
    """Load rest mesh assets for LBS visualization from a shape .npz file.

    When ``canonicalize_bind_pose`` is True (default), apply the same +Z bind-pose
    yaw used by ``get_inp_from_bvh`` so mesh vertices and ``skeleton`` stay in the
    motion/LBS frame. LBS must use this mesh ``skeleton`` as rest bind — not the
    BVH rest skel alone — otherwise a facing mismatch causes stretch ("拉皮").
    """
    path = resolve_shape_npz_path(npz_path=npz_path)
    data = np.load(str(path), allow_pickle=True)
    vertices = np.asarray(data["rest_vertices"], dtype=np.float32)
    faces = np.asarray(data["rest_faces"], dtype=np.int32)
    skin_weights = np.asarray(data["skinning_weights"], dtype=np.float32)
    skeleton = (
        np.asarray(data["skeleton"], dtype=np.float32)
        if "skeleton" in data.files
        else None
    )
    try:
        from .skeleton_io import decode_name_list
    except ImportError:
        from skeleton_io import decode_name_list

    joint_names = (
        decode_name_list(data["joint_names"]) if "joint_names" in data.files else None
    )
    topology = None
    if "topology" in data.files:
        topology = np.asarray(data["topology"], dtype=np.int64)
    elif "parents" in data.files:
        topology = np.asarray(data["parents"], dtype=np.int64)
    applied = False

    if canonicalize_bind_pose and skeleton is not None:
        try:
            from .bind_pose_canonicalize_smal33 import canonicalize_shape_npz_arrays
        except ImportError:
            from bind_pose_canonicalize_smal33 import canonicalize_shape_npz_arrays

        payload = {
            "skeleton": skeleton,
            "rest_vertices": vertices,
            "rest_faces": faces,
            "skinning_weights": skin_weights,
        }
        if joint_names is not None:
            payload["joint_names"] = np.asarray(joint_names)
        if topology is not None:
            payload["topology"] = topology
        for key in (
            "vertex_part",
            "rest_body_vertices",
            "rest_arm_vertices",
            "full_width",
            "body_width",
            "joint_shape",
        ):
            if key in data.files:
                payload[key] = np.asarray(data[key])
        payload, applied = canonicalize_shape_npz_arrays(
            payload,
            forward_mode=forward_mode,
            dot_threshold=float(bind_pose_dot_threshold),
        )
        vertices = np.asarray(payload["rest_vertices"], dtype=np.float32)
        skeleton = np.asarray(payload["skeleton"], dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "skin_weights": skin_weights,
        "skeleton": skeleton,
        "joint_names": joint_names,
        "topology": topology,
        "parents": topology,
        "bind_pose_canonicalized": bool(applied),
        "npz_path": str(path),
        "character": path.stem,
    }


def lbs_rest_skel_from_mesh(mesh_data, fallback_skel=None):
    """Prefer shape.npz skeleton (same frame as mesh vertices) for LBS rest bind."""
    skel = mesh_data.get("skeleton") if mesh_data is not None else None
    if skel is not None:
        return np.asarray(skel, dtype=np.float32)
    if fallback_skel is None:
        raise ValueError(
            "Mesh has no skeleton and no fallback_skel was provided for LBS."
        )
    return np.asarray(fallback_skel, dtype=np.float32)


def report_lbs_rest_skel_mismatch(
    mesh_skel,
    motion_skel,
    *,
    label="",
    warn_mean_threshold=0.05,
    mesh_canonicalized=None,
):
    """Log when BVH rest skel diverges from mesh bind (non-root offsets)."""
    if mesh_skel is None or motion_skel is None:
        return None
    mesh = np.asarray(mesh_skel, dtype=np.float64).reshape(-1, 3)
    motion = np.asarray(motion_skel, dtype=np.float64).reshape(-1, 3)
    if mesh.shape != motion.shape:
        print(
            f"[lbs-bind][{label}] shape mismatch mesh={mesh.shape} motion={motion.shape}"
        )
        return None
    dist = np.linalg.norm(mesh - motion, axis=-1)
    non_root = dist[1:] if len(dist) > 1 else dist
    stats = {
        "root_dist": float(dist[0]) if len(dist) else 0.0,
        "non_root_mean": float(non_root.mean()) if len(non_root) else 0.0,
        "non_root_max": float(non_root.max()) if len(non_root) else 0.0,
        "argmax": int(np.argmax(dist)) if len(dist) else -1,
    }
    prefix = f"[lbs-bind][{label}]" if label else "[lbs-bind]"
    canon_txt = (
        f", mesh_canonicalized={bool(mesh_canonicalized)}"
        if mesh_canonicalized is not None
        else ""
    )
    if stats["non_root_mean"] > float(warn_mean_threshold):
        print(
            f"{prefix} mesh/motion rest skel diverge "
            f"(non-root mean={stats['non_root_mean']:.4f}, "
            f"max={stats['non_root_max']:.4f} at joint {stats['argmax']}{canon_txt}). "
            "Using mesh skeleton for LBS."
        )
    else:
        print(
            f"{prefix} rest skel ok "
            f"(non-root mean={stats['non_root_mean']:.4f}, "
            f"root_dist={stats['root_dist']:.4f}{canon_txt})"
        )
    return stats


def load_mesh_for_character(shape_path=None, character=None, npz_path=None):
    path = resolve_shape_npz_path(
        shape_path=shape_path,
        character=character,
        npz_path=npz_path,
    )
    return load_mesh_from_npz(path)


def shape_roots_from_args(shape_path, inp_shape_path=None, tgt_shape_path=None):
    """Return (inp_root, tgt_root) for cross-retarget shape lookup."""
    default = Path(shape_path) if shape_path is not None else None
    inp_root = Path(inp_shape_path) if inp_shape_path is not None else default
    tgt_root = Path(tgt_shape_path) if tgt_shape_path is not None else default
    return inp_root, tgt_root


def load_mesh_npz_dict(*shape_dirs):
    """Load raw npz payloads keyed by stem from one or more shape directories."""
    mesh_file_dic = {}
    seen_dirs = []
    for shape_dir in shape_dirs:
        if shape_dir is None:
            continue
        root = Path(shape_dir)
        if root in seen_dirs:
            continue
        seen_dirs.append(root)
        if not root.exists():
            raise FileNotFoundError(f"Shape directory not found: {root}")
        for npz_path in sorted(root.glob("*.npz")):
            if npz_path.name.startswith("."):
                continue
            mesh_file_dic[npz_path.stem] = np.load(npz_path)
    if not mesh_file_dic:
        dirs = ", ".join(str(d) for d in seen_dirs)
        raise FileNotFoundError(f"No mesh npz files found under: {dirs}")
    return mesh_file_dic


def load_stats(stats_path):
    stats_path = Path(stats_path)
    out = {key: np.load(stats_path / fname) for key, fname in STATS_FILES.items()}
    out["local_std"] = out["local_std"].copy()
    out["local_std"][out["local_std"] == 0] = 1
    out["quat_std"] = out["quat_std"].copy()
    out["quat_std"][out["quat_std"] == 0] = 1
    return out


def get_height_from_skel(skel):
    diffs = np.sqrt((skel ** 2).sum(axis=-1))
    return (diffs[1:7].sum() + diffs[19:22].sum()) / 100.0


def normalize_motion(quat, local, skel, stats):
    local = (local - stats["local_mean"]) / stats["local_std"]
    skel = (skel - stats["local_mean"]) / stats["local_std"]
    quat = (quat - stats["quat_mean"]) / stats["quat_std"]
    return quat, local, skel


def build_model_inputs(motion, stats, shape_vec, device):
    quat = motion["quat"].astype(np.float32)
    seq = motion["seq"].astype(np.float32)
    skel = motion["skel"].astype(np.float32)

    local = np.reshape(seq[:, :-SEQ_TAIL_DIM], (seq.shape[0], NUM_JOINTS, 3))
    global_part = seq[:, -SEQ_TAIL_DIM:-4]
    height = np.array([get_height_from_skel(skel[0])], dtype=np.float32)

    quat, local, skel = normalize_motion(quat, local, skel, stats)

    seq_feat = np.concatenate([local.reshape(len(local), -1), global_part], axis=-1)
    return {
        "seq": torch.from_numpy(seq_feat)[None].float().to(device),
        "skel": torch.from_numpy(skel.reshape(len(skel), -1))[None].float().to(device),
        "quat": torch.from_numpy(quat)[None].float().to(device),
        "shape": torch.from_numpy(shape_vec.astype(np.float32))[None].float().to(device),
        "height": torch.from_numpy(height)[None].float().to(device),
        "global_raw": global_part,
        "local_norm": local,
    }


def load_retnet(weights_path, ret_model_args, device):
    import inspect as _inspect

    from src.model_skeleton_aware_smal33 import RetNet

    # Shape configs may share ret_model_args with stage-1 loading and include
    # shape-only keys (e.g. tail_dof_num) the skeleton RetNet does not accept.
    accepted = set(_inspect.signature(RetNet.__init__).parameters)
    skeleton_args = {k: v for k, v in ret_model_args.items() if k in accepted}
    model = RetNet(**skeleton_args).to(device)
    weights = torch.load(str(weights_path), map_location=device)
    cleaned = OrderedDict()
    for key, val in weights.items():
        cleaned[key.split("module.")[-1]] = val
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def load_shape_retnet(weights_path, ret_model_args, device):
    """Load shape-aware RetNet (stage-2); strict=False for shared stage-1 keys."""
    from src.model_shape_aware_smal33 import RetNet

    model = RetNet(**ret_model_args).to(device)
    weights = torch.load(str(weights_path), map_location=device)
    cleaned = OrderedDict()
    for key, val in weights.items():
        cleaned[key.split("module.")[-1]] = val
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def setup_cuda_device(device_id):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return torch.device("cuda:0")


def expand_anim_frames(anim, num_frames):
    """Resize animation frame count to match model output length."""
    anim = anim.copy()
    cur = anim.positions.shape[0]
    if cur == num_frames:
        return anim
    if cur == 1:
        anim.positions = np.repeat(anim.positions, num_frames, axis=0)
        anim.rotations.qs = np.repeat(anim.rotations.qs, num_frames, axis=0)
        return anim
    idx = np.linspace(0, cur - 1, num_frames).round().astype(int)
    anim.positions = anim.positions[idx]
    anim.rotations.qs = anim.rotations.qs[idx]
    return anim


def get_orient_start_smal33(reference):
    sdr_l, sdr_r, hip_l, hip_r = 8, 12, 18, 22
    across1 = reference[0:1, hip_l] - reference[0:1, hip_r]
    across0 = reference[0:1, sdr_l] - reference[0:1, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    return -rotation


def motion_local_positions(motion):
    """Canonical (T, 33, 3) joint positions stored in motion['seq']."""
    local_dim = NUM_JOINTS * 3
    return np.reshape(motion["seq"][:, :local_dim], (-1, NUM_JOINTS, 3))


def motion_global_part(motion):
    """Root velocity + yaw channels from motion['seq']."""
    return motion["seq"][:, -SEQ_TAIL_DIM : -SEQ_TAIL_DIM + GLOBAL_DIM]


def identity_start_rots(num_frames=1):
    """
    No-op orientation for put_in_world_bvh.

    Motion seq from get_inp_from_bvh is already canonicalized (Y-up, +Z forward)
    by process_positions; applying get_orient_start_* again causes 180/90 deg flips.
    """
    return Quaternions.id(num_frames)[:, np.newaxis]


def skeleton_offsets_to_global(offsets, parents):
    offsets = np.asarray(offsets, dtype=np.float32)
    world = np.zeros_like(offsets)
    for j, p in enumerate(parents):
        if p == -1:
            world[j] = offsets[j]
        else:
            world[j] = world[p] + offsets[j]
    return world


def bind_global_to_canonical_local(global_joints):
    """Apply the same floor/forward canonicalization as get_inp_from_bvh."""
    traj = np.asarray(global_joints, dtype=np.float32)
    if traj.ndim == 2:
        traj = traj[None, ...]
    traj = np.concatenate([traj, traj[-1:]], axis=0)
    canonical, _ = process_positions(traj)
    local_dim = NUM_JOINTS * 3
    return np.reshape(canonical[0, 3 : 3 + local_dim], (NUM_JOINTS, 3))


def build_motion_world(motion, start_rots=None):
    """Reconstruct world-space joint trajectories from a motion dict."""
    from src.utils import put_in_world_bvh

    local = motion_local_positions(motion)
    global_part = motion_global_part(motion)
    motion_total = np.concatenate(
        [local.reshape(len(local), -1), global_part],
        axis=-1,
    )
    if start_rots is None:
        start_rots = identity_start_rots(len(local))
    world, _ = put_in_world_bvh(motion_total.copy(), start_rots)
    return world[0], start_rots, motion_total


def rest_skel_to_world(skel_frame0, parents, start_rots=None):
    """
    Target bind-pose skeleton in the same Y-up / +Z world frame as build_motion_world.
    """
    from src.utils import put_in_world_bvh

    skel0 = np.asarray(skel_frame0, dtype=np.float32).reshape(NUM_JOINTS, 3)
    bind_global = skeleton_offsets_to_global(skel0, parents)
    local = bind_global_to_canonical_local(bind_global)
    if start_rots is None:
        start_rots = identity_start_rots(1)
    state = np.concatenate(
        [local.reshape(-1), np.zeros(GLOBAL_DIM, dtype=np.float32)],
        axis=-1,
    )[None]
    world, _ = put_in_world_bvh(state, start_rots)
    return world[0, 0].astype(np.float32)


def apply_inverse_axis_transform_points(points, axis_transform=None):
    matrix = np.linalg.inv(axis_transform_matrix(axis_transform))
    return np.einsum("ij,...j->...i", matrix, points)


def apply_inverse_axis_transform_quats(quats, axis_transform=None):
    matrix = np.linalg.inv(axis_transform_matrix(axis_transform))
    if np.allclose(matrix, np.eye(3)):
        return quats.copy()
    rotations = Quaternions(quats).transforms()
    transformed = np.einsum("ij,...jk,lk->...il", matrix, rotations, matrix)
    return Quaternions.from_transforms(transformed).normalized().qs


def retarget_to_bvh(
    inp_motion,
    tgt_motion,
    local_rt,
    global_rt,
    quat_rt,
    stats,
    save_dir,
    pair_tag,
    inp_bvh_path=None,
    tgt_bvh_path=None,
    local_is_normalized=True,
):
    from src.utils import put_in_world_bvh

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]

    if local_is_normalized:
        ours_l = local_rt * local_std + local_mean
    else:
        ours_l = np.asarray(local_rt, dtype=np.float32)
    ours_g = global_rt
    ours_total = np.concatenate([ours_l.reshape(len(local_rt), -1), ours_g], axis=-1)
    num_frames = len(ours_total)

    tgt_to_keep = tgt_motion["to_keep"]
    if inp_bvh_path is not None and inp_motion.get("_axis_transform") in (None, "", "none"):
        inp_anim, inp_names, inp_ftime = BVH.load(str(inp_bvh_path))
    else:
        inp_anim = inp_motion["anim"].copy()
        inp_names = inp_motion["names"]
        inp_ftime = inp_motion["ftime"]

    if tgt_bvh_path is not None and tgt_motion.get("_axis_transform") in (None, "", "none"):
        tgt_anim, tgt_names, tgt_ftime = BVH.load(str(tgt_bvh_path))
    else:
        tgt_anim = tgt_motion["anim"].copy()
        tgt_names = tgt_motion["names"]
        tgt_ftime = tgt_motion["ftime"]

    tgt_anim_rest = tgt_anim.copy()
    tgt_anim = expand_anim_frames(tgt_anim, num_frames)

    start_rots = identity_start_rots(num_frames)
    # tgt_motion["skel"] stores target BVH offsets in motion units already.
    # Applying dataset local stats here corrupts exported BVH bone lengths.
    tjoints = tgt_motion["skel"][0:1].astype(np.float32)
    tjoints = np.repeat(tjoints, num_frames, axis=0)

    output_bvh = ours_total.copy()
    output_bvh[:, -4:] = output_bvh[:, -4:] * (
        np.sign(inp_motion["seq"][:, -8:-4]) * np.sign(output_bvh[:, -4:])
    )
    output_bvh[:, -3][np.abs(inp_motion["seq"][:, -8:-4][:, 2]) <= 1e-2] = 0.0
    output_bvh[:, :3] = tgt_anim.positions[:1, 0, :].copy()

    wjs, rots = put_in_world_bvh(output_bvh.copy(), start_rots)
    tjoints[:, 0, :] = wjs[0, :, 0].copy()

    cquat = quat_rt[:, :NUM_JOINTS].copy()

    inp_copy = save_dir / f"{pair_tag}_input.bvh"
    tgt_copy = save_dir / f"{pair_tag}_target_rest.bvh"
    out_copy = save_dir / f"{pair_tag}_retarget.bvh"

    BVH.save(str(inp_copy), inp_anim, inp_names, inp_ftime)
    BVH.save(str(tgt_copy), tgt_anim_rest, tgt_names, tgt_ftime)

    tgt_anim.positions[:, tgt_to_keep] = tjoints
    tgt_anim.offsets[tgt_to_keep[1:]] = tjoints[0, 1:]
    cquat[:, 0:1, :] = (rots * Quaternions(cquat[:, 0:1, :])).qs
    tgt_anim.rotations.qs[:, tgt_to_keep] = cquat
    BVH.save(str(out_copy), tgt_anim, tgt_names, tgt_ftime)
    return inp_copy, tgt_copy, out_copy


def retarget_to_native_bvh(
    inp_motion,
    tgt_motion,
    local_rt,
    global_rt,
    quat_rt,
    stats,
    save_dir,
    pair_tag,
    tgt_bvh_path,
    axis_transform,
    local_is_normalized=True,
):
    """
    Write retarget animation onto the RAW (FBX-native) target BVH rest pose.

    Inference runs in ``axis_transform`` space (e.g. shepherd_y_z_x). This writer
    inverse-transforms local quaternions / root translation back to the native
    BVH frame and keeps the on-disk target offsets untouched so Blender can bind
    the result to the textured FBX armature.
    """
    from src.utils import put_in_world_bvh

    if tgt_bvh_path is None:
        raise ValueError("tgt_bvh_path is required for native BVH export")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]
    if local_is_normalized:
        ours_l = local_rt * local_std + local_mean
    else:
        ours_l = np.asarray(local_rt, dtype=np.float32)
    ours_g = np.asarray(global_rt, dtype=np.float32)
    ours_total = np.concatenate([ours_l.reshape(len(local_rt), -1), ours_g], axis=-1)
    num_frames = len(ours_total)

    tgt_to_keep = tgt_motion["to_keep"]
    # Always load the on-disk target BVH: this is the FBX-native rest hierarchy.
    native_anim, native_names, native_ftime = BVH.load(str(tgt_bvh_path))
    native_rest = native_anim.copy()
    native_anim = expand_anim_frames(native_anim, num_frames)

    # Model-space root + local quat (same bookkeeping as retarget_to_bvh).
    start_rots = identity_start_rots(num_frames)

    output_bvh = ours_total.copy()
    output_bvh[:, -4:] = output_bvh[:, -4:] * (
        np.sign(inp_motion["seq"][:, -8:-4]) * np.sign(output_bvh[:, -4:])
    )
    output_bvh[:, -3][np.abs(inp_motion["seq"][:, -8:-4][:, 2]) <= 1e-2] = 0.0
    # Use model-space target root rest for put_in_world consistency.
    model_root_rest = tgt_motion["anim"].positions[:1, 0, :].copy()
    output_bvh[:, :3] = model_root_rest

    wjs, rots = put_in_world_bvh(output_bvh.copy(), start_rots)
    root_model = wjs[0, :, 0].copy()  # (T, 3) model space
    cquat_model = quat_rt[:, :NUM_JOINTS].copy()
    cquat_model[:, 0:1, :] = (rots * Quaternions(cquat_model[:, 0:1, :])).qs

    # Inverse axis transform -> FBX / raw BVH coordinates.
    cquat_native = apply_inverse_axis_transform_quats(cquat_model, axis_transform)
    root_native = apply_inverse_axis_transform_points(root_model, axis_transform)

    # Keep native bone offsets; only replace motion channels on remapped joints.
    native_anim.rotations.qs[:, tgt_to_keep] = cquat_native.astype(np.float32)
    # Root translation in native space; other joint positions stay as rest offsets.
    native_anim.positions[:, 0, :] = root_native.astype(np.float32)
    for j_idx in tgt_to_keep[1:]:
        # Preserve rest local offsets as constant joint positions for non-root.
        native_anim.positions[:, j_idx, :] = native_rest.offsets[j_idx][None, :]

    out_copy = save_dir / f"{pair_tag}_native_retarget.bvh"
    rest_copy = save_dir / f"{pair_tag}_native_target_rest.bvh"
    BVH.save(str(rest_copy), native_rest, native_names, native_ftime)
    BVH.save(str(out_copy), native_anim, native_names, native_ftime)
    return rest_copy, out_copy


def world_joints_from_motion(local_rt, global_rt, stats, start_rots=None):
    from src.utils import put_in_world_bvh

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]
    ours_l = local_rt * local_std + local_mean
    ours_total = np.concatenate([ours_l.reshape(len(local_rt), -1), global_rt], axis=-1)
    if start_rots is None:
        start_rots = identity_start_rots(len(local_rt))
    wjs, _ = put_in_world_bvh(ours_total.copy(), start_rots)
    return wjs[0]


def save_skeleton_video(animations, parents, save_path, interval=33.33):
    from src.utils import animation_plot

    animation_plot(animations, str(save_path), parents, interval=interval)


def list_sequences(data_path, min_frames=32):
    data_path = Path(data_path)
    samples = []
    for char_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
        for seq_path in sorted(char_dir.glob("*_seq.npy")):
            stem = seq_path.name[: -len("_seq.npy")]
            skel_path = char_dir / f"{stem}_skel.npy"
            quat_path = char_dir / f"{stem}_quat.npy"
            if not skel_path.exists() or not quat_path.exists():
                continue
            seq = np.load(seq_path)
            if seq.shape[0] < min_frames:
                continue
            samples.append(
                {
                    "character": char_dir.name,
                    "sequence": stem,
                    "seq_path": seq_path,
                    "skel_path": skel_path,
                    "quat_path": quat_path,
                    "num_frames": int(seq.shape[0]),
                }
            )
    return samples


def split_characters(samples, val_ratio=0.1, seed=3047):
    chars = sorted({s["character"] for s in samples})
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(len(chars) * val_ratio)))
    val_chars = set(rng.choice(chars, size=n_val, replace=False).tolist())
    train_chars = [c for c in chars if c not in val_chars]
    return train_chars, sorted(val_chars)


def load_window_from_sample(sample, stats, max_length, start_idx=None, rng=None):
    seq = np.load(sample["seq_path"])
    skel = np.load(sample["skel_path"])
    quat = np.load(sample["quat_path"])
    seq_len = seq.shape[0]
    if seq_len > max_length:
        if start_idx is None:
            start_idx = 0 if rng is None else int(rng.randint(0, seq_len - max_length))
        end = start_idx + max_length
    else:
        start_idx = 0
        end = seq_len
    mask_len = end - start_idx

    seq_w = seq[start_idx:end]
    skel_w = skel[start_idx:end]
    quat_w = quat[start_idx:end]

    local = np.reshape(seq_w[:, :-SEQ_TAIL_DIM], (seq_w.shape[0], NUM_JOINTS, 3))
    global_part = seq_w[:, -SEQ_TAIL_DIM:-4]
    quat_n, local_n, skel_n = normalize_motion(quat_w, local, skel_w, stats)

    height = get_height_from_skel(skel_w[0])

    seq_feat = np.concatenate([local_n.reshape(len(local_n), -1), global_part], axis=-1)
    local_gt = local_n.copy()

    return {
        "seq": seq_feat.astype(np.float32),
        "skel": skel_n.reshape(len(skel_n), -1).astype(np.float32),
        "quat": quat_n.astype(np.float32),
        "local_gt": local_gt.astype(np.float32),
        "global_in": global_part.astype(np.float32),
        "height": np.array([height], dtype=np.float32),
        "mask_len": mask_len,
        "character": sample["character"],
        "sequence": sample["sequence"],
    }


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
