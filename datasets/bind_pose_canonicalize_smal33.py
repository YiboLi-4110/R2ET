"""
Align SMAL33 bind-pose skeleton geometry to the shepherd canonical frame.

Shepherd motion uses +Z forward / +X left-right in bind-pose offset space.
batch2_dogs rest assets exported from Blender often arrive with forward along
-X after axis_transform + process_positions, while seq local motion is already
+Z aligned. This module rotates skel offsets (and root quaternions) so FK refB
matches the same convention as shepherd.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DATASETS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASETS_DIR.parent
OUTSIDE_CODE = REPO_ROOT / "outside-code"
if str(OUTSIDE_CODE) not in sys.path:
    sys.path.insert(0, str(OUTSIDE_CODE))
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from Quaternions import Quaternions  # noqa: E402

from smal33_motion_io import NUM_JOINTS, SMAL33_PARENTS  # noqa: E402

try:
    from skeleton_io import decode_name_list, is_smal33_names, resolve_landmark_indices
except ImportError:
    from datasets.skeleton_io import (
        decode_name_list,
        is_smal33_names,
        resolve_landmark_indices,
    )

JOINT_LEFT_SCAPULA, JOINT_RIGHT_SCAPULA = 8, 12
JOINT_LEFT_THIGH, JOINT_RIGHT_THIGH = 18, 22
CANONICAL_FORWARD_XZ = np.array([0.0, 1.0], dtype=np.float64)


def _parents_for_skel(skel: np.ndarray, parents: np.ndarray | None) -> np.ndarray:
    joint_count = int(np.asarray(skel).reshape(-1, 3).shape[0])
    if parents is not None:
        parents_np = np.asarray(parents, dtype=np.int64).reshape(-1)
        if parents_np.shape[0] == joint_count:
            return parents_np
    if joint_count == NUM_JOINTS:
        return np.asarray(SMAL33_PARENTS, dtype=np.int64)
    raise ValueError(
        f"bind-pose canonicalize needs parents for a {joint_count}-joint skeleton"
    )


def _landmarks_for_skel(
    skel: np.ndarray,
    joint_names: list[str] | None,
) -> dict[str, int]:
    joint_count = int(np.asarray(skel).reshape(-1, 3).shape[0])
    names = decode_name_list(joint_names) if joint_names is not None else None
    if names and not is_smal33_names(names):
        resolved = resolve_landmark_indices(names)
        return {
            "sdr_l": int(resolved["sdr_l"]),
            "sdr_r": int(resolved["sdr_r"]),
            "hip_l": int(resolved["hip_l"]),
            "hip_r": int(resolved["hip_r"]),
        }
    if joint_count == NUM_JOINTS:
        return {
            "sdr_l": JOINT_LEFT_SCAPULA,
            "sdr_r": JOINT_RIGHT_SCAPULA,
            "hip_l": JOINT_LEFT_THIGH,
            "hip_r": JOINT_RIGHT_THIGH,
        }
    if names:
        resolved = resolve_landmark_indices(names)
        return {
            "sdr_l": int(resolved["sdr_l"]),
            "sdr_r": int(resolved["sdr_r"]),
            "hip_l": int(resolved["hip_l"]),
            "hip_r": int(resolved["hip_r"]),
        }
    raise ValueError(
        f"Cannot resolve bind-pose landmarks for J={joint_count} without joint_names"
    )


def offsets_to_global(offsets: np.ndarray, parents: np.ndarray | None = None) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    parents_np = _parents_for_skel(offsets, parents)
    out = np.zeros_like(offsets)
    for idx, parent in enumerate(parents_np):
        if parent == -1:
            out[idx] = offsets[idx]
        else:
            out[idx] = out[parent] + offsets[idx]
    return out


def bind_process_forward(
    skel_frame0: np.ndarray,
    forward_mode: str = "body",
    *,
    joint_names: list[str] | None = None,
    parents: np.ndarray | None = None,
) -> np.ndarray:
    global_joints = offsets_to_global(skel_frame0, parents=parents)
    marks = _landmarks_for_skel(skel_frame0, joint_names)
    across = (global_joints[marks["hip_l"]] - global_joints[marks["hip_r"]]) + (
        global_joints[marks["sdr_l"]] - global_joints[marks["sdr_r"]]
    )
    across = across / (np.linalg.norm(across) + 1e-8)
    forward = np.cross(across, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if forward_mode == "body":
        shoulder = 0.5 * (
            global_joints[marks["sdr_l"]] + global_joints[marks["sdr_r"]]
        )
        hip = 0.5 * (global_joints[marks["hip_l"]] + global_joints[marks["hip_r"]])
        body = shoulder - hip
        body[1] = 0.0
        norm = np.linalg.norm(body)
        if norm > 1e-8:
            forward = body / norm
    elif forward_mode != "across":
        raise ValueError(f"Unknown forward_mode '{forward_mode}'")
    return forward.astype(np.float64)


def bind_forward_dot(
    skel_frame0: np.ndarray,
    forward_mode: str = "body",
    *,
    joint_names: list[str] | None = None,
    parents: np.ndarray | None = None,
) -> float:
    forward = bind_process_forward(
        skel_frame0,
        forward_mode=forward_mode,
        joint_names=joint_names,
        parents=parents,
    )
    forward_xz = forward[[0, 2]]
    forward_xz = forward_xz / (np.linalg.norm(forward_xz) + 1e-8)
    return float(np.dot(forward_xz, CANONICAL_FORWARD_XZ))


def yaw_correction_quaternion(
    skel_frame0: np.ndarray,
    forward_mode: str = "body",
    dot_threshold: float = 0.9,
    *,
    joint_names: list[str] | None = None,
    parents: np.ndarray | None = None,
) -> Quaternions | None:
    kwargs = {"joint_names": joint_names, "parents": parents}
    dot = bind_forward_dot(skel_frame0, forward_mode=forward_mode, **kwargs)
    if dot >= dot_threshold:
        return None
    current = bind_process_forward(skel_frame0, forward_mode=forward_mode, **kwargs)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    corr = Quaternions.between(current, target)
    return corr[0:1]


def rotation_matrix_from_quaternion(q: Quaternions) -> np.ndarray:
    rot = q.qs.reshape(-1, 4)
    qw, qx, qy, qz = rot[0]
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotate_vectors(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return np.einsum("ij,...j->...i", matrix, vectors)


def apply_yaw_correction_to_skel(skel: np.ndarray, q_corr: Quaternions) -> np.ndarray:
    matrix = rotation_matrix_from_quaternion(q_corr)
    skel = np.asarray(skel, dtype=np.float32).copy()
    skel = rotate_vectors(matrix, skel).astype(np.float32)
    return skel


def apply_yaw_correction_to_quat(quat: np.ndarray, q_corr: Quaternions) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    T = quat.shape[0]
    # (1, 4) -> (1, 1, 4)，可与 (T, 1, 4) 广播
    q_corr_b = Quaternions(q_corr.qs.reshape(1, 1, 4))
    root = Quaternions(quat[:, 0:1, :])
    corrected = (q_corr_b * root).qs.astype(np.float32)
    quat[:, 0:1, :] = corrected
    return quat


def canonicalize_motion_bind_pose(
    motion: dict,
    forward_mode: str | None = None,
    dot_threshold: float = 0.9,
) -> tuple[dict, bool]:
    """
    Rotate motion['skel'] bind geometry to +Z forward when misaligned.

    seq local positions are already canonicalized by process_positions and are
    left unchanged. Root quaternions are corrected so bind FK stays consistent.
    """
    forward_mode = forward_mode or motion.get("_forward_mode", "body")
    skel = motion["skel"]
    joint_names = motion.get("joint_names")
    parents = motion.get("parents")
    if parents is None and motion.get("anim") is not None:
        parents = np.asarray(motion["anim"].parents, dtype=np.int64)
    q_corr = yaw_correction_quaternion(
        skel[0],
        forward_mode=forward_mode,
        dot_threshold=dot_threshold,
        joint_names=joint_names,
        parents=parents,
    )
    if q_corr is None:
        return motion, False

    motion = dict(motion)
    motion["skel"] = apply_yaw_correction_to_skel(skel, q_corr)
    if "quat" in motion:
        motion["quat"] = apply_yaw_correction_to_quat(motion["quat"], q_corr)
    motion["_bind_pose_canonicalized"] = True
    return motion, True


def canonicalize_skel_quat_arrays(
    skel: np.ndarray,
    quat: np.ndarray | None = None,
    forward_mode: str = "body",
    dot_threshold: float = 0.9,
    *,
    joint_names: list[str] | None = None,
    parents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    q_corr = yaw_correction_quaternion(
        skel[0],
        forward_mode=forward_mode,
        dot_threshold=dot_threshold,
        joint_names=joint_names,
        parents=parents,
    )
    if q_corr is None:
        return skel, quat, False
    skel = apply_yaw_correction_to_skel(skel, q_corr)
    if quat is not None:
        quat = apply_yaw_correction_to_quat(quat, q_corr)
    return skel, quat, True


def get_width(vertices: np.ndarray) -> np.ndarray:
    box_min = vertices.min(axis=0)
    box_max = vertices.max(axis=0)
    return box_max - box_min


def recompute_joint_shape(rest_vertices: np.ndarray, vertex_part: np.ndarray, num_joints: int | None = None):
    if num_joints is None:
        num_joints = int(np.max(vertex_part) + 1) if len(vertex_part) else NUM_JOINTS
    shape_lst = []
    for joint_idx in range(num_joints):
        mask = vertex_part == joint_idx
        if not np.any(mask):
            shape_lst.append(np.zeros(3, dtype=np.float32))
        else:
            shape_lst.append(get_width(rest_vertices[mask]).astype(np.float32))
    return np.stack(shape_lst, axis=0)


def canonicalize_shape_npz_arrays(
    payload: dict,
    forward_mode: str = "body",
    dot_threshold: float = 0.9,
) -> tuple[dict, bool]:
    skeleton = payload["skeleton"].astype(np.float64)
    joint_names = decode_name_list(payload.get("joint_names"))
    parents = payload.get("topology", payload.get("parents"))
    q_corr = yaw_correction_quaternion(
        skeleton[0] if skeleton.ndim == 3 else skeleton,
        forward_mode=forward_mode,
        dot_threshold=dot_threshold,
        joint_names=joint_names,
        parents=parents,
    )
    if q_corr is None:
        return payload, False

    matrix = rotation_matrix_from_quaternion(q_corr)
    out = {key: payload[key] for key in payload}

    skeleton_arr = np.asarray(payload["skeleton"], dtype=np.float32)
    if skeleton_arr.ndim == 2:
        out["skeleton"] = rotate_vectors(matrix, skeleton_arr).astype(np.float32)
    else:
        out["skeleton"] = rotate_vectors(matrix, skeleton_arr).astype(np.float32)

    rest_vertices = rotate_vectors(matrix, np.asarray(payload["rest_vertices"], dtype=np.float64))
    out["rest_vertices"] = rest_vertices.astype(np.float32)
    out["full_width"] = get_width(rest_vertices).astype(np.float32)

    if "vertex_part" in payload:
        out["joint_shape"] = recompute_joint_shape(
            rest_vertices,
            payload["vertex_part"],
            num_joints=int(np.asarray(payload["skeleton"]).reshape(-1, 3).shape[0]),
        ).astype(np.float32)

    if "rest_body_vertices" in payload:
        out["rest_body_vertices"] = rotate_vectors(
            matrix, np.asarray(payload["rest_body_vertices"], dtype=np.float64)
        ).astype(np.float32)
        out["body_width"] = get_width(out["rest_body_vertices"]).astype(np.float32)

    if "rest_arm_vertices" in payload:
        out["rest_arm_vertices"] = rotate_vectors(
            matrix, np.asarray(payload["rest_arm_vertices"], dtype=np.float64)
        ).astype(np.float32)

    return out, True
