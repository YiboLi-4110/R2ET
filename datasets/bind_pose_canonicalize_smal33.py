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

JOINT_LEFT_SCAPULA, JOINT_RIGHT_SCAPULA = 8, 12
JOINT_LEFT_THIGH, JOINT_RIGHT_THIGH = 18, 22
CANONICAL_FORWARD_XZ = np.array([0.0, 1.0], dtype=np.float64)


def offsets_to_global(offsets: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    out = np.zeros_like(offsets)
    for idx, parent in enumerate(SMAL33_PARENTS):
        if parent == -1:
            out[idx] = offsets[idx]
        else:
            out[idx] = out[parent] + offsets[idx]
    return out


def bind_process_forward(skel_frame0: np.ndarray, forward_mode: str = "body") -> np.ndarray:
    global_joints = offsets_to_global(skel_frame0)
    across = (global_joints[JOINT_LEFT_THIGH] - global_joints[JOINT_RIGHT_THIGH]) + (
        global_joints[JOINT_LEFT_SCAPULA] - global_joints[JOINT_RIGHT_SCAPULA]
    )
    across = across / (np.linalg.norm(across) + 1e-8)
    forward = np.cross(across, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if forward_mode == "body":
        shoulder = 0.5 * (global_joints[JOINT_LEFT_SCAPULA] + global_joints[JOINT_RIGHT_SCAPULA])
        hip = 0.5 * (global_joints[JOINT_LEFT_THIGH] + global_joints[JOINT_RIGHT_THIGH])
        body = shoulder - hip
        body[1] = 0.0
        norm = np.linalg.norm(body)
        if norm > 1e-8:
            forward = body / norm
    elif forward_mode != "across":
        raise ValueError(f"Unknown forward_mode '{forward_mode}'")
    return forward.astype(np.float64)


def bind_forward_dot(skel_frame0: np.ndarray, forward_mode: str = "body") -> float:
    forward = bind_process_forward(skel_frame0, forward_mode=forward_mode)
    forward_xz = forward[[0, 2]]
    forward_xz = forward_xz / (np.linalg.norm(forward_xz) + 1e-8)
    return float(np.dot(forward_xz, CANONICAL_FORWARD_XZ))


def yaw_correction_quaternion(
    skel_frame0: np.ndarray,
    forward_mode: str = "body",
    dot_threshold: float = 0.9,
) -> Quaternions | None:
    dot = bind_forward_dot(skel_frame0, forward_mode=forward_mode)
    if dot >= dot_threshold:
        return None
    current = bind_process_forward(skel_frame0, forward_mode=forward_mode)
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
    q_corr = yaw_correction_quaternion(
        skel[0], forward_mode=forward_mode, dot_threshold=dot_threshold
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
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    q_corr = yaw_correction_quaternion(
        skel[0], forward_mode=forward_mode, dot_threshold=dot_threshold
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


def recompute_joint_shape(rest_vertices: np.ndarray, vertex_part: np.ndarray, num_joints: int = NUM_JOINTS):
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
    q_corr = yaw_correction_quaternion(
        skeleton[0] if skeleton.ndim == 3 else skeleton,
        forward_mode=forward_mode,
        dot_threshold=dot_threshold,
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
            rest_vertices, payload["vertex_part"], num_joints=NUM_JOINTS
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
