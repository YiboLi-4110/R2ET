#!/usr/bin/env python3
"""Find motion config that adapts cat_actions → batch3_dogs under direct CopyQuat.

Uses numpy FK (no CUDA). Compares source intent vs CopyQuat-on-target-rest
limb/root motion in the body frame after facing canon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO), str(_REPO / "outside-code")]

from Quaternions import Quaternions  # noqa: E402
from datasets.smal33_motion_io import (  # noqa: E402
    SMAL33_PARENTS,
    get_inp_from_bvh,
    get_height_from_skel,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
)

# Remapped joint indices (Root=0): LeftUpperArm=8, LeftForeLeg=9, LeftFrontPaw=10
PAW_L, FORE_L, UPPER_L = 10, 9, 8
PAW_R = 14


def q_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, last dim=4 as (w,x,y,z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def q_rot_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    zeros = np.zeros(q.shape[:-1] + (1,), dtype=q.dtype)
    vq = np.concatenate([zeros, v], axis=-1)
    return q_mul(q_mul(q, vq), q_conj(q))[..., 1:]


def q_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def fk_positions(rest_skel: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """rest_skel [J,3] offsets/positions rest; quat [T,J,4] -> global pos [T,J,3]."""
    rest = np.asarray(rest_skel, dtype=np.float64)
    quat = q_normalize(np.asarray(quat, dtype=np.float64))
    t, j, _ = quat.shape
    parents = np.asarray(SMAL33_PARENTS, dtype=np.int64)
    # Local translation: rest joint position relative to parent (skel from get_skel)
    local_pos = rest.copy()
    for i in range(1, j):
        p = int(parents[i])
        if p >= 0:
            local_pos[i] = rest[i] - rest[p]

    glob_q = np.zeros_like(quat)
    glob_p = np.zeros((t, j, 3), dtype=np.float64)
    glob_q[:, 0] = quat[:, 0]
    glob_p[:, 0] = local_pos[0]  # will overwrite with animated root below
    # Root translation often stored separately; for relative limb analysis we
    # put rest root and rely on body-local metrics.
    for i in range(j):
        p = int(parents[i])
        if p < 0:
            glob_q[:, i] = quat[:, i]
            glob_p[:, i] = np.broadcast_to(local_pos[i], (t, 3))
            continue
        glob_q[:, i] = q_mul(glob_q[:, p], quat[:, i])
        glob_p[:, i] = glob_p[:, p] + q_rot_vec(glob_q[:, p], np.broadcast_to(local_pos[i], (t, 3)))
    return glob_p


def body_frame(joints: np.ndarray):
    """joints [T,J,3] -> fwd, up, right unit vectors [T,3] (Y-up world preferred)."""
    sdr = 0.5 * (joints[:, 9] + joints[:, 13])
    hip = 0.5 * (joints[:, 19] + joints[:, 23])
    fwd = sdr - hip
    fwd[:, 1] = 0.0
    n = np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-12
    fwd = fwd / n
    up = np.zeros_like(fwd)
    up[:, 1] = 1.0
    right = np.cross(up, fwd)
    rn = np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12
    right = right / rn
    # re-orthogonalize up
    up = np.cross(fwd, right)
    return fwd, up, right


def project_series(vec: np.ndarray, fwd, up, right):
    return np.stack(
        [
            np.einsum("tj,tj->t", vec, fwd),
            np.einsum("tj,tj->t", vec, up),
            np.einsum("tj,tj->t", vec, right),
        ],
        axis=-1,
    )


def summarize_clip(label: str, joints: np.ndarray, global_vel_y: float | None = None):
    fwd, up, right = body_frame(joints)
    # root proxy: joint 0 displacement in body frame of frame 0
    root_disp = joints[-1, 0] - joints[0, 0]
    f0, u0, r0 = fwd[0], up[0], right[0]
    root_b = np.array([root_disp @ f0, root_disp @ u0, root_disp @ r0])

    paw = np.diff(joints[:, PAW_L], axis=0)
    mid = slice(len(paw) // 4, max(len(paw) // 4 + 1, 3 * len(paw) // 4))
    paw_mid = paw[mid]
    f, u, r = fwd[mid], up[mid], right[mid]
    # align lengths: fwd[mid] is T_mid, paw_mid is T_mid
    paw_b = project_series(paw_mid, f, u, r).mean(axis=0)

    # elbow bend plane: Upper->Fore vs Fore->Paw, preference along fwd vs right
    mid_i = len(joints) // 2
    b1 = joints[mid_i, FORE_L] - joints[mid_i, UPPER_L]
    b2 = joints[mid_i, PAW_L] - joints[mid_i, FORE_L]
    swing_axis = np.cross(b1, b2)
    sa = swing_axis / (np.linalg.norm(swing_axis) + 1e-12)
    # sagittal swing axis ~ right; coronal swing axis ~ fwd
    sag = abs(float(sa @ right[mid_i]))
    cor = abs(float(sa @ fwd[mid_i]))

    facing = fwd[mid_i]
    print(
        f"{label:42s} root_body(f/u/r)={root_b[0]:+.3f}/{root_b[1]:+.3f}/{root_b[2]:+.3f} "
        f"paw_vel(f/u/r)={paw_b[0]:+.4f}/{paw_b[1]:+.4f}/{paw_b[2]:+.4f} "
        f"swing(sag/cor)={sag:.2f}/{cor:.2f} facing_xz=({facing[0]:+.2f},{facing[2]:+.2f})"
        + (f" gvy={global_vel_y:+.4f}" if global_vel_y is not None else "")
    )
    return {
        "root_b": root_b,
        "paw_b": paw_b,
        "sag": sag,
        "cor": cor,
        "facing": facing,
        "gvy": global_vel_y,
    }


def copyquat_on_target(inp_motion, tgt_rest_skel: np.ndarray):
    quat = inp_motion["quat"].astype(np.float64)
    # Apply scaled root translation into FK by shifting joint 0 each frame from seq
    # seq layout after process_positions: flattened joints then velocity...
    # For limb-relative analysis, FK with rest root is enough if we compare body frame.
    # Still apply global Y from seq velocity cumulative as optional root path.
    global_in = inp_motion["seq"][:, -8:-4].astype(np.float64)
    h_in = float(get_height_from_skel(inp_motion["skel"][0]))
    h_tgt = float(get_height_from_skel(tgt_rest_skel))
    scale = 1.0 if abs(h_in) < 1e-8 else (h_tgt / h_in)
    gvy = float(np.mean(global_in[:, 1])) * scale

    joints = fk_positions(tgt_rest_skel, quat)
    # Integrate scaled horizontal/vertical root velocity onto joint 0 for travel metrics
    vel = global_in[:, :3] * scale
    root = np.cumsum(vel, axis=0)
    root = np.vstack([np.zeros((1, 3)), root[:-1]]) if len(root) else root
    if len(root) == len(joints):
        delta = root - joints[:, 0]
        joints = joints + delta[:, None, :]
    return joints, gvy


def score_against_ref(ref: dict, cur: dict) -> dict:
    """Higher better. Prefer paw_fwd agreement, paw_up agreement, sagittal swing, facing +Z."""
    paw_dot_fu = float(
        np.dot(ref["paw_b"][:2] / (np.linalg.norm(ref["paw_b"][:2]) + 1e-12),
               cur["paw_b"][:2] / (np.linalg.norm(cur["paw_b"][:2]) + 1e-12))
    )
    root_up_agree = float(np.sign(ref["root_b"][1] + 1e-9) == np.sign(cur["root_b"][1] + 1e-9))
    if abs(ref["root_b"][1]) < 1e-3 and abs(cur["root_b"][1]) < 1e-3:
        root_up_agree = 1.0
    lateral_penalty = abs(cur["paw_b"][2]) / (abs(cur["paw_b"][0]) + abs(cur["paw_b"][1]) + 1e-6)
    sag_ratio = cur["sag"] / (cur["sag"] + cur["cor"] + 1e-6)
    facing_z = float(cur["facing"][2])  # want +1 after canon ideally; dog often -1 consistently
    return {
        "paw_fu_dot": paw_dot_fu,
        "root_up_agree": root_up_agree,
        "lateral_ratio": lateral_penalty,
        "sag_ratio": sag_ratio,
        "facing_z": facing_z,
        "gvy_sign_agree": float(np.sign(ref["gvy"] or 0) == np.sign(cur["gvy"] or 0))
        if ref["gvy"] is not None
        else float("nan"),
    }


def main():
    cfg = yaml.safe_load((_REPO / "config/visualization_arp_sequence_smal33.yaml").read_text())
    motion = cfg.get("motion") or {}
    print("Current config motion:", {k: motion.get(k) for k in [
        "inp_axis_transform", "inp_forward_mode", "inp_post_axis_yaw_deg",
        "tgt_axis_transform", "tgt_forward_mode", "tgt_post_axis_yaw_deg",
    ]})

    src_swim = _REPO / "datasets/shepherd/cat_actions/train_char/sand_cat_juvenile/sand_cat_juvenile@deepswim02_smal.bvh"
    src_walk = _REPO / "datasets/shepherd/cat_actions/train_char/sand_cat_female/sand_cat_male@walkbase_smal.bvh"
    tgt_bvh = _REPO / Path(cfg["target"]["tgt_bvh_path"])
    shape = _REPO / "datasets/shepherd/batch3_dogs/batch3_dogs_shape/lihua_1_lihua-0.npz"

    tgt_axis = motion.get("tgt_axis_transform", "shepherd_y_z_x")
    tgt_fwd = motion.get("tgt_forward_mode", "body")
    tgt_yaw = float(motion.get("tgt_post_axis_yaw_deg", 0) or 0)

    tgt_m = get_inp_from_bvh(
        str(tgt_bvh),
        axis_transform=tgt_axis,
        forward_mode=tgt_fwd,
        post_axis_yaw_deg=tgt_yaw,
    )
    mesh = load_mesh_from_npz(
        str(shape),
        forward_mode=tgt_fwd,
    )
    rest = lbs_rest_skel_from_mesh(mesh, fallback_skel=tgt_m["skel"][0])
    print(f"Target rest from mesh canonicalized={mesh.get('bind_pose_canonicalized')} "
          f"shape={rest.shape}")

    candidates = [
        # (axis, forward, yaw, note)
        ("none", "across", 0, "prev_best_facing"),
        ("none", "body", 0, "none_body"),
        ("none", "across", 90, "current_yaml"),
        ("none", "across", -90, "yaw_-90"),
        ("none", "across", 180, "yaw_180"),
        ("none", "body", 180, "body_yaw_180"),
        ("shepherd_y_z_x", "body", 0, "shep0"),
        ("shepherd_y_z_x", "body", 180, "shep180"),
        ("shepherd_y_negz_x", "body", 0, "shep_negz"),
        ("shepherd_y_negz_x", "body", 90, "shep_negz_90"),
        ("shepherd_y_negz_x", "body", 180, "shep_negz_180"),
    ]

    for clip_name, src_path in [("SWIM", src_swim), ("WALK", src_walk)]:
        print(f"\n======== {clip_name} {src_path.name} ========")
        # Reference: source motion FK on its own rest skel (intent)
        ref_m = get_inp_from_bvh(
            str(src_path),
            axis_transform="none",
            forward_mode="across",
            post_axis_yaw_deg=0,
        )
        ref_joints = fk_positions(ref_m["skel"][0], ref_m["quat"])
        gvy_ref = float(np.mean(ref_m["seq"][:, -8:-4][:, 1]))
        ref = summarize_clip("REF src@src_skel none/across/0", ref_joints, gvy_ref)

        rows = []
        for axis, fwd, yaw, note in candidates:
            m = get_inp_from_bvh(
                str(src_path),
                axis_transform=axis,
                forward_mode=fwd,
                post_axis_yaw_deg=yaw,
            )
            if m is None:
                print(f"  skip {note}: parse failed")
                continue
            joints, gvy = copyquat_on_target(m, rest)
            cur = summarize_clip(f"CQ {note} ({axis}/{fwd}/{yaw})", joints, gvy)
            sc = score_against_ref(ref, cur)
            rows.append((note, sc, cur))
            print(
                f"  -> score paw_fu_dot={sc['paw_fu_dot']:+.3f} "
                f"root_up_agree={sc['root_up_agree']:.0f} "
                f"lateral_ratio={sc['lateral_ratio']:.3f} "
                f"sag_ratio={sc['sag_ratio']:.2f} facing_z={sc['facing_z']:+.2f}"
            )

        # Rank: high paw_fu_dot, root_up_agree, sag_ratio; low lateral_ratio
        def key(item):
            _, sc, _ = item
            return (
                sc["paw_fu_dot"]
                + 0.5 * sc["root_up_agree"]
                + 0.3 * sc["sag_ratio"]
                - 0.4 * min(sc["lateral_ratio"], 3.0)
            )

        rows.sort(key=key, reverse=True)
        print(f"\nTop candidates for {clip_name}:")
        for note, sc, cur in rows[:5]:
            print(
                f"  {note:20s} paw_fu={sc['paw_fu_dot']:+.3f} up_agree={sc['root_up_agree']:.0f} "
                f"lat={sc['lateral_ratio']:.2f} sag={sc['sag_ratio']:.2f} "
                f"paw={cur['paw_b']}"
            )

    # Extra: yaw0 with mesh yaw_180 on FK joints (simulate direct.mesh_orientation_correction)
    print("\n======== Post mesh orientation on CQ(none/across/0) SWIM ========")
    m0 = get_inp_from_bvh(str(src_swim), axis_transform="none", forward_mode="across", post_axis_yaw_deg=0)
    joints0, gvy0 = copyquat_on_target(m0, rest)
    for mode, fn in [
        ("identity", lambda v: v),
        ("yaw_180", lambda v: v * np.array([-1, 1, -1])),
        ("mirror_x", lambda v: v * np.array([-1, 1, 1])),
        ("mirror_z", lambda v: v * np.array([1, 1, -1])),
        ("flip_y", lambda v: v * np.array([1, -1, 1])),
        ("yaw_180_flip_y", lambda v: v * np.array([-1, -1, -1])),
    ]:
        j = fn(joints0.copy())
        summarize_clip(f"mesh_orient {mode}", j, gvy0 * (-1 if "flip_y" in mode else 1))


if __name__ == "__main__":
    main()
