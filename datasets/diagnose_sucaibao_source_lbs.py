#!/usr/bin/env python3
"""Diagnose sucaibao / generic Source-lane LBS failures (orientation + 拉皮).

What this checks
---------------
1) Shape npz vs BVH rest offsets (export axis consistency)
2) Which world axis is "up" / "forward" before & after canonicalize
3) Joint-ball LBS vs BVH FK: joint translations can be fine while mesh AABB
   explodes — classic sign that this armature needs bone-oriented bind
   matrices (Blender), not R2ET's identity-rest joint-ball LBS
4) A/B facing configs (shepherd_y_z_x ± yaw ± canonicalize)

Run (R2ET conda, from repo root)::

  python datasets/diagnose_sucaibao_source_lbs.py
  python datasets/diagnose_sucaibao_source_lbs.py \\
    --bvh datasets/shepherd/cat_actions_sucaibao/train_char/JumpFw/JumpFw_IP.bvh \\
    --npz datasets/shepherd/cat_actions_sucaibao/train_shape/JumpFw.npz \\
    --device 0

Optional Blender ground-truth (compares depsgraph skinned mesh to joint-ball)::

  blender -b -P datasets/diagnose_sucaibao_source_lbs_blender.py -- \\
    --fbx datasets/shepherd/cat_actions_sucaibao/train_char/JumpFw/JumpFw_IP.fbx \\
    --out /tmp/sucaibao_blender_skin.npz
  python datasets/diagnose_sucaibao_source_lbs.py --blender_skin /tmp/sucaibao_blender_skin.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTSIDE = REPO_ROOT / "outside-code"
if str(OUTSIDE) not in sys.path:
    sys.path.insert(0, str(OUTSIDE))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bvh",
        type=Path,
        default=REPO_ROOT
        / "datasets/shepherd/cat_actions_sucaibao/train_char/JumpFw/JumpFw_IP.bvh",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=REPO_ROOT / "datasets/shepherd/cat_actions_sucaibao/train_shape/JumpFw.npz",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--blender_skin", type=Path, default=None, help="Optional npz from blender helper.")
    p.add_argument("--json_out", type=Path, default=None)
    return p.parse_args()


def offsets_to_global(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    out = np.zeros_like(offsets, dtype=np.float64)
    for i, parent in enumerate(parents):
        out[i] = offsets[i] if int(parent) < 0 else out[int(parent)] + offsets[i]
    return out


def aabb_span(verts: np.ndarray) -> np.ndarray:
    return verts.max(0) - verts.min(0)


def dominant_axis(span: np.ndarray) -> str:
    return "xyz"[int(np.argmax(span))]


def report_export_consistency(bvh: Path, npz: Path) -> dict:
    import BVH
    from datasets.skeleton_io import parse_bvh_hierarchy_names, remap_anim_to_names
    from datasets.smal33_motion_io import AXIS_TRANSFORMS

    raw = np.load(npz, allow_pickle=True)
    names = [str(x) for x in raw["joint_names"]]
    topo = np.asarray(raw["topology"], dtype=np.int64)
    skel = np.asarray(raw["skeleton"], dtype=np.float64)
    verts = np.asarray(raw["rest_vertices"], dtype=np.float64)

    anim, _, _ = BVH.load(str(bvh))
    anim, _ = remap_anim_to_names(anim, parse_bvh_hierarchy_names(bvh), names)
    off = np.asarray(anim.offsets, dtype=np.float64)
    M = AXIS_TRANSFORMS["shepherd_y_z_x"]
    off_t = (M @ off.T).T

    def dir_cos(a, b):
        dots = []
        for i in range(1, len(names)):
            na, nb = np.linalg.norm(a[i]), np.linalg.norm(b[i])
            if na > 1e-8 and nb > 1e-8:
                dots.append(float(np.dot(a[i], b[i]) / (na * nb)))
        return float(np.mean(dots)), float(np.min(dots))

    cos_raw = dir_cos(skel, off)
    cos_shep = dir_cos(skel, off_t)
    span = aabb_span(verts)
    g = offsets_to_global(skel, topo)
    hi, ri = names.index("head"), names.index("root_bone")
    sl, sr = names.index("shoulder_blade.L"), names.index("shoulder_blade.R")
    hl, hr = names.index("hip_b.L"), names.index("hip_b.R")
    head_vec = g[hi] - g[ri]
    across = g[sl] - g[sr]
    forward = 0.5 * (g[sl] + g[sr]) - 0.5 * (g[hl] + g[hr])
    forward[1] = 0.0

    out = {
        "n_joints": len(names),
        "bone_length_max_abs_diff_after_shepherd": float(
            np.max(np.abs(np.linalg.norm(skel, axis=1) - np.linalg.norm(off, axis=1)))
        ),
        "dir_cos_raw_npz_vs_bvh_mean_min": cos_raw,
        "dir_cos_shepherd_npz_vs_bvh_mean_min": cos_shep,
        "npz_vert_aabb_span": span.tolist(),
        "npz_vert_dominant_axis": dominant_axis(span),
        "npz_head_minus_root": head_vec.tolist(),
        "npz_shoulder_across": across.tolist(),
        "npz_forward_shoulder_minus_hip": forward.tolist(),
        "export_ok_with_shepherd_y_z_x": cos_shep[0] > 0.999,
    }
    return out, names, topo, raw


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(T,J,4) wxyz → (T,J,3,3)."""
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    return np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )


def skin_joint_ball(quat, rest, verts, weights, parents, device=None):
    """CPU joint-ball LBS matching R2ET (identity rest orientation).

    Avoids ``src.linear_blend_skin`` so diagnosis works without CUDA.
    """
    del device  # API compat with callers
    quat = np.asarray(quat, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    t_len, n_j = quat.shape[0], rest.shape[0]

    # rest global = [I | g], g accumulated from offsets
    g = np.zeros((n_j, 3), dtype=np.float64)
    for i, p in enumerate(parents):
        g[i] = rest[i] if int(p) < 0 else g[int(p)] + rest[i]

    R_loc = _quat_to_mat(quat)  # (T,J,3,3)
    # posed local transform: [R | offset]
    R_glob = np.zeros_like(R_loc)
    t_glob = np.zeros((t_len, n_j, 3), dtype=np.float64)
    for i, p in enumerate(parents):
        if int(p) < 0:
            R_glob[:, i] = R_loc[:, i]
            t_glob[:, i] = rest[i]
        else:
            R_glob[:, i] = R_glob[:, int(p)] @ R_loc[:, i]
            t_glob[:, i] = t_glob[:, int(p)] + np.einsum(
                "tij,j->ti", R_glob[:, int(p)], rest[i]
            )

    # bone_world = pose * inv(rest) ; rest=[I|g] → apply R*(v-g)+t
    # skinned = sum_j w_j * (R_j (v - g_j) + t_j)
    out = np.zeros((t_len, verts.shape[0], 3), dtype=np.float64)
    for j in range(n_j):
        wj = weights[:, j]
        if float(wj.max()) < 1e-12:
            continue
        delta = verts - g[j]
        posed = np.einsum("tij,vj->tvi", R_glob[:, j], delta) + t_glob[:, j][:, None, :]
        out += posed * wj[None, :, None]
    return out.astype(np.float32)


def report_lbs_vs_fk(bvh: Path, names, topo, raw, device) -> dict:
    import BVH
    import Animation
    from datasets.skeleton_io import parse_bvh_hierarchy_names, remap_anim_to_names
    from datasets.smal33_motion_io import apply_axis_transform_anim

    anim, _, _ = BVH.load(str(bvh))
    anim, _ = remap_anim_to_names(anim, parse_bvh_hierarchy_names(bvh), names)
    anim = apply_axis_transform_anim(anim, "shepherd_y_z_x")
    qs = anim.rotations.qs.astype(np.float32)
    mid = len(qs) // 2
    g = Animation.positions_global(anim)
    fk_travel = np.linalg.norm(g[mid] - g[0], axis=1)

    verts = np.asarray(raw["rest_vertices"], dtype=np.float32)
    skel = np.asarray(raw["skeleton"], dtype=np.float32)
    w = np.asarray(raw["skinning_weights"], dtype=np.float32)
    skinned = skin_joint_ball(qs, skel, verts, w, topo, device)
    mesh_travel = np.linalg.norm(skinned[mid] - skinned[0], axis=1)

    span0 = aabb_span(skinned[0])
    spanm = aabb_span(skinned[mid])
    idq = np.zeros((1, len(names), 4), np.float32)
    idq[..., 0] = 1.0
    id_err = np.linalg.norm(
        skin_joint_ball(idq, skel, verts, w, topo, device)[0] - verts, axis=1
    )

    # Dominant-part centroids vs FK for a few bones
    part = np.argmax(w, axis=1)
    bone_rows = []
    for jname in ("head", "hip_b.L", "foot_f.L", "leg_f.L", "spine_05"):
        j = names.index(jname)
        mask = part == j
        if int(mask.sum()) < 5:
            continue
        c0 = skinned[0, mask].mean(0)
        cm = skinned[mid, mask].mean(0)
        bone_rows.append(
            {
                "joint": jname,
                "n_verts": int(mask.sum()),
                "mesh_centroid_delta": (cm - c0).tolist(),
                "fk_joint_delta": (g[mid, j] - g[0, j]).tolist(),
                "mesh_travel": float(np.linalg.norm(cm - c0)),
                "fk_travel": float(np.linalg.norm(g[mid, j] - g[0, j])),
            }
        )

    return {
        "identity_lbs_vs_rest_mean_max": [float(id_err.mean()), float(id_err.max())],
        "fk_joint_travel_mean_max": [float(fk_travel.mean()), float(fk_travel.max())],
        "mesh_vert_travel_mean_max": [float(mesh_travel.mean()), float(mesh_travel.max())],
        "aabb_span_frame0": span0.tolist(),
        "aabb_span_mid": spanm.tolist(),
        "aabb_explosion_ratio_xz": float(
            (spanm[0] * spanm[2] + 1e-8) / (span0[0] * span0[2] + 1e-8)
        ),
        "bone_centroid_vs_fk": bone_rows,
        "diagnosis_joint_ball_mismatch": bool(
            float((spanm[0] * spanm[2] + 1e-8) / (span0[0] * span0[2] + 1e-8)) > 3.0
            or float(mesh_travel.mean()) > 2.0 * float(fk_travel.mean()) + 0.05
            or any(
                r["mesh_travel"] > max(0.05, 5.0 * r["fk_travel"] + 0.02)
                for r in bone_rows
            )
        ),
    }


def report_pipeline_configs(bvh: Path, npz: Path, names, topo, device) -> list[dict]:
    from datasets.bind_pose_canonicalize_smal33 import bind_process_forward
    from datasets.smal33_motion_io import get_inp_from_bvh, load_mesh_from_npz
    from datasets.smal33_motion_io import yaw_rotation_matrix_degrees

    rows = []
    combos = [
        ("current_cfg", "shepherd_y_z_x", 0.0, True, False),
        ("shepherd_yaw+90_mesh_match", "shepherd_y_z_x", 90.0, True, True),
        ("shepherd_yaw-90_mesh_match", "shepherd_y_z_x", -90.0, True, True),
        ("shepherd_no_canon", "shepherd_y_z_x", 0.0, False, False),
    ]
    for label, ax, yaw, can, yaw_mesh in combos:
        mesh = load_mesh_from_npz(str(npz), canonicalize_bind_pose=can, forward_mode="body")
        mot = get_inp_from_bvh(
            str(bvh),
            axis_transform=ax,
            forward_mode="body",
            keep_joint_names=names,
            post_axis_yaw_deg=yaw,
            canonicalize_bind_pose=can,
        )
        verts = mesh["vertices"]
        skel = mesh["skeleton"]
        if yaw_mesh and abs(yaw) > 1e-6:
            mesh0 = load_mesh_from_npz(str(npz), canonicalize_bind_pose=False)
            R = yaw_rotation_matrix_degrees(yaw)
            verts = np.einsum("ij,...j->...i", R, mesh0["vertices"]).astype(np.float32)
            skel = np.einsum("ij,...j->...i", R, mesh0["skeleton"]).astype(np.float32)
            if can:
                from datasets.bind_pose_canonicalize_smal33 import (
                    canonicalize_shape_npz_arrays,
                )

                payload = {
                    "skeleton": skel,
                    "rest_vertices": verts,
                    "rest_faces": mesh0["faces"],
                    "skinning_weights": mesh0["skin_weights"],
                    "joint_names": np.asarray(names),
                    "topology": topo,
                }
                payload, _ = canonicalize_shape_npz_arrays(payload, forward_mode="body")
                verts = payload["rest_vertices"]
                skel = payload["skeleton"]
            mesh = dict(mesh0)
            mesh["vertices"] = verts
            mesh["skeleton"] = skel

        skinned = skin_joint_ball(
            mot["quat"], skel, mesh["vertices"], mesh["skin_weights"], topo, device
        )
        mid = len(skinned) // 2
        span0 = aabb_span(skinned[0])
        spanm = aabb_span(skinned[mid])
        fwd = bind_process_forward(
            np.asarray(skel, dtype=np.float64),
            "body",
            joint_names=names,
            parents=topo,
        )
        rows.append(
            {
                "label": label,
                "span0": span0.tolist(),
                "span_mid": spanm.tolist(),
                "dominant0": dominant_axis(span0),
                "y_over_xz0": float(span0[1] / (0.5 * (span0[0] + span0[2]) + 1e-8)),
                "aabb_explosion_xz": float(
                    (spanm[0] * spanm[2] + 1e-8) / (span0[0] * span0[2] + 1e-8)
                ),
                "forward": fwd.tolist(),
                "motion_canonicalized": bool(mot.get("_bind_pose_canonicalized")),
            }
        )
    return rows


def compare_blender_skin(path: Path, joint_ball_mid_span: np.ndarray) -> dict:
    data = np.load(path)
    verts = np.asarray(data["vertices"], dtype=np.float32)
    span0 = aabb_span(verts[0])
    mid = len(verts) // 2
    spanm = aabb_span(verts[mid])
    return {
        "blender_frames": int(len(verts)),
        "blender_span0": span0.tolist(),
        "blender_span_mid": spanm.tolist(),
        "blender_explosion_xz": float(
            (spanm[0] * spanm[2] + 1e-8) / (span0[0] * span0[2] + 1e-8)
        ),
        "joint_ball_span_mid": joint_ball_mid_span.tolist(),
        "blender_much_more_stable": float(
            (spanm[0] * spanm[2]) / (span0[0] * span0[2] + 1e-8)
        )
        < 0.5
        * float(
            (joint_ball_mid_span[0] * joint_ball_mid_span[2])
            / (span0[0] * span0[2] + 1e-8)
        ),
    }


def main():
    args = parse_args()
    if not args.bvh.is_file():
        raise SystemExit(f"BVH not found: {args.bvh}")
    if not args.npz.is_file():
        raise SystemExit(f"npz not found: {args.npz}")

    device = None  # joint-ball diag is CPU NumPy
    print(f"device=cpu-numpy (ignore --device={args.device})")
    print(f"bvh={args.bvh}")
    print(f"npz={args.npz}")

    export, names, topo, raw = report_export_consistency(args.bvh, args.npz)
    print("\n== 1) npz export vs BVH ==")
    for k, v in export.items():
        print(f"  {k}: {v}")

    lbs = report_lbs_vs_fk(args.bvh, names, topo, raw, device)
    print("\n== 2) joint-ball LBS vs BVH FK ==")
    for k, v in lbs.items():
        if k == "bone_centroid_vs_fk":
            print("  bone_centroid_vs_fk:")
            for row in v:
                print(
                    f"    {row['joint']:10s} mesh_travel={row['mesh_travel']:.4f} "
                    f"fk_travel={row['fk_travel']:.4f}"
                )
        else:
            print(f"  {k}: {v}")

    configs = report_pipeline_configs(args.bvh, args.npz, names, topo, device)
    print("\n== 3) facing / canonicalize A/B ==")
    for row in configs:
        print(
            f"  {row['label']:28s} dom0={row['dominant0']} y/xz={row['y_over_xz0']:.3f} "
            f"explXZ={row['aabb_explosion_xz']:.2f} fwd={np.round(row['forward'], 3)}"
        )

    blender = None
    if args.blender_skin is not None:
        blender = compare_blender_skin(
            args.blender_skin, np.asarray(lbs["aabb_span_mid"], dtype=np.float64)
        )
        print("\n== 4) Blender depsgraph skin ==")
        for k, v in blender.items():
            print(f"  {k}: {v}")

    print("\n== verdict ==")
    if export["export_ok_with_shepherd_y_z_x"]:
        print("  - npz rest offsets match BVH after shepherd_y_z_x → export axis is OK.")
    else:
        print("  - npz/BVH axis mismatch → re-check extract_shape_sucaibao_cat flags.")
    if lbs["diagnosis_joint_ball_mismatch"]:
        print(
            "  - Mesh travel >> FK travel / AABB explodes while joints stay stable →"
            " R2ET joint-ball LBS is WRONG for this armature (needs Blender bone bind)."
        )
        print(
            "  - show_source should evaluate FBX in Blender (depsgraph), not Python LBS."
        )
    else:
        print("  - No strong joint-ball mismatch on this clip; check facing A/B next.")
    if configs[0]["dominant0"] != "y":
        print(
            f"  - Current cfg frame0 dominant axis is {configs[0]['dominant0']} "
            "(expected y for upright LBS Y-up) → orientation/axis issue."
        )

    payload = {
        "export": export,
        "lbs_vs_fk": lbs,
        "configs": configs,
        "blender": blender,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
