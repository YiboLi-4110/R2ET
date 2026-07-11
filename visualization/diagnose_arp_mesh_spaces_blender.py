#!/usr/bin/env python3
"""
Diagnose ARP mesh coordinate-space choices inside Blender.

For one case, this script runs ARP retarget once and samples evaluated mesh
vertices in four spaces:
  1) world_zup            : eval_obj.matrix_world @ vert.co
  2) mesh_local_zup       : vert.co (evaluated mesh local/object space)
  3) armature_local_zup   : target_arm.matrix_world^-1 * world
  4) root_bone_local_zup  : root_bone.matrix_world^-1 * world

Optional ARP operators such as clear_root_motion are skipped with a warning
when they fail in headless batch context (missing ARP trajectory bone).

Then (optionally) compares each space against a fourway_compare.npz reference
lane (copyquat / ours), reporting:
  - best orientation transform (identity / yaw_180 / mirror_x / mirror_z)
  - trajectory stats before and after lock-horizontal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_blender_common import (
    assign_arp_rigs,
    blender_argv,
    call_arp_operator,
    clean_scene,
    ensure_blender_ui_context,
    get_action_frame_range,
    import_bvh_armature,
    install_remap_preset,
    load_cfg,
    print_arp_diagnostics,
    reset_armature_object_transform,
    resolve_operator_sequence,
    try_enable_addons,
)
from compare_assets import enrich_case_assets
from arp_export_mesh_blender import import_target_fbx, mesh_faces_from_eval


def blender_argv() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ARP mesh spaces in Blender.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--case_id", type=str, required=True)
    parser.add_argument("--fourway_npz", type=str, required=True)
    parser.add_argument(
        "--reference_lane",
        type=str,
        default="copyquat",
        choices=["copyquat", "ours"],
    )
    parser.add_argument(
        "--lock_anchor",
        type=str,
        default="first",
        choices=["first", "median"],
    )
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--out_spaces_npz", type=str, default=None)
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
    )
    parser.add_argument("--operator_sequence", type=str, nargs="+", default=None)
    return parser.parse_args(argv)


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def align_frames(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if len(arr) == num_frames:
        return arr
    if len(arr) <= 1:
        return np.repeat(arr, num_frames, axis=0)
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(np.int64)
    return arr[idx]


def blender_z_up_to_lbs_y_up(vertices: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    return verts[..., [0, 2, 1]]


def bbox_center_per_frame(vertices: np.ndarray) -> np.ndarray:
    mn = vertices.min(axis=1)
    mx = vertices.max(axis=1)
    return (mn + mx) * 0.5


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    centers = bbox_center_per_frame(vertices)
    return vertices - centers[:, None, :]


def apply_orientation_transform(vertices: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float64).copy()
    if name == "identity":
        return out
    if name == "yaw_180":
        out[..., 0] *= -1.0
        out[..., 2] *= -1.0
        return out
    if name == "mirror_x":
        out[..., 0] *= -1.0
        return out
    if name == "mirror_z":
        out[..., 2] *= -1.0
        return out
    raise ValueError(f"Unknown orientation candidate: {name}")


def candidate_shape_relation(subject: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if subject.shape != reference.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch: subject={tuple(subject.shape)} ref={tuple(reference.shape)}",
        }
    s = center_vertices(subject).astype(np.float64)
    r = center_vertices(reference).astype(np.float64)
    candidates = ("identity", "yaw_180", "mirror_x", "mirror_z")
    scores = {}
    for name in candidates:
        t = apply_orientation_transform(s, name)
        rmse = np.sqrt(np.mean((t - r) ** 2, axis=(1, 2)))
        scores[name] = {
            "rmse_mean": float(rmse.mean()),
            "rmse_p95": float(np.percentile(rmse, 95.0)),
        }
    best = min(scores, key=lambda k: scores[k]["rmse_mean"])
    return {"comparable": True, "best_candidate": best, "scores": scores}


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) <= 1:
        return verts, np.zeros((len(verts), 3), dtype=np.float32)
    centers = bbox_center_per_frame(verts)
    if anchor == "median":
        ref = np.median(centers, axis=0, keepdims=True).astype(np.float32)
    else:
        ref = centers[:1].astype(np.float32)
    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    return verts - drift[:, None, :], drift


def trajectory_metrics(vertices: np.ndarray, ref_vertices: np.ndarray) -> dict[str, Any]:
    c = bbox_center_per_frame(vertices).astype(np.float64)
    r = bbox_center_per_frame(ref_vertices).astype(np.float64)
    c2 = c[:, [0, 2]]
    r2 = r[:, [0, 2]]
    disp = c2[-1] - c2[0]
    span = c.max(axis=0) - c.min(axis=0)
    step = c[1:] - c[:-1]
    if np.linalg.norm(disp) < 1e-8:
        fwd = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd = disp / np.linalg.norm(disp)
    right = np.array([fwd[1], -fwd[0]], dtype=np.float64)
    lateral = (c2 - c2[0]) @ right
    ref_fwd = (r2 - r2[0]) @ fwd
    sub_fwd = (c2 - c2[0]) @ fwd
    corr = 0.0
    if np.std(ref_fwd) > 1e-8 and np.std(sub_fwd) > 1e-8:
        corr = float(np.corrcoef(ref_fwd, sub_fwd)[0, 1])
    return {
        "center_span_xyz": span.tolist(),
        "center_span_xz": float(np.linalg.norm(span[[0, 2]])),
        "step_speed_xz_mean": float(np.linalg.norm(step[:, [0, 2]], axis=-1).mean()) if len(step) else 0.0,
        "lateral_std": float(np.std(lateral)),
        "lateral_sign_changes": int(np.sum((np.sign(lateral[1:]) * np.sign(lateral[:-1])) < 0)) if len(lateral) > 1 else 0,
        "forward_corr_with_reference": corr,
    }


ROOT_BONE_CANDIDATES = (
    "smal:Root",
    "Root",
    "root",
    "c_root",
    "c_traj",
    "Hips",
    "hips",
)


def find_root_pose_bone(armature_obj):
    arm_data = armature_obj.data
    pose_bones = armature_obj.pose.bones
    for name in ROOT_BONE_CANDIDATES:
        if name in pose_bones:
            return pose_bones[name]
    if len(pose_bones) > 0:
        # Fallback: first bone in armature data order (SMAL33 root is index 0).
        return pose_bones[arm_data.bones[0].name]
    raise RuntimeError(f"No pose bones found on armature: {armature_obj.name}")


def sample_spaces(mesh_obj, armature_obj, frame_start: int, frame_end: int) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_inv = armature_obj.matrix_world.inverted()
    root_bone = find_root_pose_bone(armature_obj)
    root_bone_name = root_bone.name

    verts_world = []
    verts_mesh_local = []
    verts_arm_local = []
    verts_root_bone_local = []
    faces = None

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        eval_arm = armature_obj.evaluated_get(depsgraph)
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_eval = eval_obj.to_mesh()
        if mesh_eval is None:
            raise RuntimeError(f"Failed mesh evaluation at frame={frame}")
        try:
            if faces is None:
                faces = mesh_faces_from_eval(mesh_eval)

            root_pose = eval_arm.pose.bones[root_bone_name]
            root_world = eval_arm.matrix_world @ root_pose.matrix
            root_world_inv = root_world.inverted()

            fw = []
            fl = []
            fa = []
            fr = []
            for vert in mesh_eval.vertices:
                local = vert.co
                world = eval_obj.matrix_world @ local
                arm_local = arm_inv @ world
                root_local = root_world_inv @ world
                fw.append(world[:])
                fl.append(local[:])
                fa.append(arm_local[:])
                fr.append(root_local[:])
            verts_world.append(np.asarray(fw, dtype=np.float32))
            verts_mesh_local.append(np.asarray(fl, dtype=np.float32))
            verts_arm_local.append(np.asarray(fa, dtype=np.float32))
            verts_root_bone_local.append(np.asarray(fr, dtype=np.float32))
        finally:
            eval_obj.to_mesh_clear()

    spaces = {
        "world_zup": np.stack(verts_world, axis=0),
        "mesh_local_zup": np.stack(verts_mesh_local, axis=0),
        "armature_local_zup": np.stack(verts_arm_local, axis=0),
        "root_bone_local_zup": np.stack(verts_root_bone_local, axis=0),
    }
    return spaces, faces.astype(np.int32), root_bone_name


def pick_case(cfg: dict[str, Any], case_id: str) -> dict[str, Any]:
    assets_cfg = cfg.get("assets", {})
    cases = cfg.get("cases", [])
    for case in cases:
        if case.get("case_id") == case_id:
            return enrich_case_assets(case, assets_cfg)
    raise KeyError(f"case_id not found in config: {case_id}")


def run_case_and_collect_spaces(
    case_cfg: dict[str, Any],
    cfg: dict[str, Any],
    op_sequence: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, str, list[str]]:
    arp_cfg = cfg.get("arp", {})
    clean_scene()

    source_arm = import_bvh_armature(Path(case_cfg["inp_bvh_path"]))
    target_arm, target_mesh = import_target_fbx(Path(case_cfg["tgt_fbx_path"]))
    reset_armature_object_transform(source_arm)
    reset_armature_object_transform(target_arm)
    assign_arp_rigs(source_arm, target_arm)

    frame_start, frame_end = get_action_frame_range(source_arm)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    print(f"[diag-arp-space] frame range: {frame_start}-{frame_end}")

    skipped_ops: list[str] = []
    for op_name in op_sequence:
        result = call_arp_operator(
            op_name,
            target_arm,
            arp_cfg,
            frame_range=(frame_start, frame_end),
        )
        print(f"[diag-arp-space] bpy.ops.arp.{op_name} -> {result}")
        if isinstance(result, set) and "CANCELLED" in result:
            skipped_ops.append(op_name)

    spaces, faces, root_bone_name = sample_spaces(target_mesh, target_arm, frame_start, frame_end)
    return spaces, faces, root_bone_name, skipped_ops


def main() -> None:
    ensure_blender_ui_context()
    args = parse_args(blender_argv())
    cfg = load_cfg(args.config)
    case_cfg = pick_case(cfg, args.case_id)
    op_sequence = resolve_operator_sequence(cfg, args.operator_sequence)

    enabled, failed = try_enable_addons(args.arp_addon_modules)
    print_arp_diagnostics(args.arp_addon_modules)
    if enabled:
        print("[diag-arp-space] addon enable ok:", enabled)
    if failed:
        print("[diag-arp-space] addon enable failed:", json.dumps(failed, ensure_ascii=False, indent=2))
    if "import_config_preset" in op_sequence:
        install_remap_preset(cfg.get("arp", {}))

    spaces, faces, root_bone_name, skipped_ops = run_case_and_collect_spaces(case_cfg, cfg, op_sequence)

    fourway = np.load(args.fourway_npz)
    ref_key = "copyquat_vertices" if args.reference_lane == "copyquat" else "ours_vertices"
    if ref_key not in fourway.files:
        raise KeyError(f"reference lane key missing in fourway npz: {ref_key}")
    ref = np.asarray(fourway[ref_key], dtype=np.float32)
    num_frames = int(ref.shape[0])

    converted = {
        name: align_frames(blender_z_up_to_lbs_y_up(verts), num_frames)
        for name, verts in spaces.items()
    }

    report: dict[str, Any] = {
        "inputs": {
            "config": str(args.config),
            "case_id": args.case_id,
            "fourway_npz": str(args.fourway_npz),
            "reference_lane": args.reference_lane,
            "lock_anchor": args.lock_anchor,
            "operator_sequence": op_sequence,
            "skipped_operators": skipped_ops,
            "root_bone_name": root_bone_name,
        },
        "space_shapes": {k: list(v.shape) for k, v in spaces.items()},
        "reference_shape": list(ref.shape),
        "space_comparison_to_reference": {},
        "space_trajectory": {
            "reference": trajectory_metrics(ref, ref),
        },
    }

    for name, verts in converted.items():
        locked, drift = remove_horizontal_drift(verts, vertical_axis=1, anchor=args.lock_anchor)
        report["space_comparison_to_reference"][name] = {
            "raw": candidate_shape_relation(verts, ref),
            "locked_horizontal": candidate_shape_relation(locked, ref),
        }
        report["space_trajectory"][name] = {
            "raw": trajectory_metrics(verts, ref),
            "locked_horizontal": trajectory_metrics(locked, ref),
            "lock_drift_norm": {
                "max": float(np.linalg.norm(drift, axis=-1).max()) if len(drift) else 0.0,
                "p95": float(np.percentile(np.linalg.norm(drift, axis=-1), 95.0)) if len(drift) else 0.0,
            },
        }

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diag-arp-space] wrote json: {out_json_path}")

    if args.out_spaces_npz:
        out_npz = Path(args.out_spaces_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            faces=faces.astype(np.int32),
            root_bone_name=np.asarray(root_bone_name),
            world_zup=spaces["world_zup"].astype(np.float32),
            mesh_local_zup=spaces["mesh_local_zup"].astype(np.float32),
            armature_local_zup=spaces["armature_local_zup"].astype(np.float32),
            root_bone_local_zup=spaces["root_bone_local_zup"].astype(np.float32),
        )
        print(f"[diag-arp-space] wrote spaces npz: {out_npz}")


if __name__ == "__main__":
    main()
