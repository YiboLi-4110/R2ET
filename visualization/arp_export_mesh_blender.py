#!/usr/bin/env python3
"""
Run ARP retargeting on the textured target FBX and export evaluated mesh vertices.

This follows the stable bad_case path: ARP is treated as an external baseline and
we cache its final deformed mesh sequence, instead of reinterpreting ARP BVH
local rotations with our Python LBS skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

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
    import_source_fbx_with_anim,
    install_remap_preset,
    load_cfg,
    pick_armature_by_hint,
    print_arp_diagnostics,
    reset_armature_object_transform,
    resolve_operator_sequence,
    try_enable_addons,
)
from compare_assets import enrich_case_assets
from fbx_bake_cache import save_fbx_bake_cache


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Export ARP evaluated mesh cache.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
    )
    parser.add_argument("--operator_sequence", type=str, nargs="+", default=None)
    parser.add_argument("--check_only", action="store_true", default=False)
    return parser.parse_args(argv)


def imported_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def import_target_fbx(path: Path):
    before = set(obj.name for obj in bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    imported = imported_objects(before)
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not armatures:
        raise RuntimeError(f"No target armature imported from FBX: {path}")
    if not meshes:
        raise RuntimeError(f"No target mesh imported from FBX: {path}")
    target_arm = armatures[-1]
    target_mesh = max(meshes, key=lambda obj: len(obj.data.vertices))
    return target_arm, target_mesh


def mesh_faces_from_eval(mesh_eval):
    mesh_eval.calc_loop_triangles()
    return np.asarray([tri.vertices[:] for tri in mesh_eval.loop_triangles], dtype=np.int32)


ROOT_BONE_CANDIDATES = (
    "smal:Root",
    "Root",
    "root",
    "root_bone",  # sucaibao / game-pack cats
    "c_root",
    "c_traj",
    "Hips",
    "hips",
)


def find_root_pose_bone(armature_obj):
    pose_bones = armature_obj.pose.bones
    for name in ROOT_BONE_CANDIDATES:
        if name in pose_bones:
            return pose_bones[name]
    if len(pose_bones) > 0:
        return pose_bones[armature_obj.data.bones[0].name]
    raise RuntimeError(f"No pose bones found on armature: {armature_obj.name}")


def sample_evaluated_mesh(mesh_obj, armature_obj, frame_start: int, frame_end: int, export_space: str):
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_inv = armature_obj.matrix_world.inverted()
    root_bone_name = find_root_pose_bone(armature_obj).name
    vertices = []
    faces = None
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        eval_arm = armature_obj.evaluated_get(depsgraph)
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_eval = eval_obj.to_mesh()
        if mesh_eval is None:
            raise RuntimeError(f"Failed to evaluate mesh at frame {frame}: {mesh_obj.name}")
        try:
            mw = eval_obj.matrix_world
            root_pose = eval_arm.pose.bones[root_bone_name]
            root_world = eval_arm.matrix_world @ root_pose.matrix
            root_world_inv = root_world.inverted()
            # Root bone world translation (Blender Z-up: horizontal = X/Y, vertical = Z).
            root_t = root_world.to_translation()
            verts_out = np.empty((len(mesh_eval.vertices), 3), dtype=np.float32)
            for i, vert in enumerate(mesh_eval.vertices):
                world_co = mw @ vert.co
                if export_space == "blender_world_z_up":
                    co = world_co
                elif export_space == "mesh_local_zup":
                    co = vert.co
                elif export_space == "armature_local_zup":
                    co = arm_inv @ world_co
                elif export_space == "world_root_h_locked_zup":
                    # Cancel ONLY the root bone's horizontal translation. Keep the
                    # root's rotation and vertical translation so the jump arc and
                    # torso pitch survive; the animal just stops sliding around.
                    co = world_co.copy()
                    co.x -= root_t.x
                    co.y -= root_t.y
                else:  # root_bone_local_zup
                    co = root_world_inv @ world_co
                verts_out[i, 0] = co.x
                verts_out[i, 1] = co.y
                verts_out[i, 2] = co.z
            vertices.append(verts_out)
            if faces is None:
                faces = mesh_faces_from_eval(mesh_eval)
        finally:
            eval_obj.to_mesh_clear()
    return np.stack(vertices, axis=0), faces, root_bone_name


def blender_matrix_to_np(mat) -> np.ndarray:
    arr = np.eye(4, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            arr[i, j] = float(mat[i][j])
    return arr


def sample_armature_bake_cache(armature_obj, frame_start: int, frame_end: int):
    """Rest + posed bone matrices in armature space (same FBX the packer re-imports)."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bone_names = [bone.name for bone in armature_obj.data.bones]
    rest = np.stack(
        [
            blender_matrix_to_np(armature_obj.data.bones[name].matrix_local)
            for name in bone_names
        ],
        axis=0,
    )
    frames = []
    for frame in range(int(frame_start), int(frame_end) + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        eval_arm = armature_obj.evaluated_get(depsgraph)
        mats = []
        for name in bone_names:
            pose_bone = eval_arm.pose.bones.get(name)
            if pose_bone is None:
                mats.append(np.eye(4, dtype=np.float64))
            else:
                mats.append(blender_matrix_to_np(pose_bone.matrix))
        frames.append(np.stack(mats, axis=0))
    ours = np.stack(frames, axis=0)
    fps = max(int(scene.render.fps), 1)
    return rest, ours, bone_names, 1.0 / float(fps)


def export_target_bvh(target_arm, out_path: Path, frame_start: int, frame_end: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.export_anim.bvh(
        filepath=str(out_path),
        check_existing=False,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=False,
    )


def run_case(case_cfg, cfg, op_sequence):
    case_id = case_cfg["case_id"]
    arp_cfg = cfg.get("arp", {})
    clean_scene()

    use_source_fbx = bool(arp_cfg.get("use_source_fbx", True))
    source_fbx = case_cfg.get("inp_fbx_path")
    if use_source_fbx and source_fbx and Path(source_fbx).exists():
        source_arm = import_source_fbx_with_anim(Path(source_fbx))
        print(f"[arp-mesh][{case_id}] source from FBX: {source_fbx}")
    else:
        source_arm = import_bvh_armature(Path(case_cfg["inp_bvh_path"]))
        print(f"[arp-mesh][{case_id}] source from BVH: {case_cfg['inp_bvh_path']}")
    target_arm, target_mesh = import_target_fbx(Path(case_cfg["tgt_fbx_path"]))

    source_hint = arp_cfg.get("source_armature_hint")
    target_hint = arp_cfg.get("target_armature_hint")
    if source_hint:
        source_arm = pick_armature_by_hint(source_hint, exclude_names={target_arm.name})
    if target_hint:
        target_arm = pick_armature_by_hint(target_hint, exclude_names={source_arm.name})

    reset_armature_object_transform(source_arm)
    reset_armature_object_transform(target_arm)
    assign_arp_rigs(source_arm, target_arm)

    frame_start, frame_end = get_action_frame_range(source_arm)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    print(f"[arp-mesh][{case_id}] frame range: {frame_start}-{frame_end}")

    for op_name in op_sequence:
        result = call_arp_operator(
            op_name,
            target_arm,
            arp_cfg,
            frame_range=(frame_start, frame_end),
        )
        print(f"[arp-mesh][{case_id}] bpy.ops.arp.{op_name} -> {result}")

    mesh_dir = Path(arp_cfg.get("mesh_output_dir", "./visualization/videos/compare/arp_mesh_outputs"))
    mesh_dir.mkdir(parents=True, exist_ok=True)
    bake_dir = Path(arp_cfg.get("bake_output_dir", mesh_dir))
    bake_dir.mkdir(parents=True, exist_ok=True)
    bvh_dir = Path(arp_cfg.get("output_dir", "./visualization/videos/compare/arp_outputs"))
    bvh_dir.mkdir(parents=True, exist_ok=True)

    export_mesh = bool(arp_cfg.get("export_mesh", True))
    export_bvh = bool(arp_cfg.get("export_bvh", True))
    export_bake = bool(arp_cfg.get("export_bake", False))

    mesh_path = Path(case_cfg["mesh_path"]) if case_cfg.get("mesh_path") else mesh_dir / f"{case_id}_arp_mesh.npz"
    bake_path = (
        Path(case_cfg["bake_path"])
        if case_cfg.get("bake_path")
        else bake_dir / f"{case_id}_ours_fbx_bake.npz"
    )
    suffix = arp_cfg.get("output_suffix", "_arp_retarget.bvh")
    bvh_path = (
        Path(case_cfg["arp_bvh_path"])
        if case_cfg.get("arp_bvh_path")
        else bvh_dir / f"{case_id}{suffix}"
    )

    if export_mesh:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        export_space = str(arp_cfg.get("mesh_export_space", "world_root_h_locked_zup")).lower()
        valid_spaces = {
            "blender_world_z_up",
            "mesh_local_zup",
            "armature_local_zup",
            "root_bone_local_zup",
            "world_root_h_locked_zup",
        }
        if export_space not in valid_spaces:
            print(
                f"[arp-mesh][warn] unknown mesh_export_space='{export_space}', "
                "fallback to world_root_h_locked_zup"
            )
            export_space = "world_root_h_locked_zup"
        verts, faces, root_bone_name = sample_evaluated_mesh(
            target_mesh,
            target_arm,
            frame_start,
            frame_end,
            export_space,
        )
        np.savez_compressed(
            mesh_path,
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            frame_start=np.int32(frame_start),
            frame_end=np.int32(frame_end),
            space=export_space,
            root_bone_name=np.asarray(root_bone_name),
        )
        print(
            f"[arp-mesh][{case_id}] mesh cache: {mesh_path} "
            f"(space={export_space}, root_bone={root_bone_name})"
        )

    if export_bake:
        bake_path.parent.mkdir(parents=True, exist_ok=True)
        rest_g, ours_g, joint_names, frametime = sample_armature_bake_cache(
            target_arm, frame_start, frame_end
        )
        save_fbx_bake_cache(
            bake_path,
            {
                "rest_globals": rest_g.astype(np.float64),
                "ours_globals": ours_g.astype(np.float64),
                "joint_names": np.asarray(joint_names, dtype=object),
                "frametime": np.float64(frametime),
                "space": np.asarray("fbx_armature"),
            },
        )
        print(
            f"[arp-mesh][{case_id}] bake cache: {bake_path} "
            f"(T={ours_g.shape[0]} J={ours_g.shape[1]} space=fbx_armature)"
        )

    if export_bvh:
        export_target_bvh(target_arm, bvh_path, frame_start, frame_end)
        print(f"[arp-mesh][{case_id}] bvh export: {bvh_path}")
    return mesh_path if export_mesh else bake_path


def main():
    ensure_blender_ui_context()
    args = parse_args(blender_argv())
    cfg = load_cfg(args.config)
    arp_cfg = cfg.get("arp", {})
    op_sequence = resolve_operator_sequence(cfg, args.operator_sequence)

    enabled, failed = try_enable_addons(args.arp_addon_modules)
    print_arp_diagnostics(args.arp_addon_modules)
    if enabled:
        print("[arp-mesh] addon enable ok:", enabled)
    if failed:
        print("[arp-mesh] addon enable failed:", json.dumps(failed, ensure_ascii=False, indent=2))
    if "import_config_preset" in op_sequence:
        install_remap_preset(arp_cfg)

    case_ids = set(args.case_ids) if args.case_ids else None
    cases = cfg.get("cases", [])
    if case_ids is not None:
        cases = [case for case in cases if case.get("case_id") in case_ids]
    if not cases:
        raise SystemExit("No cases selected for ARP mesh export.")

    assets_cfg = cfg.get("assets", {})
    resolved_cases = [enrich_case_assets(case, assets_cfg) for case in cases]
    if args.check_only:
        print("[arp-mesh] operator sequence:", " -> ".join(op_sequence))
        return

    for case in resolved_cases:
        run_case(case, cfg, op_sequence)


if __name__ == "__main__":
    main()
