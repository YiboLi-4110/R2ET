#!/usr/bin/env python3
"""
ARP retarget each source clip onto one target FBX character, then export a
complete skinned FBX (armature + meshes + materials + animation).

Run (usually via batch_arp_export_fbx_smal33.py):
  blender --background --python visualization/arp_export_fbx_blender.py -- \\
    --config /path/to/_arp_fbx_batch_config.yaml \\
    --arp_addon_modules auto_rig_pro-master
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

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
from arp_export_mesh_blender import import_target_fbx
from compare_assets import enrich_case_assets


def parse_args(argv):
    parser = argparse.ArgumentParser(description="ARP retarget and export full FBX.")
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


def object_subtree(root_obj):
    """Root object plus all descendants (by parent links)."""
    found = []
    stack = [root_obj]
    seen = set()
    while stack:
        obj = stack.pop()
        if obj.name in seen:
            continue
        seen.add(obj.name)
        found.append(obj)
        stack.extend(list(obj.children))
    return found


def meshes_deformed_by_armature(armature_obj):
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        bound = False
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and mod.object == armature_obj:
                bound = True
                break
        if not bound and obj.parent == armature_obj:
            bound = True
        if bound:
            meshes.append(obj)
    return meshes


def delete_objects(objects):
    names = [obj.name for obj in objects]
    # Delete non-armatures first, then armatures.
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None and obj.type != "ARMATURE":
            bpy.data.objects.remove(obj, do_unlink=True)
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def collect_export_objects(target_arm):
    """Armature + all skinned/parented meshes (full character, not arm-only)."""
    objs = [target_arm]
    for mesh_obj in meshes_deformed_by_armature(target_arm):
        if mesh_obj not in objs:
            objs.append(mesh_obj)
        # Include empty parents / helper objects under the mesh if any.
        for child in object_subtree(mesh_obj):
            if child not in objs and child.type in {"MESH", "EMPTY", "ARMATURE"}:
                objs.append(child)
    return objs


def select_only(objects):
    bpy.ops.object.select_all(action="DESELECT")
    active = None
    for obj in objects:
        if obj.name not in bpy.data.objects:
            continue
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.select_set(True)
        active = obj
    if active is None:
        raise RuntimeError("No objects available for FBX export selection.")
    # Prefer armature as active for animation export.
    arms = [obj for obj in objects if obj.type == "ARMATURE" and obj.name in bpy.data.objects]
    bpy.context.view_layer.objects.active = arms[0] if arms else active


def export_skinned_fbx(
    out_path: Path,
    objects,
    *,
    frame_start: int,
    frame_end: int,
    embed_textures: bool = True,
    add_leaf_bones: bool = False,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_start = int(frame_start)
    scene.frame_end = int(frame_end)
    scene.frame_current = int(frame_start)
    select_only(objects)

    # Keep Armature modifiers (do not apply) so the FBX stays skinned.
    # bake_anim writes the retargeted pose animation onto the armature.
    kwargs = dict(
        filepath=str(out_path),
        check_existing=False,
        use_selection=True,
        use_active_collection=False,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        use_mesh_modifiers=False,
        use_mesh_modifiers_render=False,
        add_leaf_bones=bool(add_leaf_bones),
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="COPY" if embed_textures else "AUTO",
        embed_textures=bool(embed_textures),
        axis_forward=str(axis_forward),
        axis_up=str(axis_up),
    )
    # Blender 4.x FBX exporter accepts these; ignore unknown keys if an older
    # operator signature is present (defensive for mixed installs).
    try:
        bpy.ops.export_scene.fbx(**kwargs)
    except TypeError as exc:
        print(f"[arp-fbx][warn] full export kwargs failed ({exc}); retrying minimal set")
        minimal = {
            "filepath": str(out_path),
            "check_existing": False,
            "use_selection": True,
            "object_types": {"ARMATURE", "MESH"},
            "use_mesh_modifiers": False,
            "add_leaf_bones": bool(add_leaf_bones),
            "bake_anim": True,
            "bake_anim_use_all_actions": False,
            "path_mode": "COPY" if embed_textures else "AUTO",
            "embed_textures": bool(embed_textures),
        }
        bpy.ops.export_scene.fbx(**minimal)


def run_case(case_cfg, cfg, op_sequence):
    case_id = case_cfg["case_id"]
    arp_cfg = cfg.get("arp", {})
    fbx_cfg = cfg.get("fbx_export", {}) or {}
    clean_scene()

    before_source = {obj.name for obj in bpy.data.objects}
    use_source_fbx = bool(arp_cfg.get("use_source_fbx", True))
    source_fbx = case_cfg.get("inp_fbx_path")
    if use_source_fbx and source_fbx and Path(source_fbx).exists():
        source_arm = import_source_fbx_with_anim(Path(source_fbx))
        print(f"[arp-fbx][{case_id}] source from FBX: {source_fbx}")
    else:
        source_arm = import_bvh_armature(Path(case_cfg["inp_bvh_path"]))
        print(f"[arp-fbx][{case_id}] source from BVH: {case_cfg['inp_bvh_path']}")
    source_objects = [
        obj for obj in bpy.data.objects if obj.name not in before_source
    ]

    before_target = {obj.name for obj in bpy.data.objects}
    target_arm, _target_mesh = import_target_fbx(Path(case_cfg["tgt_fbx_path"]))
    target_import_names = {
        obj.name for obj in bpy.data.objects if obj.name not in before_target
    }

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
    print(f"[arp-fbx][{case_id}] frame range: {frame_start}-{frame_end}")
    print(
        f"[arp-fbx][{case_id}] rigs: source={source_arm.name} target={target_arm.name}"
    )

    for op_name in op_sequence:
        result = call_arp_operator(
            op_name,
            target_arm,
            arp_cfg,
            frame_range=(frame_start, frame_end),
        )
        print(f"[arp-fbx][{case_id}] bpy.ops.arp.{op_name} -> {result}")

    # Drop source character so the FBX only contains the retargeted target.
    delete_objects(source_objects)
    # Keep only objects from the target import that are still relevant.
    export_objects = collect_export_objects(target_arm)
    # Also keep any leftover target-import empties/meshes that were not bound
    # (rare), as long as they came from the target FBX.
    for name in target_import_names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj in export_objects:
            continue
        if obj.type in {"MESH", "EMPTY"}:
            export_objects.append(obj)

    mesh_count = sum(1 for obj in export_objects if obj.type == "MESH")
    arm_count = sum(1 for obj in export_objects if obj.type == "ARMATURE")
    if mesh_count < 1 or arm_count < 1:
        raise RuntimeError(
            f"[{case_id}] FBX export selection incomplete: "
            f"armatures={arm_count}, meshes={mesh_count}. "
            "Need both skinned mesh and armature."
        )

    out_path = Path(case_cfg["out_fbx_path"])
    export_skinned_fbx(
        out_path,
        export_objects,
        frame_start=frame_start,
        frame_end=frame_end,
        embed_textures=bool(fbx_cfg.get("embed_textures", True)),
        add_leaf_bones=bool(fbx_cfg.get("add_leaf_bones", False)),
        axis_forward=str(fbx_cfg.get("axis_forward", "-Z")),
        axis_up=str(fbx_cfg.get("axis_up", "Y")),
    )
    print(
        f"[arp-fbx][{case_id}] exported FBX: {out_path} "
        f"(armatures={arm_count}, meshes={mesh_count}, frames={frame_start}-{frame_end})"
    )
    return out_path


def main():
    ensure_blender_ui_context()
    args = parse_args(blender_argv())
    cfg = load_cfg(args.config)
    arp_cfg = cfg.get("arp", {})
    op_sequence = resolve_operator_sequence(cfg, args.operator_sequence)

    enabled, failed = try_enable_addons(args.arp_addon_modules)
    print_arp_diagnostics(args.arp_addon_modules)
    if enabled:
        print("[arp-fbx] addon enable ok:", enabled)
    if failed:
        print("[arp-fbx] addon enable failed:", json.dumps(failed, ensure_ascii=False, indent=2))
    if "import_config_preset" in op_sequence:
        install_remap_preset(arp_cfg)

    case_ids = set(args.case_ids) if args.case_ids else None
    cases = cfg.get("cases", [])
    if case_ids is not None:
        cases = [case for case in cases if case.get("case_id") in case_ids]
    if not cases:
        raise SystemExit("No cases selected for ARP FBX export.")

    assets_cfg = cfg.get("assets", {})
    resolved_cases = [enrich_case_assets(case, assets_cfg) for case in cases]
    if args.check_only:
        print("[arp-fbx] operator sequence:", " -> ".join(op_sequence))
        print("[arp-fbx] remap_preset_name:", arp_cfg.get("remap_preset_name"))
        return

    for case in resolved_cases:
        if not case.get("out_fbx_path"):
            raise SystemExit(f"Case '{case.get('case_id')}' missing out_fbx_path.")
        run_case(case, cfg, op_sequence)


if __name__ == "__main__":
    main()
