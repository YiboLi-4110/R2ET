#!/usr/bin/env python3
"""
Pack R2ET results into one .blend per dog_id.

pack_mode (from manifest.blend.pack_mode):
  armature (default): keep target FBX armature + mesh; bake FBX-native Ours BVH
                      onto FBX bones (editable skeleton, no BVH re-import)
  lbs:      LBS vertex shape-keys (no armature; matches fourway video)
  both:     armature collections + optional LBS overlay collections

Blender 4.5.3:
  blender --background --python visualization/pack_r2et_dog_blend_blender.py -- \\
    --manifest visualization/blend/_work/博美_3/dog_actions_manifest.json \\
    --blend_output_dir visualization/blend
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bake_bvh_onto_fbx_armature import (  # noqa: E402
    apply_root_object_transform,
    bake_cache_onto_fbx_armature,
    import_fbx_armature_and_meshes,
    move_objects_to_collection,
)


AXIS_PRESETS = {
    "identity": np.eye(3, dtype=np.float32),
    "y_up_to_z_up": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
    "y_up_to_z_up_neg_y": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
}


def blender_argv():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Pack R2ET dog actions into Blender .blend files."
    )
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--manifest_index", type=str, default=None)
    parser.add_argument("--blend_output_dir", type=str, required=True)
    parser.add_argument("--dog_ids", type=str, nargs="+", default=None)
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def set_scene_animation_range(num_frames: int, fps: int):
    scene = bpy.context.scene
    scene.render.fps = max(int(fps), 1)
    scene.frame_start = 1
    scene.frame_end = max(int(num_frames), 1)
    scene.frame_current = 1


def save_blend(path: Path, *, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(path),
        check_existing=False,
        compress=bool(compress),
        relative_remap=True,
        copy=False,
    )


def layout_offset(index: int, layout: str, spacing: float, grid_cols: int):
    layout = (layout or "grid").lower()
    spacing = float(spacing)
    if layout in ("origin", "none", "stack"):
        return 0.0, 0.0, 0.0
    if layout in ("line_x", "line"):
        return index * spacing, 0.0, 0.0
    if layout in ("line_y",):
        return 0.0, index * spacing, 0.0
    cols = max(int(grid_cols), 1)
    row = index // cols
    col = index % cols
    return col * spacing, row * spacing, 0.0


def layout_offset_np(index: int, layout: str, spacing: float, grid_cols: int):
    x, y, z = layout_offset(index, layout, spacing, grid_cols)
    return np.array([x, y, z], dtype=np.float32)


def set_collection_hide(collection_name: str, hide: bool):
    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        return
    coll.hide_viewport = bool(hide)
    coll.hide_render = bool(hide)


def unique_collection_name(base: str, used: set[str]) -> str:
    name = base
    suffix = 1
    while name in used or name in bpy.data.collections:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def rot_z_matrix(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def transform_lbs_vertices(vertices, *, axis_matrix, yaw_deg, offset_xyz):
    verts = np.asarray(vertices, dtype=np.float32)
    verts = np.einsum("ij,tvj->tvi", axis_matrix, verts)
    center = 0.5 * (verts.reshape(-1, 3).min(axis=0) + verts.reshape(-1, 3).max(axis=0))
    yaw = rot_z_matrix(yaw_deg)
    verts = np.einsum("ij,tvj->tvi", yaw, verts - center[None, None, :])
    verts = verts + center[None, None, :]
    zmin = float(verts[..., 2].min())
    verts = verts.copy()
    verts[..., 2] -= zmin
    verts = verts + offset_xyz[None, None, :]
    return verts


def imported_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def import_fbx_template_mesh(fbx_path: Path):
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    imported = imported_objects(before)
    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
        raise RuntimeError(f"No mesh object in FBX: {fbx_path}")
    mesh_obj = max(mesh_objs, key=lambda o: len(o.data.vertices))
    mesh_data = mesh_obj.data.copy()
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    return mesh_data


def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def add_shapekey_animation(obj, vertices: np.ndarray):
    mesh = obj.data
    if mesh.shape_keys is not None:
        obj.shape_key_clear()
    num_frames = int(vertices.shape[0])
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    for vi, co in enumerate(vertices[0]):
        basis.data[vi].co = co
    key_blocks = []
    for fi in range(num_frames):
        kb = obj.shape_key_add(name=f"F{fi:04d}", from_mix=False)
        for vi, co in enumerate(vertices[fi]):
            kb.data[vi].co = co
        kb.value = 0.0
        key_blocks.append(kb)
    for fi, kb in enumerate(key_blocks):
        for t in range(num_frames):
            kb.value = 1.0 if t == fi else 0.0
            kb.keyframe_insert(data_path="value", frame=t + 1)
    return num_frames


def create_lbs_action_object(*, name, template_mesh, vertices, collection):
    mesh = template_mesh.copy()
    mesh.name = f"{name}_mesh"
    if len(mesh.vertices) != vertices.shape[1]:
        raise RuntimeError(
            f"Vertex count mismatch for {name}: "
            f"FBX template={len(mesh.vertices)} vs LBS={vertices.shape[1]}"
        )
    for vert, co in zip(mesh.vertices, vertices[0]):
        vert.co = co
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    add_shapekey_animation(obj, vertices)
    return obj


def pack_armature_actions(manifest, blend_opts, used_names):
    """
    Keep the textured FBX armature/mesh and bake Ours motion onto FBX bones.

    Requires per-action ``ours_fbx_bake_path`` (npz cache from export). Do NOT use
    mesh_visualize(FBX, BVH): Blender BVH import breaks rest axes vs FBX bind pose.
    """
    fbx_path = Path(manifest["target_fbx_path"])
    actions = list(manifest.get("actions") or [])
    layout = str(blend_opts.get("layout", "grid"))
    spacing = float(blend_opts.get("spacing", 2.5))
    grid_cols = int(blend_opts.get("grid_cols", 10))
    hide_all_but_first = bool(blend_opts.get("hide_all_but_first", True))
    rot_z_deg = float(blend_opts.get("rot_z_deg", 0.0))
    max_frames = 1

    for idx, action in enumerate(actions):
        action_id = str(action["action_id"])
        bake_path = action.get("ours_fbx_bake_path")
        if not bake_path:
            # Backward-compatible hint if user only has native BVH from older export.
            raise FileNotFoundError(
                f"[{manifest['dog_id']}/{action_id}] missing ours_fbx_bake_path; "
                "re-export (armature pack now bakes FBX bones from an npz cache)"
            )
        bake_path = Path(bake_path)
        if not bake_path.exists():
            raise FileNotFoundError(
                f"[{manifest['dog_id']}/{action_id}] bake cache missing: {bake_path}"
            )

        collection_name = unique_collection_name(
            str(action.get("collection_name") or action_id), used_names
        )
        x_bias, y_bias, z_bias = layout_offset(idx, layout, spacing, grid_cols)
        print(
            f"[pack-blend][armature-bake] ({idx + 1}/{len(actions)}) {action_id} "
            f"-> {collection_name} offset=({x_bias:.2f},{y_bias:.2f},{z_bias:.2f})"
        )

        arm, _meshes, imported = import_fbx_armature_and_meshes(fbx_path)
        n_frames = bake_cache_onto_fbx_armature(
            arm,
            bake_cache_path=bake_path,
            action_name=f"{manifest['dog_id']}_{action_id}_ours",
            frame_start=1,
        )
        apply_root_object_transform(
            imported,
            x_bias=x_bias,
            y_bias=y_bias,
            z_bias=z_bias,
            rot_z_deg=rot_z_deg,
        )
        move_objects_to_collection(imported, collection_name)

        max_frames = max(
            max_frames,
            int(action.get("num_frames") or 1),
            int(n_frames),
        )
        if hide_all_but_first and idx > 0:
            set_collection_hide(collection_name, True)
    return max_frames


def pack_lbs_actions(manifest, blend_opts, used_names, *, name_suffix=""):
    fbx_path = Path(manifest["target_fbx_path"])
    actions = list(manifest.get("actions") or [])
    layout = str(blend_opts.get("layout", "grid"))
    spacing = float(blend_opts.get("spacing", 2.5))
    grid_cols = int(blend_opts.get("grid_cols", 10))
    hide_all_but_first = bool(blend_opts.get("hide_all_but_first", True))
    yaw_deg = float(blend_opts.get("lbs_global_yaw_deg", blend_opts.get("rot_z_deg", 0.0)))
    axis_name = str(blend_opts.get("lbs_axis_transform", "y_up_to_z_up"))
    if axis_name not in AXIS_PRESETS:
        raise ValueError(f"Unknown lbs_axis_transform={axis_name}")
    axis_matrix = AXIS_PRESETS[axis_name]

    template = import_fbx_template_mesh(fbx_path)
    print(f"[pack-blend][lbs] FBX template verts={len(template.vertices)}")
    max_frames = 1

    for idx, action in enumerate(actions):
        action_id = str(action["action_id"])
        lbs_path = action.get("ours_lbs_path")
        if not lbs_path:
            raise FileNotFoundError(
                f"[{manifest['dog_id']}/{action_id}] missing ours_lbs_path"
            )
        lbs_path = Path(lbs_path)
        if not lbs_path.exists():
            raise FileNotFoundError(f"LBS npz missing: {lbs_path}")

        verts = np.load(str(lbs_path))["ours_vertices"].astype(np.float32)
        offset = layout_offset_np(idx, layout, spacing, grid_cols)
        verts = transform_lbs_vertices(
            verts, axis_matrix=axis_matrix, yaw_deg=yaw_deg, offset_xyz=offset
        )
        collection_name = unique_collection_name(
            f"{action.get('collection_name') or action_id}{name_suffix}", used_names
        )
        coll = ensure_collection(collection_name)
        print(
            f"[pack-blend][lbs] ({idx + 1}/{len(actions)}) {action_id} "
            f"frames={len(verts)} -> {collection_name}"
        )
        create_lbs_action_object(
            name=f"{action_id}{name_suffix}",
            template_mesh=template,
            vertices=verts,
            collection=coll,
        )
        max_frames = max(max_frames, int(verts.shape[0]))
        if hide_all_but_first and idx > 0:
            set_collection_hide(collection_name, True)
    return max_frames


def pack_one_dog(manifest: dict, blend_output_dir: Path, compress_override: bool | None) -> Path:
    dog_id = manifest["dog_id"]
    fbx_path = Path(manifest["target_fbx_path"])
    if not fbx_path.exists():
        raise FileNotFoundError(f"[{dog_id}] target FBX missing: {fbx_path}")

    actions = list(manifest.get("actions") or [])
    if not actions:
        raise RuntimeError(f"[{dog_id}] manifest has no exported actions")

    blend_opts = manifest.get("blend", {}) or {}
    pack_mode = str(
        blend_opts.get("pack_mode") or manifest.get("pack_mode") or "armature"
    ).strip().lower()
    fps = int(blend_opts.get("fps", 24))
    compress = (
        bool(compress_override)
        if compress_override is not None
        else bool(blend_opts.get("compress", True))
    )

    clean_scene()
    used_names: set[str] = set()
    max_frames = 1

    if pack_mode in ("armature", "both"):
        max_frames = max(max_frames, pack_armature_actions(manifest, blend_opts, used_names))
    if pack_mode in ("lbs", "both"):
        suffix = "_LBS" if pack_mode == "both" else ""
        max_frames = max(
            max_frames,
            pack_lbs_actions(manifest, blend_opts, used_names, name_suffix=suffix),
        )
    if pack_mode not in ("armature", "lbs", "both"):
        raise ValueError(f"Unknown pack_mode={pack_mode}")

    set_scene_animation_range(max_frames, fps)
    out_path = blend_output_dir / f"{dog_id}.blend"
    save_blend(out_path, compress=compress)
    print(
        f"[pack-blend] saved {out_path} "
        f"(actions={len(actions)}, max_frames={max_frames}, fps={fps}, pack_mode={pack_mode})"
    )
    return out_path


def load_manifests(args) -> list[dict]:
    manifests: list[dict] = []
    if args.manifest_index:
        index = json.loads(Path(args.manifest_index).read_text(encoding="utf-8"))
        manifests.extend(index)
    if args.manifest:
        manifests.append(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    if not manifests:
        raise SystemExit("Provide --manifest and/or --manifest_index")
    if args.dog_ids:
        wanted = set(args.dog_ids)
        manifests = [m for m in manifests if m.get("dog_id") in wanted]
    if not manifests:
        raise SystemExit("No manifests left after dog_ids filter.")
    return manifests


def main():
    args = parse_args(blender_argv())
    manifests = load_manifests(args)
    blend_output_dir = Path(args.blend_output_dir)
    blend_output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(str(_SCRIPT_DIR.parent))

    produced = []
    for manifest in manifests:
        produced.append(str(pack_one_dog(manifest, blend_output_dir, args.compress)))
    print(f"[pack-blend] done: {len(produced)} file(s) -> {blend_output_dir}")


if __name__ == "__main__":
    main()
