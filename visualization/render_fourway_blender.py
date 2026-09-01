#!/usr/bin/env python3
"""
Render four-way textured mesh comparison videos in Blender.

Expected input: manifest index produced by export_fourway_compare_smal33.py.

Run:
  blender --background --python visualization/render_fourway_blender.py -- \
      --manifest_index visualization/videos/compare/fourway_manifest_index.json \
      --render_engine eevee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import bpy
import math
import mathutils
import numpy as np


import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from visualize import clean_scene, mesh_visualize
from compare_lanes import compare_lane_titles, include_copyquat_lane, lane_layout_offset


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render four-way textured compare videos.")
    parser.add_argument("--manifest_index", type=str, required=True)
    parser.add_argument("--render_engine", type=str, default=None, choices=["eevee", "cycles"])
    parser.add_argument("--resolution_x", type=int, default=None)
    parser.add_argument("--resolution_y", type=int, default=None)
    parser.add_argument("--lane_spacing", type=float, default=None)
    parser.add_argument("--frame_step", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args(argv)


def blender_argv():
    import sys

    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def add_floor(size=80.0, checker_scale=24.0, rotation_deg=30.0, z_height=0.0):
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        enter_editmode=False,
        location=(0, 0, float(z_height)),
    )
    floor = bpy.context.object
    floor.name = "compare_floor"
    mat = bpy.data.materials.new(name="compare_floor_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    checker = mat.node_tree.nodes.new("ShaderNodeTexChecker")
    checker.inputs[3].default_value = float(checker_scale)
    mat.node_tree.links.new(bsdf.inputs["Base Color"], checker.outputs["Color"])
    floor.data.materials.append(mat)
    floor.rotation_euler[2] = math.radians(rotation_deg)
    return floor


def add_lights():
    bpy.ops.object.light_add(type="SUN", location=(20, -20, 35))
    sun = bpy.context.object
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(55), math.radians(-20), math.radians(40))

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 18))
    fill = bpy.context.object
    fill.data.energy = 600
    fill.data.size = 20
    return sun, fill


def bbox_world_for_meshes():
    corners = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for c in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(c))
    if not corners:
        return mathutils.Vector((0, 0, 0)), 1.0
    arr = np.asarray([[v.x, v.y, v.z] for v in corners], dtype=np.float64)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    center = (mn + mx) * 0.5
    radius = float(np.max(mx - mn) * 0.5)
    radius = max(radius, 0.8)
    return mathutils.Vector(center.tolist()), radius


def setup_camera(
    center,
    radius,
    elev_deg=18.0,
    azim_deg=-75.0,
    zoom=1.0,
    distance_scale=3.0,
    lens_mm=45.0,
):
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    dist = (radius * float(distance_scale)) / max(float(zoom), 1e-6)

    offset = mathutils.Vector(
        (
            dist * math.cos(elev) * math.cos(azim),
            dist * math.cos(elev) * math.sin(azim),
            dist * math.sin(elev) + radius * 0.2,
        )
    )
    cam_loc = center + offset
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    direction = (center - cam_loc).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = float(lens_mm)
    return cam


def setup_camera_from_absolute(position_xyz, target_xyz, lens_mm=45.0):
    cam_loc = mathutils.Vector(
        (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]))
    )
    target = mathutils.Vector(
        (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]))
    )
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    direction = (target - cam_loc).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = float(lens_mm)
    return cam


def parse_vec3(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected vec3 list/tuple, got: {value}")
    return [float(value[0]), float(value[1]), float(value[2])]


def resolve_lane_label_z(render_cfg, max_top_z, floor_z=0.0):
    """
    Absolute Blender Z for lane title labels.

    When ``lane_label_z`` is set in render config, use it directly so label
    height stays constant across actions (e.g. Jump no longer pushes titles
    out of frame). Otherwise fall back to mesh bbox top + offset.
    """
    render_cfg = render_cfg or {}
    fixed = render_cfg.get("lane_label_z")
    if fixed is not None:
        return float(fixed)
    label_z_offset = float(render_cfg.get("lane_label_z_offset", 0.25))
    return max(float(max_top_z), float(floor_z)) + label_z_offset


def add_lane_labels(center, lane_spacing, z_lift=2.3, font_size=0.35, render_cfg=None):
    render_cfg = render_cfg or {}
    labels = compare_lane_titles(render_cfg)
    y_offsets = [
        lane_layout_offset(i, len(labels), lane_spacing) for i in range(len(labels))
    ]
    label_z = resolve_lane_label_z(render_cfg, max_top_z=center.z + z_lift, floor_z=center.z)
    for text, y in zip(labels, y_offsets):
        bpy.ops.object.text_add(location=(center.x, center.y + y, label_z))
        obj = bpy.context.object
        obj.data.body = text
        obj.data.size = float(font_size)
        obj.rotation_euler = (math.radians(90), 0.0, math.radians(90))


def set_render_settings(scene, out_path, fps, frame_count, engine, res_x, res_y, frame_step=1):
    scene.render.resolution_x = int(res_x)
    scene.render.resolution_y = int(res_y)
    scene.render.fps = int(fps)
    scene.frame_start = 1
    scene.frame_end = max(int(frame_count), 1)
    scene.frame_step = max(int(frame_step), 1)

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "GPU"
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(out_path)


def assert_exists(path_str, what):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def lane_rotation_map(render_cfg):
    default_rot = float(render_cfg.get("lane_rot_z_deg_default", 90.0))
    lane_cfg = render_cfg.get("lane_rot_z_deg", {}) or {}
    return {
        "source": float(lane_cfg.get("source", default_rot)),
        "copyquat": float(lane_cfg.get("copyquat", default_rot)),
        "arp": float(lane_cfg.get("arp", default_rot)),
        "ours": float(lane_cfg.get("ours", default_rot)),
    }


def collection_min_z(collection_name: str, frame_start: int, frame_end: int) -> float | None:
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_z = None
    for frame in range(int(frame_start), int(frame_end) + 1):
        bpy.context.scene.frame_set(frame)
        for obj in collection.all_objects:
            if obj.type != "MESH":
                continue
            eval_obj = obj.evaluated_get(depsgraph)
            mesh_eval = eval_obj.to_mesh()
            if mesh_eval is None:
                continue
            try:
                for v in mesh_eval.vertices:
                    z = (eval_obj.matrix_world @ v.co).z
                    if min_z is None or z < min_z:
                        min_z = z
            finally:
                eval_obj.to_mesh_clear()
    return min_z


def shift_collection_z(collection_name: str, delta_z: float):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return
    for obj in collection.all_objects:
        obj.location.z += float(delta_z)


def align_collections_to_floor(collection_names, floor_z: float, frame_count: int):
    if frame_count <= 0:
        return
    for cname in collection_names:
        min_z = collection_min_z(cname, 1, frame_count)
        if min_z is None:
            continue
        shift = float(floor_z) - float(min_z)
        if abs(shift) > 1e-6:
            shift_collection_z(cname, shift)


def render_case(manifest, args):
    case_id = manifest["case_id"]
    case_dir = Path(manifest["npz_path"]).parent
    out_mp4 = case_dir / "fourway_compare.mp4"

    assert_exists(manifest["source_fbx_path"], "source_fbx_path")
    assert_exists(manifest["target_fbx_path"], "target_fbx_path")
    assert_exists(manifest["source_bvh_path"], "source_bvh_path")
    assert_exists(manifest["copyquat_bvh_path"], "copyquat_bvh_path")
    assert_exists(manifest["ours_bvh_path"], "ours_bvh_path")
    assert_exists(manifest["arp_bvh_path"], "arp_bvh_path")

    npz = np.load(manifest["npz_path"])
    frame_count = int(npz["frame_count"])
    fps = int(npz["fps"])

    render_cfg = manifest.get("render", {})
    floor_z = float(render_cfg.get("floor_z", 0.0))
    clean_scene()
    if bool(render_cfg.get("add_floor", True)):
        add_floor(
            size=float(render_cfg.get("floor_size", 80.0)),
            checker_scale=float(render_cfg.get("floor_checker_scale", 24.0)),
            rotation_deg=float(render_cfg.get("floor_rotation_deg", 30.0)),
            z_height=floor_z,
        )
    add_lights()

    lane_spacing = float(
        args.lane_spacing
        if args.lane_spacing is not None
        else render_cfg.get("lane_spacing", 3.6)
    )
    lane_rots = lane_rotation_map(render_cfg)
    lane_specs = [
        ("source", manifest["source_fbx_path"], manifest["source_bvh_path"], lane_rots["source"]),
        ("ours", manifest["target_fbx_path"], manifest["ours_bvh_path"], lane_rots["ours"]),
        ("arp", manifest["target_fbx_path"], manifest["arp_bvh_path"], lane_rots["arp"]),
    ]
    if include_copyquat_lane(render_cfg):
        lane_specs.append(
            ("copyquat", manifest["target_fbx_path"], manifest["copyquat_bvh_path"], lane_rots["copyquat"])
        )
    y_offsets = [
        lane_layout_offset(i, len(lane_specs), lane_spacing) for i in range(len(lane_specs))
    ]
    lane_collection_names = [f"{case_id}_{lane_key}" for lane_key, *_ in lane_specs]
    for lane_idx, (lane_key, fbx_path, bvh_path, rot_z_deg) in enumerate(lane_specs):
        mesh_visualize(
            fbx_path,
            bvh_path,
            x_bias=0.0,
            y_bias=y_offsets[lane_idx],
            z_bias=0.0,
            rot_z_deg=rot_z_deg,
            collection_name=lane_collection_names[lane_idx],
            source_format="h36m",
        )

    if bool(render_cfg.get("auto_align_floor", True)):
        align_collections_to_floor(
            lane_collection_names,
            floor_z=floor_z,
            frame_count=frame_count,
        )

    center, radius = bbox_world_for_meshes()
    camera_cfg = manifest.get("camera", {})
    absolute_pos = parse_vec3(render_cfg.get("camera_position_xyz"))
    absolute_target = parse_vec3(render_cfg.get("camera_target_xyz"))
    if absolute_pos is not None:
        if absolute_target is None:
            absolute_target = [center.x, center.y, center.z]
        cam = setup_camera_from_absolute(
            absolute_pos,
            absolute_target,
            lens_mm=float(render_cfg.get("camera_lens_mm", 45.0)),
        )
    else:
        cam = setup_camera(
            center,
            radius,
            elev_deg=float(camera_cfg.get("view_elev", 18.0)),
            azim_deg=float(camera_cfg.get("view_azim", -75.0)),
            zoom=float(camera_cfg.get("camera_zoom", 1.0)),
            distance_scale=float(render_cfg.get("camera_distance_scale", 3.0)),
            lens_mm=float(render_cfg.get("camera_lens_mm", 45.0)),
        )
    add_lane_labels(
        center,
        lane_spacing,
        z_lift=radius * 0.9 + 1.2,
        font_size=float(render_cfg.get("lane_label_size", 0.35)),
        render_cfg=render_cfg,
    )
    bpy.context.scene.camera = cam

    engine = args.render_engine or render_cfg.get("render_engine", "eevee")
    res_x = args.resolution_x or int(render_cfg.get("resolution_x", 1920))
    res_y = args.resolution_y or int(render_cfg.get("resolution_y", 1080))
    set_render_settings(
        bpy.context.scene,
        out_mp4,
        fps=fps,
        frame_count=frame_count,
        engine=engine,
        res_x=res_x,
        res_y=res_y,
        frame_step=(
            args.frame_step
            if args.frame_step is not None
            else int(render_cfg.get("frame_step", 1))
        ),
    )

    if args.dry_run:
        print(f"[render][dry-run] {case_id}: {out_mp4}")
    else:
        bpy.ops.render.render(animation=True)
        print(f"[render] done: {case_id} -> {out_mp4}")


def main():
    args = parse_args(blender_argv())
    manifest_index = Path(args.manifest_index)
    if not manifest_index.exists():
        raise SystemExit(f"manifest_index not found: {manifest_index}")

    manifests = json.loads(manifest_index.read_text(encoding="utf-8"))
    if not isinstance(manifests, list) or not manifests:
        raise SystemExit(f"Invalid manifest index: {manifest_index}")

    for m in manifests:
        render_case(m, args)


if __name__ == "__main__":
    main()
