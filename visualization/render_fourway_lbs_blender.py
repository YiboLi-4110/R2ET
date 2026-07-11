#!/usr/bin/env python3
"""
Render four-way SMAL33 comparison videos from precomputed LBS vertices.

This renderer intentionally avoids FBX<->BVH re-binding. It consumes
``fourway_compare.npz`` produced by ``export_fourway_compare_smal33.py`` and
updates mesh vertices frame-by-frame, matching the stable LBS path used by the
inspect scripts.

Run:
  blender --background --python visualization/render_fourway_lbs_blender.py -- \
    --manifest_index visualization/videos/compare/fourway_manifest_index.json \
    --render_engine eevee
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import mathutils
import numpy as np

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from render_fourway_blender import (
    add_floor,
    bbox_world_for_meshes,
    clean_scene,
    parse_vec3,
    set_render_settings,
    setup_camera,
    setup_camera_from_absolute,
)
from compare_lanes import include_copyquat_lane, lane_layout_offset


COPYQUAT_LANE = (
    "copyquat",
    "CopyQuat",
    "copyquat_vertices",
    "target_faces",
    (0.95, 0.48, 0.18, 1.0),
)
BASE_LANES = [
    ("source", "Source", "source_vertices", "source_faces", (0.72, 0.72, 0.72, 1.0)),
    ("ours", "R2ET", "ours_vertices", "target_faces", (0.25, 0.80, 0.35, 1.0)),
    ("arp", "ARP", "arp_vertices", "target_faces", (0.20, 0.48, 0.95, 1.0)),
]


def select_lanes(render_cfg):
    lanes = list(BASE_LANES)
    if include_copyquat_lane(render_cfg):
        lanes.append(COPYQUAT_LANE)
    return lanes

AXIS_PRESETS = {
    "identity": np.eye(3, dtype=np.float32),
    # Model/inspect vertices are currently Y-up. Blender floor is XY with Z-up.
    # new_x = old_x, new_y = old_z, new_z = old_y
    "y_up_to_z_up": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
    # Alternative handedness if the default appears mirrored.
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
    import sys

    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render four-way LBS compare videos.")
    parser.add_argument("--manifest_index", type=str, required=True)
    parser.add_argument("--render_engine", type=str, default=None, choices=["eevee", "cycles"])
    parser.add_argument("--resolution_x", type=int, default=None)
    parser.add_argument("--resolution_y", type=int, default=None)
    parser.add_argument("--lane_spacing", type=float, default=None)
    parser.add_argument("--frame_step", type=int, default=None)
    parser.add_argument("--output_name", type=str, default="fourway_compare_lbs.mp4")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args(argv)


def make_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def imported_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def import_fbx_template_mesh(fbx_path: str):
    before = set(obj.name for obj in bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    imported = imported_objects(before)
    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh object in FBX: {fbx_path}")
    mesh_obj = max(mesh_objs, key=lambda o: len(o.data.vertices))
    mesh_data = mesh_obj.data.copy()
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    return mesh_data


def mesh_from_vertices(name, vertices, faces, material):
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def mesh_from_template_vertices(name, vertices, template_mesh):
    mesh = template_mesh.copy()
    if len(mesh.vertices) != len(vertices):
        raise RuntimeError(
            f"Vertex count mismatch for {name}: template={len(mesh.vertices)} "
            f"vs vertices={len(vertices)}"
        )
    for vert, co in zip(mesh.vertices, vertices):
        vert.co = co
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def rot_z_matrix(deg):
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_lane_vertices(
    vertices,
    lane_index,
    num_lanes,
    lane_spacing,
    right_vec,
    row_center,
    floor_z,
    axis_matrix,
    lane_yaw_deg,
):
    verts = np.asarray(vertices, dtype=np.float32)
    verts = np.einsum("ij,tvj->tvi", axis_matrix, verts)

    center_before_yaw = (verts.reshape(-1, 3).min(axis=0) + verts.reshape(-1, 3).max(axis=0)) * 0.5
    yaw = rot_z_matrix(lane_yaw_deg)
    verts = np.einsum("ij,tvj->tvi", yaw, verts - center_before_yaw[None, None, :])
    verts = verts + center_before_yaw[None, None, :]

    mn = verts.reshape(-1, 3).min(axis=0)
    mx = verts.reshape(-1, 3).max(axis=0)
    center = (mn + mx) * 0.5
    lane_offset = lane_layout_offset(lane_index, num_lanes, lane_spacing)
    target_center = np.asarray(row_center, dtype=np.float32) + np.asarray(right_vec, dtype=np.float32) * lane_offset
    delta = target_center - center
    delta[2] = float(floor_z) - float(mn[2])
    return verts + delta[None, None, :]


def lbs_axis_matrix(render_cfg):
    axis_name = str(render_cfg.get("lbs_axis_transform", "y_up_to_z_up"))
    if axis_name not in AXIS_PRESETS:
        known = ", ".join(sorted(AXIS_PRESETS))
        raise ValueError(f"Unknown render.lbs_axis_transform '{axis_name}'. Known: {known}")
    return AXIS_PRESETS[axis_name]


def lane_yaw_degrees(render_cfg, lanes):
    default_yaw = float(render_cfg.get("lbs_lane_yaw_deg_default", 0.0))
    lane_cfg = render_cfg.get("lbs_lane_yaw_deg", {}) or {}
    global_yaw = float(render_cfg.get("lbs_global_yaw_deg", 0.0))
    return {
        lane: global_yaw + float(lane_cfg.get(lane, default_yaw))
        for lane, *_ in lanes
    }


def lane_template_key(lane_key: str) -> str:
    return "source" if lane_key == "source" else "target"


def camera_right_vector(render_cfg, bbox_center):
    pos = parse_vec3(render_cfg.get("camera_position_xyz"))
    target = parse_vec3(render_cfg.get("camera_target_xyz"))
    if pos is not None:
        if target is None:
            target = [bbox_center.x, bbox_center.y, bbox_center.z]
        direction = mathutils.Vector(target) - mathutils.Vector(pos)
    else:
        # Auto camera usually orbits around the scene; use world Y as the stable
        # horizontal row axis until the auto camera is created.
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if direction.length < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    direction.normalize()
    up = mathutils.Vector((0.0, 0.0, 1.0))
    right = direction.cross(up)
    if right.length < 1e-8:
        right = mathutils.Vector((0.0, 1.0, 0.0))
    else:
        right.normalize()
    return np.array([right.x, right.y, right.z], dtype=np.float32)


def make_frame_handler(animated):
    def fourway_lbs_frame_handler(scene):
        frame = max(scene.frame_current - scene.frame_start, 0)
        for item in animated:
            verts = item["vertices"]
            idx = min(frame, len(verts) - 1)
            obj = item["object"]
            mesh = obj.data
            frame_vertices = item["vertices"][idx]
            for vert, co in zip(mesh.vertices, frame_vertices):
                vert.co = co
            mesh.update()

    fourway_lbs_frame_handler.__name__ = "fourway_lbs_frame_handler"
    return fourway_lbs_frame_handler


def clear_frame_handlers():
    for handler in list(bpy.app.handlers.frame_change_pre):
        if getattr(handler, "__name__", "") == "fourway_lbs_frame_handler":
            bpy.app.handlers.frame_change_pre.remove(handler)


def add_legacy_lights(render_cfg):
    """Legacy lighting preset (SUN + AREA), with optional config overrides."""
    sun_loc = parse_vec3(render_cfg.get("sun_light_position_xyz")) or [20.0, -20.0, 35.0]
    sun_rot_deg = parse_vec3(render_cfg.get("sun_light_rotation_deg")) or [55.0, -20.0, 40.0]
    sun_energy = float(render_cfg.get("sun_light_energy", 3.5))
    bpy.ops.object.light_add(type="SUN", location=tuple(sun_loc))
    sun = bpy.context.object
    sun.name = "compare_sun_light"
    sun.data.energy = sun_energy
    sun.rotation_euler = tuple(math.radians(v) for v in sun_rot_deg)

    fill_loc = parse_vec3(render_cfg.get("fill_light_position_xyz")) or [0.0, 0.0, 18.0]
    fill_energy = float(render_cfg.get("fill_light_energy", 600.0))
    fill_size = float(render_cfg.get("fill_light_size", 20.0))
    bpy.ops.object.light_add(type="AREA", location=tuple(fill_loc))
    fill = bpy.context.object
    fill.name = "compare_fill_light"
    fill.data.energy = fill_energy
    fill.data.size = fill_size
    return sun, fill


def add_labels(label_specs, font_size, color):
    mat = make_material("lane_label_mat", color)
    label_objects = []
    for label, location in label_specs:
        bpy.ops.object.text_add(location=location)
        obj = bpy.context.object
        obj.name = f"label_{label}"
        obj.data.body = label
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.size = float(font_size)
        obj.data.materials.append(mat)
        obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(90.0))
        label_objects.append(obj)
    return label_objects


def lane_bbox_center_top(vertices_frame):
    mn = vertices_frame.reshape(-1, 3).min(axis=0)
    mx = vertices_frame.reshape(-1, 3).max(axis=0)
    center = (mn + mx) * 0.5
    return center, float(mx[2])


def render_case(manifest, args):
    case_id = manifest["case_id"]
    case_dir = Path(manifest["npz_path"]).parent
    out_mp4 = case_dir / args.output_name
    npz = np.load(manifest["npz_path"])
    frame_count = int(npz["frame_count"])
    fps = int(npz["fps"])

    render_cfg = manifest.get("render", {})
    camera_cfg = manifest.get("camera", {})
    lanes = select_lanes(render_cfg)
    num_lanes = len(lanes)
    floor_z = float(render_cfg.get("floor_z", 0.0))
    use_fbx_materials = bool(render_cfg.get("lbs_use_fbx_materials", False))
    lane_spacing = float(
        args.lane_spacing
        if args.lane_spacing is not None
        else render_cfg.get("lane_spacing", 3.6)
    )

    clean_scene()
    clear_frame_handlers()
    if bool(render_cfg.get("add_floor", True)):
        add_floor(
            size=float(render_cfg.get("floor_size", 80.0)),
            checker_scale=float(render_cfg.get("floor_checker_scale", 24.0)),
            rotation_deg=float(render_cfg.get("floor_rotation_deg", 30.0)),
            z_height=floor_z,
        )
    row_center = parse_vec3(render_cfg.get("layout_center_xyz"))
    if row_center is None:
        row_center = parse_vec3(render_cfg.get("camera_target_xyz")) or [0.0, 0.0, 0.0]
    add_legacy_lights(render_cfg)
    dummy_center = mathutils.Vector(row_center)
    right_vec = camera_right_vector(render_cfg, dummy_center)
    axis_matrix = lbs_axis_matrix(render_cfg)
    lane_yaws = lane_yaw_degrees(render_cfg, lanes)
    template_meshes = {}
    if use_fbx_materials:
        template_meshes["source"] = import_fbx_template_mesh(manifest["source_fbx_path"])
        template_meshes["target"] = import_fbx_template_mesh(manifest["target_fbx_path"])

    animated = []
    label_xy = []
    max_top_z = float(floor_z)
    label_z_offset = float(render_cfg.get("lane_label_z_offset", 0.25))
    for lane_idx, (lane_key, title, verts_key, faces_key, color) in enumerate(lanes):
        verts = transform_lane_vertices(
            npz[verts_key],
            lane_idx,
            num_lanes,
            lane_spacing,
            right_vec,
            row_center,
            floor_z,
            axis_matrix,
            lane_yaws[lane_key],
        )
        if use_fbx_materials:
            template_key = lane_template_key(lane_key)
            obj = mesh_from_template_vertices(
                f"{case_id}_{lane_key}",
                verts[0],
                template_meshes[template_key],
            )
        else:
            material = make_material(f"{case_id}_{lane_key}_mat", color)
            obj = mesh_from_vertices(f"{case_id}_{lane_key}", verts[0], npz[faces_key], material)
        animated.append({"object": obj, "vertices": verts})

        mn = verts.reshape(-1, 3).min(axis=0)
        mx = verts.reshape(-1, 3).max(axis=0)
        lane_offset = lane_layout_offset(lane_idx, num_lanes, lane_spacing)
        center_xy = np.asarray(row_center, dtype=np.float32) + np.asarray(right_vec, dtype=np.float32) * lane_offset
        max_top_z = max(max_top_z, float(mx[2]))
        label_xy.append((title, float(center_xy[0]), float(center_xy[1])))

    unified_label_z = max_top_z + label_z_offset
    label_specs = [(title, (x, y, unified_label_z)) for title, x, y in label_xy]

    center, radius = bbox_world_for_meshes()
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
    bpy.context.scene.camera = cam
    add_labels(
        label_specs,
        float(render_cfg.get("lane_label_size", 0.35)),
        tuple(render_cfg.get("lane_label_color", [1.0, 0.0, 0.0, 1.0])),
    )

    bpy.context.scene.frame_set(1)
    clear_frame_handlers()
    bpy.app.handlers.frame_change_pre.append(
        make_frame_handler(animated)
    )

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
        print(f"[render-lbs][dry-run] {case_id}: {out_mp4}")
    else:
        bpy.ops.render.render(animation=True)
        print(f"[render-lbs] done: {case_id} -> {out_mp4}")


def main():
    args = parse_args(blender_argv())
    manifest_index = Path(args.manifest_index)
    if not manifest_index.exists():
        raise SystemExit(f"manifest_index not found: {manifest_index}")

    manifests = json.loads(manifest_index.read_text(encoding="utf-8"))
    if not isinstance(manifests, list) or not manifests:
        raise SystemExit(f"Invalid manifest index: {manifest_index}")

    for manifest in manifests:
        render_case(manifest, args)


if __name__ == "__main__":
    main()

