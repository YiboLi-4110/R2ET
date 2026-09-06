#!/usr/bin/env python3
"""
Render a single-animal ARP sequence video from stitched LBS vertices.

Consumes ``arp_sequence.npz`` + manifest produced by
``batch_arp_sequence_smal33.py``. Camera / floor / lighting knobs match the
four-way compare LBS renderer.

When ``render.simple_preview`` is true, skips the checker floor. Source clay is
self-lit (emission). Retarget stays matte and is lifted with isotropic world
lighting plus a small albedo emission — no directional lights, which faceted
the textured mesh. The camera still sees the dark preview background.

When ``render.show_source`` / manifest ``show_source`` is true, shows a Source
skinned-mesh lane beside the retargeted result (fourway-style side-by-side).

When ``render.show_action_label`` is true, shows the current clip BVH stem
(no suffix) as a top-left camera HUD.

Run:
  blender --background --python visualization/render_arp_sequence_lbs_blender.py -- \
    --manifest visualization/videos/arp_sequence/<id>/arp_sequence_manifest.json \
    --render_engine eevee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
import mathutils
import numpy as np

import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_sequence_common import (  # noqa: E402
    action_labels_per_frame,
    build_action_label_segments,
    build_clip_local_frame_map,
    load_arp_mesh_for_sequence,
)
from compare_lanes import lane_layout_offset  # noqa: E402
from render_fourway_blender import (  # noqa: E402
    add_floor,
    bbox_world_for_meshes,
    clean_scene,
    parse_vec3,
    resolve_lane_label_z,
    set_render_settings,
    setup_camera,
    setup_camera_from_absolute,
)
from render_fourway_lbs_blender import (  # noqa: E402
    add_labels,
    add_legacy_lights,
    camera_right_vector,
    clear_frame_handlers,
    import_fbx_template_mesh,
    lbs_axis_matrix,
    make_material,
    mesh_from_template_vertices,
    mesh_from_vertices,
    transform_lane_vertices,
)


def transform_lane_vertices_by_clips(
    stitched_vertices: np.ndarray,
    clip_frame_counts: list[int],
    *,
    pause_frames: int,
    pause_mode: str,
    lane_index: int,
    num_lanes: int,
    lane_spacing: float,
    right_vec,
    row_center,
    floor_z: float,
    axis_matrix,
    lane_yaw_deg: float,
) -> np.ndarray:
    """
    Floor/center each clip independently, then scatter back to the stitched timeline.

    Matches Source-lane per-clip placement so multi-clip sequences keep Source and
    Retarget on the same ground plane for each action (global AABB over the full
    stitch would lift Retarget using the lowest clip and desync heights).
    """
    stitched = np.asarray(stitched_vertices, dtype=np.float32)
    if stitched.ndim != 3:
        raise ValueError(f"stitched_vertices must be [T,V,3], got {stitched.shape}")
    frame_map = build_clip_local_frame_map(
        clip_frame_counts,
        pause_frames=pause_frames,
        pause_mode=pause_mode,
    )
    if len(frame_map) != stitched.shape[0]:
        raise ValueError(
            f"clip frame map length {len(frame_map)} != stitched T={stitched.shape[0]}"
        )

    groups: dict[int, list[int]] = {}
    for timeline_idx, (clip_idx, _local) in enumerate(frame_map):
        groups.setdefault(int(clip_idx), []).append(timeline_idx)

    out = np.empty_like(stitched)
    for clip_idx in sorted(groups):
        times = groups[clip_idx]
        block = stitched[times]
        placed = transform_lane_vertices(
            block,
            lane_index=lane_index,
            num_lanes=num_lanes,
            lane_spacing=lane_spacing,
            right_vec=right_vec,
            row_center=row_center,
            floor_z=floor_z,
            axis_matrix=axis_matrix,
            lane_yaw_deg=lane_yaw_deg,
        )
        out[times] = placed
    return out


def blender_argv():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render ARP multi-clip sequence video.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--render_engine", type=str, default=None, choices=["eevee", "cycles"])
    parser.add_argument("--resolution_x", type=int, default=None)
    parser.add_argument("--resolution_y", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=None)
    parser.add_argument("--output_name", type=str, default="arp_sequence.mp4")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args(argv)


def arp_yaw_degrees(render_cfg: dict, lane_key: str = "retarget") -> float:
    global_yaw = float(render_cfg.get("lbs_global_yaw_deg", 0.0))
    default_yaw = float(render_cfg.get("lbs_lane_yaw_deg_default", 0.0))
    lane_cfg = render_cfg.get("lbs_lane_yaw_deg", {}) or {}
    # Backward-compatible aliases: retarget <- arp / ours
    if lane_key in lane_cfg:
        lane_yaw = float(lane_cfg.get(lane_key))
    elif lane_key == "retarget":
        lane_yaw = float(lane_cfg.get("arp", lane_cfg.get("ours", default_yaw)))
    else:
        lane_yaw = float(default_yaw)
    return global_yaw + lane_yaw


def resolve_mesh_scale(render_cfg: dict, lane_key: str = "retarget") -> float:
    """Uniform character scale. 1.0 keeps LBS native size."""
    per_lane = render_cfg.get("mesh_scale_by_lane") or {}
    if lane_key in per_lane:
        scale = float(per_lane.get(lane_key))
    else:
        scale = float(render_cfg.get("mesh_scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"render.mesh_scale must be > 0, got {scale}")
    return scale


def scale_mesh_vertices_about_floor(
    vertices: np.ndarray,
    scale: float,
    floor_z: float,
    *,
    up_axis: int = 2,
) -> np.ndarray:
    """Scale about the clip's horizontal AABB center, pinned to the floor."""
    verts = np.asarray(vertices, dtype=np.float32)
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return verts
    flat = verts.reshape(-1, 3)
    pivot = 0.5 * (flat.min(axis=0) + flat.max(axis=0))
    pivot[int(up_axis)] = float(floor_z)
    return (verts - pivot) * np.float32(scale) + pivot


def _parse_rgb(value, default=(0.92, 0.92, 0.92)):
    if value is None:
        return tuple(float(v) for v in default)
    if len(value) < 3:
        return tuple(float(v) for v in default)
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_rgba(value, default=(0.08, 0.08, 0.08, 1.0)):
    if value is None:
        return tuple(float(v) for v in default)
    vals = [float(v) for v in value]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) < 4:
        return tuple(float(v) for v in default)
    return (vals[0], vals[1], vals[2], vals[3])


def setup_simple_preview_world(
    bg_rgb=(0.92, 0.92, 0.92),
    strength=1.15,
    light_rgb=None,
    light_strength=None,
):
    """
    Visible background vs BSDF illumination.

    Camera rays keep ``bg_rgb`` (so Source clay stays readable). Indirect /
    lighting rays use a brighter env so textured retarget animals are not
    crushed when the preview backdrop is dark.
    """
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    cam_bg = nodes.new(type="ShaderNodeBackground")
    cam_bg.inputs["Color"].default_value = (
        float(bg_rgb[0]),
        float(bg_rgb[1]),
        float(bg_rgb[2]),
        1.0,
    )
    cam_bg.inputs["Strength"].default_value = float(strength)
    out = nodes.new(type="ShaderNodeOutputWorld")
    if light_rgb is None and light_strength is None:
        links.new(cam_bg.outputs["Background"], out.inputs["Surface"])
        return
    lit_rgb = light_rgb if light_rgb is not None else (0.50, 0.50, 0.52)
    lit_strength = 1.25 if light_strength is None else float(light_strength)
    lit_bg = nodes.new(type="ShaderNodeBackground")
    lit_bg.inputs["Color"].default_value = (
        float(lit_rgb[0]),
        float(lit_rgb[1]),
        float(lit_rgb[2]),
        1.0,
    )
    lit_bg.inputs["Strength"].default_value = float(lit_strength)
    mix = nodes.new(type="ShaderNodeMixShader")
    lp = nodes.new(type="ShaderNodeLightPath")
    if "Is Camera Ray" in lp.outputs:
        links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(lit_bg.outputs["Background"], mix.inputs[1])
    links.new(cam_bg.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])


def disable_scene_shadows(scene):
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        if hasattr(data, "use_shadow"):
            data.use_shadow = False
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "use_shadows"):
        eevee.use_shadows = False


def _copy_color_socket(links, src_input, dst_input):
    if src_input is None or dst_input is None:
        return
    for link in list(dst_input.links):
        links.remove(link)
    if getattr(src_input, "is_linked", False) and src_input.links:
        links.new(src_input.links[0].from_socket, dst_input)
        return
    try:
        dst_input.default_value = src_input.default_value
    except Exception:
        pass


def soften_object_materials(obj, roughness=0.9, specular=0.05, emission_lift=0.0):
    """Keep FBX fur readable under flat ambient: high roughness, optional lift."""
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not getattr(mat, "use_nodes", False):
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            continue
        links = mat.node_tree.links
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = float(roughness)
        for key in ("Specular IOR Level", "Specular"):
            if key in bsdf.inputs:
                try:
                    bsdf.inputs[key].default_value = float(specular)
                except Exception:
                    pass
                break
        if emission_lift <= 1e-6:
            continue
        emit_color = None
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                emit_color = bsdf.inputs[key]
                break
        emit_strength = (
            bsdf.inputs["Emission Strength"]
            if "Emission Strength" in bsdf.inputs
            else None
        )
        base_color = bsdf.inputs["Base Color"] if "Base Color" in bsdf.inputs else None
        if emit_color is None or emit_strength is None:
            continue
        _copy_color_socket(links, base_color, emit_color)
        emit_strength.default_value = float(emission_lift)


def _sun_rotation_from_dir(light_dir):
    incoming = mathutils.Vector(
        (float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))
    )
    if incoming.length < 1e-8:
        incoming = mathutils.Vector((0.32, -0.22, 0.92))
    incoming.normalize()
    return (-incoming).to_track_quat("-Z", "Y").to_euler()


def add_simple_preview_retarget_lights(render_cfg):
    """
    Shadowless key/fill for Principled (retarget) meshes.

    Source clay is emission, so these lights do not wash it out.
    """
    if not bool(render_cfg.get("simple_preview_retarget_lights", False)):
        return
    key_dir = _parse_vec3_cfg(
        render_cfg.get("simple_preview_key_dir")
        or render_cfg.get("source_key_light_dir"),
        (0.32, -0.22, 0.92),
    )
    fill_dir = _parse_vec3_cfg(
        render_cfg.get("simple_preview_fill_dir"),
        (0.55, 0.35, 0.76),
    )
    key_energy = float(render_cfg.get("simple_preview_key_energy", 3.2))
    fill_energy = float(render_cfg.get("simple_preview_fill_energy", 1.1))
    key_color = _parse_rgb(render_cfg.get("simple_preview_key_color"), (1.0, 0.98, 0.94))
    fill_color = _parse_rgb(render_cfg.get("simple_preview_fill_color"), (0.92, 0.94, 1.0))

    def _add_sun(name, direction, energy, color):
        bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 6.0))
        sun = bpy.context.object
        sun.name = name
        sun.data.energy = float(energy)
        sun.data.color = (float(color[0]), float(color[1]), float(color[2]))
        if hasattr(sun.data, "angle"):
            try:
                sun.data.angle = 0.40
            except Exception:
                pass
        if hasattr(sun.data, "use_shadow"):
            sun.data.use_shadow = False
        sun.rotation_euler = _sun_rotation_from_dir(direction)
        return sun

    if key_energy > 1e-6:
        _add_sun("preview_retarget_key", key_dir, key_energy, key_color)
    if fill_energy > 1e-6:
        _add_sun("preview_retarget_fill", fill_dir, fill_energy, fill_color)


def make_emission_material(name, color, strength=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emit = nodes.new(type="ShaderNodeEmission")
    emit.inputs["Color"].default_value = tuple(float(c) for c in color)
    emit.inputs["Strength"].default_value = float(strength)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _set_mix_rgba(mix_node, fac, color_a, color_b):
    """Blender 4.x ShaderNodeMix (RGBA) or legacy MixRGB."""
    color_a = tuple(float(c) for c in color_a)
    color_b = tuple(float(c) for c in color_b)
    if mix_node.bl_idname == "ShaderNodeMix":
        mix_node.data_type = "RGBA"
        # Duplicate socket names exist; use enabled RGBA sockets by index.
        # Factor(float)=0, A(RGBA)=6, B(RGBA)=7, Result(RGBA)=2
        fac_sock = mix_node.inputs[0]
        a_sock = mix_node.inputs[6]
        b_sock = mix_node.inputs[7]
        out_sock = mix_node.outputs[2]
        fac_sock.default_value = float(fac)
        a_sock.default_value = color_a
        b_sock.default_value = color_b
        return fac_sock, out_sock
    mix_node.blend_type = "MIX"
    mix_node.inputs["Fac"].default_value = float(fac)
    mix_node.inputs["Color1"].default_value = color_a
    mix_node.inputs["Color2"].default_value = color_b
    return mix_node.inputs["Fac"], mix_node.outputs["Color"]


def source_material_mode(render_cfg) -> str:
    raw = str((render_cfg or {}).get("source_material_mode", "wireframe")).strip().lower()
    if raw in ("wireframe", "wire", "tris", "triangles"):
        return "wireframe"
    if raw in ("shaded", "fill", "smooth", "nowire", "no_wire", "no_wireframe"):
        return "shaded"
    if raw in ("solid", "principled", "diffuse"):
        return "solid"
    if raw in ("emission", "emit", "unlit"):
        return "emission"
    return "wireframe"


def make_source_fill_material(name, render_cfg, *, with_wireframe: bool):
    """
    Face fill + mild facing shade (same look as the original Source lane).

    ``with_wireframe=True`` overlays triangle edges; False keeps the fill only.
    """
    face_color = _parse_rgba(
        render_cfg.get("source_material_color"),
        (0.86, 0.86, 0.90, 1.0),
    )
    shade_color = _parse_rgba(
        render_cfg.get("source_shade_color"),
        (
            max(face_color[0] * 0.55, 0.05),
            max(face_color[1] * 0.55, 0.05),
            max(face_color[2] * 0.55, 0.05),
            face_color[3],
        ),
    )
    wire_color = _parse_rgba(
        render_cfg.get("source_wire_color"),
        (0.12, 0.12, 0.14, 1.0),
    )
    face_strength = float(render_cfg.get("source_emission_strength", 1.15))
    wire_strength = float(render_cfg.get("source_wire_emission_strength", 1.6))
    wire_size = float(render_cfg.get("source_wire_size", 0.7))
    use_pixel_size = bool(render_cfg.get("source_wire_pixel_size", True))
    form_blend = float(render_cfg.get("source_form_blend", 0.42))

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")

    # Mild facing term so silhouettes/folds read as 3D under flat ambient light.
    layer = nodes.new(type="ShaderNodeLayerWeight")
    if "Blend" in layer.inputs:
        layer.inputs["Blend"].default_value = form_blend
    try:
        mix_rgb = nodes.new(type="ShaderNodeMix")
    except Exception:
        mix_rgb = nodes.new(type="ShaderNodeMixRGB")
    # Mix: Result ~= (1-Fac)*A + Fac*B. Facing=1 (to camera) -> bright face (B).
    fac_in, mix_out = _set_mix_rgba(mix_rgb, 0.5, shade_color, face_color)
    if "Facing" in layer.outputs:
        links.new(layer.outputs["Facing"], fac_in)
    else:
        links.new(layer.outputs[0], fac_in)

    face_emit = nodes.new(type="ShaderNodeEmission")
    face_emit.inputs["Strength"].default_value = face_strength
    links.new(mix_out, face_emit.inputs["Color"])

    if not with_wireframe:
        links.new(face_emit.outputs["Emission"], out.inputs["Surface"])
        return mat

    wire = nodes.new(type="ShaderNodeWireframe")
    wire.use_pixel_size = use_pixel_size
    if "Size" in wire.inputs:
        wire.inputs["Size"].default_value = wire_size
    wire_emit = nodes.new(type="ShaderNodeEmission")
    wire_emit.inputs["Color"].default_value = tuple(float(c) for c in wire_color)
    wire_emit.inputs["Strength"].default_value = wire_strength

    mix_shader = nodes.new(type="ShaderNodeMixShader")
    links.new(wire.outputs["Fac"], mix_shader.inputs["Fac"])
    links.new(face_emit.outputs["Emission"], mix_shader.inputs[1])
    links.new(wire_emit.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], out.inputs["Surface"])
    return mat


def _force_opaque_material(mat):
    """Keep Source clay fully opaque in EEVEE / EEVEE Next."""
    if hasattr(mat, "blend_method"):
        mat.blend_method = "OPAQUE"
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = "OPAQUE"
        except Exception:
            pass
    if hasattr(mat, "show_transparent_back"):
        mat.show_transparent_back = False
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = False
    if hasattr(mat, "use_raytrace_refraction"):
        mat.use_raytrace_refraction = False
    if hasattr(mat, "surface_render_method"):
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            pass
    if hasattr(mat, "use_transparency_overlap"):
        mat.use_transparency_overlap = False
    if hasattr(mat, "diffuse_color") and len(mat.diffuse_color) >= 4:
        mat.diffuse_color = (
            float(mat.diffuse_color[0]),
            float(mat.diffuse_color[1]),
            float(mat.diffuse_color[2]),
            1.0,
        )


def _parse_vec3_cfg(value, default):
    if value is None:
        return tuple(float(v) for v in default)
    vals = [float(v) for v in value]
    if len(vals) < 3:
        return tuple(float(v) for v in default)
    return (vals[0], vals[1], vals[2])


def _math(nodes, op, a=None, b=None):
    node = nodes.new(type="ShaderNodeMath")
    node.operation = op
    if a is not None:
        node.inputs[0].default_value = float(a)
    if b is not None and len(node.inputs) > 1:
        node.inputs[1].default_value = float(b)
    return node


def make_source_volume_material(name, render_cfg):
    """
    Clay-like Source shading without triangle edges.

    Self-contained (emission) so simple_preview's flat world does not wash it
    out, and the textured retarget dog is left unchanged: wrap Lambert key
    light + cavity AO + fresnel rim.
    """
    face_color = _parse_rgba(
        render_cfg.get("source_material_color"),
        (0.98, 0.96, 0.90, 1.0),
    )
    shade_color = _parse_rgba(
        render_cfg.get("source_shade_color"),
        (0.16, 0.16, 0.18, 1.0),
    )
    # Extra darken on the unlit side so a near-white fill still reads as volume.
    contrast = float(render_cfg.get("source_volume_contrast", 1.85))
    ambient = tuple(
        max(min(c / max(contrast, 1e-6), 1.0), 0.0) for c in shade_color[:3]
    ) + (shade_color[3],)
    wrap = float(render_cfg.get("source_key_light_wrap", 0.32))
    wrap = max(min(wrap, 0.95), 0.0)
    ao_distance = float(render_cfg.get("source_ao_distance", 0.18))
    ao_factor = float(render_cfg.get("source_ao_factor", 0.72))
    rim_strength = float(render_cfg.get("source_rim_strength", 0.18))
    rim_color = _parse_rgba(
        render_cfg.get("source_rim_color"),
        (
            min(face_color[0] * 1.05, 1.0),
            min(face_color[1] * 1.05, 1.0),
            min(face_color[2] * 1.08, 1.0),
            1.0,
        ),
    )
    strength = float(render_cfg.get("source_shaded_emission_strength", 1.05))
    form_blend = float(render_cfg.get("source_form_blend", 0.42))
    light_dir = _parse_vec3_cfg(
        render_cfg.get("source_key_light_dir"),
        (0.28, -0.18, 0.94),  # world +Z up, slight camera-side key
    )

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    _force_opaque_material(mat)
    # Winding is corrected on the mesh; cull inner faces so legs cannot show through.
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    geom = nodes.new(type="ShaderNodeNewGeometry")

    light = nodes.new(type="ShaderNodeCombineXYZ")
    light.inputs[0].default_value = float(light_dir[0])
    light.inputs[1].default_value = float(light_dir[1])
    light.inputs[2].default_value = float(light_dir[2])
    light_n = nodes.new(type="ShaderNodeVectorMath")
    light_n.operation = "NORMALIZE"
    links.new(light.outputs["Vector"], light_n.inputs[0])

    nrm = nodes.new(type="ShaderNodeVectorMath")
    nrm.operation = "NORMALIZE"
    if "Normal" in geom.outputs:
        links.new(geom.outputs["Normal"], nrm.inputs[0])
    else:
        links.new(geom.outputs[0], nrm.inputs[0])
    # Flipped winding: outer faces are backfacing, so N points inward and
    # a +Z key looks like ground-up lighting. Flip N on backfaces.
    nrm_neg = nodes.new(type="ShaderNodeVectorMath")
    nrm_neg.operation = "SCALE"
    links.new(nrm.outputs["Vector"], nrm_neg.inputs[0])
    scale_in = nrm_neg.inputs["Scale"] if "Scale" in nrm_neg.inputs else nrm_neg.inputs[3]
    scale_in.default_value = -1.0
    try:
        mix_n = nodes.new(type="ShaderNodeMix")
        mix_n.data_type = "VECTOR"
        n_fac = mix_n.inputs[0]
        n_a = mix_n.inputs[4] if len(mix_n.inputs) > 4 else mix_n.inputs[1]
        n_b = mix_n.inputs[5] if len(mix_n.inputs) > 5 else mix_n.inputs[2]
        n_out = mix_n.outputs[1] if len(mix_n.outputs) > 1 else mix_n.outputs[0]
    except Exception:
        mix_n = nodes.new(type="ShaderNodeMixRGB")
        n_fac = mix_n.inputs["Fac"]
        n_a = mix_n.inputs["Color1"]
        n_b = mix_n.inputs["Color2"]
        n_out = mix_n.outputs["Color"]
    back = geom.outputs["Backfacing"] if "Backfacing" in geom.outputs else None
    if back is not None:
        links.new(back, n_fac)
    links.new(nrm.outputs["Vector"], n_a)
    links.new(nrm_neg.outputs["Vector"], n_b)

    dot = nodes.new(type="ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    links.new(n_out, dot.inputs[0])
    links.new(light_n.outputs["Vector"], dot.inputs[1])
    dot_out = dot.outputs["Value"] if "Value" in dot.outputs else dot.outputs[0]

    # wrap Lambert: saturate((N·L + wrap) / (1 + wrap))
    add_w = _math(nodes, "ADD", b=wrap)
    links.new(dot_out, add_w.inputs[0])
    div_w = _math(nodes, "DIVIDE", b=1.0 + wrap)
    links.new(add_w.outputs["Value"], div_w.inputs[0])
    clamp_w = _math(nodes, "MAXIMUM", b=0.0)
    links.new(div_w.outputs["Value"], clamp_w.inputs[0])
    clamp_h = _math(nodes, "MINIMUM", b=1.0)
    links.new(clamp_w.outputs["Value"], clamp_h.inputs[0])
    # Contrast: raise wrapped term so the dark side stays dark.
    gamma = _math(nodes, "POWER", b=1.25)
    links.new(clamp_h.outputs["Value"], gamma.inputs[0])

    try:
        mix_lit = nodes.new(type="ShaderNodeMix")
    except Exception:
        mix_lit = nodes.new(type="ShaderNodeMixRGB")
    fac_lit, lit_out = _set_mix_rgba(mix_lit, 0.5, ambient, face_color)
    links.new(gamma.outputs["Value"], fac_lit)

    ao = nodes.new(type="ShaderNodeAmbientOcclusion")
    if "Normal" in ao.inputs:
        links.new(n_out, ao.inputs["Normal"])
    if "Distance" in ao.inputs:
        ao.inputs["Distance"].default_value = ao_distance
    if hasattr(ao, "samples"):
        try:
            ao.samples = 8
        except Exception:
            pass
    ao_inv = _math(nodes, "SUBTRACT", a=1.0)
    ao_src = ao.outputs["AO"] if "AO" in ao.outputs else ao.outputs[0]
    links.new(ao_src, ao_inv.inputs[1])
    ao_scl = _math(nodes, "MULTIPLY", b=ao_factor)
    links.new(ao_inv.outputs["Value"], ao_scl.inputs[0])

    try:
        mix_ao = nodes.new(type="ShaderNodeMix")
    except Exception:
        mix_ao = nodes.new(type="ShaderNodeMixRGB")
    dark = (0.04, 0.04, 0.05, 1.0)
    dummy = (0.5, 0.5, 0.5, 1.0)
    fac_ao, ao_out = _set_mix_rgba(mix_ao, 0.0, dummy, dark)
    if mix_ao.bl_idname == "ShaderNodeMix":
        links.new(lit_out, mix_ao.inputs[6])
    else:
        links.new(lit_out, mix_ao.inputs["Color1"])
    links.new(ao_scl.outputs["Value"], fac_ao)

    layer = nodes.new(type="ShaderNodeLayerWeight")
    if "Normal" in layer.inputs:
        links.new(n_out, layer.inputs["Normal"])
    if "Blend" in layer.inputs:
        layer.inputs["Blend"].default_value = form_blend
    try:
        mix_rim = nodes.new(type="ShaderNodeMix")
    except Exception:
        mix_rim = nodes.new(type="ShaderNodeMixRGB")
    fac_rim, rim_out = _set_mix_rgba(mix_rim, 0.0, dummy, rim_color)
    if mix_rim.bl_idname == "ShaderNodeMix":
        links.new(ao_out, mix_rim.inputs[6])
    else:
        links.new(ao_out, mix_rim.inputs["Color1"])
    rim_scale = _math(nodes, "MULTIPLY", b=rim_strength)
    fresnel = layer.outputs["Fresnel"] if "Fresnel" in layer.outputs else layer.outputs[0]
    links.new(fresnel, rim_scale.inputs[0])
    links.new(rim_scale.outputs["Value"], fac_rim)

    emit = nodes.new(type="ShaderNodeEmission")
    emit.inputs["Strength"].default_value = strength
    links.new(rim_out, emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_source_wireframe_material(name, render_cfg):
    """Face fill + triangle wireframe, with mild facing shading for volume."""
    return make_source_fill_material(name, render_cfg, with_wireframe=True)


def make_source_readable_material(name, render_cfg):
    """
    Source-lane material.

    Modes:
      wireframe — original: bright fill + triangle edges (+ facing shade)
      shaded    — clay volume (key light + AO + rim), no triangle edges
      emission  — uniform unlit color
      solid     — Principled BSDF
    """
    color = _parse_rgba(
        render_cfg.get("source_material_color"),
        (0.86, 0.86, 0.90, 1.0),
    )
    mode = source_material_mode(render_cfg)
    if mode == "wireframe":
        return make_source_fill_material(name, render_cfg, with_wireframe=True)
    if mode == "shaded":
        return make_source_volume_material(name, render_cfg)
    if mode == "solid":
        mat = make_material(name, color)
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes else None
        if bsdf is not None:
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.45
            for key in ("Specular IOR Level", "Specular"):
                if key in bsdf.inputs:
                    try:
                        bsdf.inputs[key].default_value = 0.15
                    except Exception:
                        pass
                    break
        return mat
    strength = float(render_cfg.get("source_emission_strength", 1.25))
    return make_emission_material(name, color, strength=strength)


def replace_object_materials(obj, material):
    mesh = obj.data
    mesh.materials.clear()
    mesh.materials.append(material)


def apply_source_flat_shading(obj):
    """Flat faces make triangle facets readable together with the wireframe."""
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = False
    if hasattr(mesh, "update"):
        mesh.update()


def apply_source_smooth_shading(obj):
    """Hide triangle facets when Source is drawn without a wire overlay."""
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    if hasattr(mesh, "update"):
        mesh.update()


def apply_retarget_preview_shading(obj):
    """
    LBS overwrites FBX vertex positions; leftover custom/auto-smooth normals
    then read as hard triangle facets under any directional lighting.
    """
    mesh = obj.data
    for mod in list(obj.modifiers):
        name_l = str(mod.name).lower()
        if mod.type in {"WEIGHTED_NORMAL", "NORMAL_EDIT"} or "smooth by angle" in name_l:
            try:
                obj.modifiers.remove(mod)
            except Exception:
                pass
    try:
        if getattr(mesh, "has_custom_normals", False):
            mesh.free_normals_split()
    except Exception:
        pass
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = False
    for poly in mesh.polygons:
        poly.use_smooth = True
    if hasattr(mesh, "calc_normals_split"):
        try:
            mesh.calc_normals_split()
        except Exception:
            pass
    if hasattr(mesh, "update"):
        mesh.update()


def winding_corrected_faces(faces, axis_matrix):
    """
    ``y_up_to_z_up`` has det -1, which reverses triangle winding.

    Without this, outward faces become backfaces: EEVEE culls them (see-through
    hind legs) and Lambert lighting reads as coming from below.
    """
    faces = np.asarray(faces, dtype=np.int32)
    det = float(np.linalg.det(np.asarray(axis_matrix, dtype=np.float64)))
    if det >= 0.0 or faces.ndim != 2 or faces.shape[1] < 3:
        return faces
    return np.ascontiguousarray(faces[:, ::-1])


def ensure_outward_normals(obj):
    mesh = obj.data
    try:
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(mesh)
        if bm.faces:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
    except Exception as exc:
        print(f"[render-arp-seq][warn] recalc_face_normals failed: {exc}")
    if hasattr(mesh, "calc_normals_split"):
        try:
            mesh.calc_normals_split()
        except Exception:
            pass
    if hasattr(mesh, "update"):
        mesh.update()


def correct_source_mesh_winding(obj, axis_matrix, *, faces_already_reversed: bool):
    det = float(np.linalg.det(np.asarray(axis_matrix, dtype=np.float64)))
    if det < 0.0 and not faces_already_reversed:
        try:
            obj.data.flip_normals()
        except Exception as exc:
            print(f"[render-arp-seq][warn] flip_normals failed: {exc}")
    ensure_outward_normals(obj)


def resolve_action_labels(manifest: dict, frame_count: int) -> list[str]:
    segments = manifest.get("action_segments")
    if not segments:
        clips = manifest.get("source_clips") or []
        if clips:
            segments = build_action_label_segments(
                clips,
                pause_frames=int(manifest.get("pause_frames", 0)),
                pause_mode=str(manifest.get("pause_mode", "hold_last")),
            )
    if not segments:
        return [""] * max(int(frame_count), 0)
    return action_labels_per_frame(segments, frame_count)


def add_action_name_hud(cam, scene, render_cfg, initial_text=""):
    """Camera-parented text anchored near the top-left of the frame."""
    color = _parse_rgba(render_cfg.get("action_label_color"), (0.08, 0.08, 0.08, 1.0))
    mat = make_emission_material("action_hud_mat", color)

    curve = bpy.data.curves.new(name="action_hud_curve", type="FONT")
    curve.body = str(initial_text)
    curve.align_x = "LEFT"
    curve.align_y = "TOP"
    curve.size = float(render_cfg.get("action_label_size", 0.06))
    obj = bpy.data.objects.new("action_hud", curve)
    scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    obj.parent = cam

    # Identity rotation: Blender text is readable from the camera when parented
    # (double-sided curve). LEFT/TOP keep the string in the top-left corner.
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)

    try:
        corners = cam.data.view_frame(scene=scene)
        # Pick by camera-space axes (order differs across Blender versions).
        tl = min(corners, key=lambda v: (v.x, -v.y))  # leftmost, then topmost
        tr = min(corners, key=lambda v: (-v.x, -v.y))  # rightmost, then topmost
        bl = min(corners, key=lambda v: (v.x, v.y))  # leftmost, then bottommost
        frame_w = max((tr - tl).length, 1e-6)
        frame_h = max((tl - bl).length, 1e-6)
        right = (tr - tl).normalized()
        down = (bl - tl).normalized()
        margin_x = float(render_cfg.get("action_label_margin_x", 0.035)) * frame_w
        margin_y = float(render_cfg.get("action_label_margin_y", 0.04)) * frame_h
        loc = tl + right * margin_x + down * margin_y
        depth_scale = float(render_cfg.get("action_label_depth_scale", 1.0))
        obj.location = loc * depth_scale
        size_frac = float(render_cfg.get("action_label_size_frac", 0.04))
        curve.size = max(frame_h * size_frac, 1e-4)
    except Exception:
        aspect = float(scene.render.resolution_x) / max(float(scene.render.resolution_y), 1.0)
        depth = float(render_cfg.get("action_label_depth", 1.5))
        obj.location = (
            float(render_cfg.get("action_label_x", -0.55 * aspect)),
            float(render_cfg.get("action_label_y", 0.38)),
            -depth,
        )

    return obj


def clear_arp_sequence_handlers():
    clear_frame_handlers()
    for handler in list(bpy.app.handlers.frame_change_pre):
        if getattr(handler, "__name__", "") == "arp_sequence_frame_handler":
            bpy.app.handlers.frame_change_pre.remove(handler)


def make_sequence_frame_handler(
    animated,
    hud_obj,
    labels_by_frame,
    *,
    source_clips=None,
    source_frame_map=None,
):
    """
    ``animated``: continuous vertex streams (retarget lane).
    ``source_clips``: optional list of {object, vertices} per source clip; only the
    active clip (from ``source_frame_map``) is shown and updated each frame.
    """

    def arp_sequence_frame_handler(scene):
        frame = max(scene.frame_current - scene.frame_start, 0)
        for item in animated:
            verts = item["vertices"]
            idx = min(frame, len(verts) - 1)
            mesh = item["object"].data
            for vert, co in zip(mesh.vertices, verts[idx]):
                vert.co = co
            mesh.update()

        if source_clips and source_frame_map:
            map_idx = min(frame, len(source_frame_map) - 1)
            clip_idx, local_idx = source_frame_map[map_idx]
            for i, item in enumerate(source_clips):
                obj = item["object"]
                visible = i == clip_idx
                obj.hide_render = not visible
                obj.hide_viewport = not visible
                if not visible:
                    continue
                verts = item["vertices"]
                local = min(int(local_idx), len(verts) - 1)
                mesh = obj.data
                for vert, co in zip(mesh.vertices, verts[local]):
                    vert.co = co
                mesh.update()

        if hud_obj is not None and labels_by_frame:
            idx = min(frame, len(labels_by_frame) - 1)
            text = labels_by_frame[idx]
            if hud_obj.data.body != text:
                hud_obj.data.body = text

    arp_sequence_frame_handler.__name__ = "arp_sequence_frame_handler"
    return arp_sequence_frame_handler


def _source_mesh_load_cfg(manifest: dict, render_cfg: dict) -> dict:
    """Resolve Source-lane mesh load knobs (default: no horizontal bbox lock)."""
    source_cfg = manifest.get("source") or {}
    lock = render_cfg.get("source_mesh_lock_horizontal_translation")
    if lock is None:
        lock = source_cfg.get("mesh_lock_horizontal_translation", False)
    ref = render_cfg.get("source_mesh_lock_reference")
    if ref is None:
        ref = source_cfg.get("mesh_lock_reference", "first")
    orient = source_cfg.get("mesh_orientation_correction", "identity")
    return {
        "mesh_lock_horizontal_translation": bool(lock),
        "mesh_lock_reference": ref,
        "mesh_orientation_correction": orient,
    }


def _load_source_lane_clips(manifest: dict, render_cfg: dict):
    clips_meta = manifest.get("source_clips") or []
    if not clips_meta:
        raise RuntimeError("show_source=true but manifest.source_clips is empty.")
    load_cfg = _source_mesh_load_cfg(manifest, render_cfg)
    loaded = []
    for clip in clips_meta:
        mesh_path = clip.get("source_mesh_path")
        if not mesh_path:
            raise RuntimeError(
                f"Clip '{clip.get('clip_id')}' missing source_mesh_path in manifest."
            )
        verts, faces = load_arp_mesh_for_sequence(Path(mesh_path), load_cfg)
        loaded.append(
            {
                "clip_id": clip.get("clip_id"),
                "vertices_raw": verts.astype(np.float32),
                "faces": faces.astype(np.int32),
                "fbx_path": clip.get("source_fbx_path"),
                "frame_count": int(verts.shape[0]),
            }
        )
    frame_map = build_clip_local_frame_map(
        [c["frame_count"] for c in loaded],
        pause_frames=int(manifest.get("pause_frames", 0)),
        pause_mode=str(manifest.get("pause_mode", "hold_last")),
    )
    return loaded, frame_map


def render_sequence(manifest: dict, args):
    sequence_id = manifest["sequence_id"]
    case_dir = Path(manifest["npz_path"]).parent
    out_mp4 = case_dir / args.output_name
    npz = np.load(manifest["npz_path"])
    frame_count = int(npz["frame_count"])
    fps = int(npz["fps"])
    verts_raw = npz["arp_vertices"].astype(np.float32)
    faces = npz["target_faces"].astype(np.int32)

    render_cfg = manifest.get("render", {}) or {}
    camera_cfg = manifest.get("camera", {}) or {}
    floor_z = float(render_cfg.get("floor_z", 0.0))
    use_fbx_materials = bool(render_cfg.get("lbs_use_fbx_materials", False))
    show_label = bool(render_cfg.get("show_lane_label", False))
    show_action_label = bool(render_cfg.get("show_action_label", True))
    simple_preview = bool(render_cfg.get("simple_preview", False))
    show_source = bool(
        manifest.get("show_source", render_cfg.get("show_source", False))
    )

    clean_scene()
    clear_arp_sequence_handlers()

    if simple_preview:
        light_rgb = render_cfg.get("simple_preview_light_rgb")
        light_strength = render_cfg.get("simple_preview_light_strength")
        setup_simple_preview_world(
            bg_rgb=_parse_rgb(render_cfg.get("simple_preview_bg_rgb")),
            strength=float(render_cfg.get("simple_preview_world_strength", 1.15)),
            light_rgb=_parse_rgb(light_rgb, (0.58, 0.58, 0.60)),
            light_strength=float(
                1.45 if light_strength is None else light_strength
            ),
        )
        add_simple_preview_retarget_lights(render_cfg)
    else:
        if bool(render_cfg.get("add_floor", True)):
            add_floor(
                size=float(render_cfg.get("floor_size", 80.0)),
                checker_scale=float(render_cfg.get("floor_checker_scale", 24.0)),
                rotation_deg=float(render_cfg.get("floor_rotation_deg", 30.0)),
                z_height=floor_z,
            )
        add_legacy_lights(render_cfg)

    row_center = parse_vec3(render_cfg.get("layout_center_xyz"))
    if row_center is None:
        row_center = parse_vec3(render_cfg.get("camera_target_xyz")) or [0.0, 0.0, 0.0]
    right_vec = camera_right_vector(render_cfg, mathutils.Vector(row_center))
    axis_matrix = lbs_axis_matrix(render_cfg)

    num_lanes = 2 if show_source else 1
    lane_spacing = float(render_cfg.get("lane_spacing", 2.8 if show_source else 0.0))
    retarget_lane_index = 1 if show_source else 0
    source_lane_index = 0

    retarget_yaw = arp_yaw_degrees(render_cfg, "retarget")
    clips_meta = manifest.get("source_clips") or []
    clip_frame_counts = [int(c.get("frame_count", 0)) for c in clips_meta]
    pause_frames = int(manifest.get("pause_frames", 0))
    pause_mode = str(manifest.get("pause_mode", "hold_last"))
    if clip_frame_counts and sum(clip_frame_counts) > 0:
        # Per-clip floor/center (same policy as Source lane) for multi-clip height sync.
        verts = transform_lane_vertices_by_clips(
            verts_raw,
            clip_frame_counts,
            pause_frames=pause_frames,
            pause_mode=pause_mode,
            lane_index=retarget_lane_index,
            num_lanes=num_lanes,
            lane_spacing=lane_spacing,
            right_vec=right_vec,
            row_center=row_center,
            floor_z=floor_z,
            axis_matrix=axis_matrix,
            lane_yaw_deg=retarget_yaw,
        )
    else:
        verts = transform_lane_vertices(
            verts_raw,
            lane_index=retarget_lane_index,
            num_lanes=num_lanes,
            lane_spacing=lane_spacing,
            right_vec=right_vec,
            row_center=row_center,
            floor_z=floor_z,
            axis_matrix=axis_matrix,
            lane_yaw_deg=retarget_yaw,
        )
    retarget_scale = resolve_mesh_scale(render_cfg, "retarget")
    verts = scale_mesh_vertices_about_floor(verts, retarget_scale, floor_z)
    if abs(retarget_scale - 1.0) > 1e-6:
        print(f"[render-arp-seq] mesh_scale retarget={retarget_scale:g}")

    if use_fbx_materials:
        template = import_fbx_template_mesh(manifest["target_fbx_path"])
        obj = mesh_from_template_vertices(f"{sequence_id}_retarget", verts[0], template)
    else:
        material = make_material(f"{sequence_id}_retarget_mat", (0.20, 0.48, 0.95, 1.0))
        obj = mesh_from_vertices(f"{sequence_id}_retarget", verts[0], faces, material)

    if simple_preview:
        apply_retarget_preview_shading(obj)
        soften_object_materials(
            obj,
            roughness=float(render_cfg.get("retarget_preview_roughness", 0.9)),
            specular=float(render_cfg.get("retarget_preview_specular", 0.05)),
            emission_lift=float(render_cfg.get("retarget_emission_lift", 0.22)),
        )

    source_clip_objs = []
    source_frame_map = None
    if show_source:
        source_yaw = arp_yaw_degrees(render_cfg, "source")
        source_loaded, source_frame_map = _load_source_lane_clips(manifest, render_cfg)
        if len(source_frame_map) != frame_count:
            raise RuntimeError(
                f"Source timeline length {len(source_frame_map)} != "
                f"retarget frame_count {frame_count}"
            )
        fbx_templates = {}
        force_solid = bool(render_cfg.get("source_force_solid_material", True))
        for clip_i, clip in enumerate(source_loaded):
            clip_verts = transform_lane_vertices(
                clip["vertices_raw"],
                lane_index=source_lane_index,
                num_lanes=num_lanes,
                lane_spacing=lane_spacing,
                right_vec=right_vec,
                row_center=row_center,
                floor_z=floor_z,
                axis_matrix=axis_matrix,
                lane_yaw_deg=source_yaw,
            )
            source_scale = resolve_mesh_scale(render_cfg, "source")
            clip_verts = scale_mesh_vertices_about_floor(
                clip_verts, source_scale, floor_z
            )
            name = f"{sequence_id}_source_{clip_i}_{clip.get('clip_id', 'clip')}"
            src_obj = None
            src_faces = winding_corrected_faces(clip["faces"], axis_matrix)
            # Prefer solid/emission for Source readability (no texture + ambient
            # preview otherwise looks like a shadow blob).
            if not force_solid:
                fbx_path = clip.get("fbx_path")
                if use_fbx_materials and fbx_path:
                    if fbx_path not in fbx_templates:
                        try:
                            fbx_templates[fbx_path] = import_fbx_template_mesh(fbx_path)
                        except Exception as exc:
                            print(
                                f"[render-arp-seq][warn] source FBX template failed "
                                f"({fbx_path}): {exc}; falling back to solid material."
                            )
                            fbx_templates[fbx_path] = None
                    template = fbx_templates.get(fbx_path)
                    if template is not None and len(template.vertices) == len(clip_verts[0]):
                        src_obj = mesh_from_template_vertices(
                            name, clip_verts[0], template
                        )
            if src_obj is None:
                src_obj = mesh_from_vertices(
                    name,
                    clip_verts[0],
                    src_faces,
                    make_source_readable_material(f"{name}_mat", render_cfg),
                )
                faces_already_reversed = True
            else:
                replace_object_materials(
                    src_obj,
                    make_source_readable_material(f"{name}_mat", render_cfg),
                )
                # FBX template faces are independent of the numpy winding fix.
                faces_already_reversed = False
            correct_source_mesh_winding(
                src_obj,
                axis_matrix,
                faces_already_reversed=faces_already_reversed,
            )
            if source_material_mode(render_cfg) == "wireframe":
                apply_source_flat_shading(src_obj)
            else:
                apply_source_smooth_shading(src_obj)
            # Only first clip visible initially.
            src_obj.hide_render = clip_i != 0
            src_obj.hide_viewport = clip_i != 0
            source_clip_objs.append({"object": src_obj, "vertices": clip_verts})

    if show_label:
        label_specs = []
        max_top_z = float(verts.reshape(-1, 3).max(axis=0)[2])
        if show_source and source_clip_objs:
            for item in source_clip_objs:
                max_top_z = max(
                    max_top_z, float(item["vertices"].reshape(-1, 3).max(axis=0)[2])
                )
        label_z = resolve_lane_label_z(render_cfg, max_top_z, floor_z=floor_z)
        if show_source:
            src_offset = lane_layout_offset(source_lane_index, num_lanes, lane_spacing)
            src_center = np.asarray(row_center, dtype=np.float32) + np.asarray(
                right_vec, dtype=np.float32
            ) * src_offset
            label_specs.append(
                ("Source", (float(src_center[0]), float(src_center[1]), label_z))
            )
        tgt_offset = lane_layout_offset(retarget_lane_index, num_lanes, lane_spacing)
        tgt_center = np.asarray(row_center, dtype=np.float32) + np.asarray(
            right_vec, dtype=np.float32
        ) * tgt_offset
        title = str(manifest.get("label", "Retarget"))
        label_specs.append(
            (title, (float(tgt_center[0]), float(tgt_center[1]), label_z))
        )
        add_labels(
            label_specs,
            float(render_cfg.get("lane_label_size", 0.35)),
            tuple(render_cfg.get("lane_label_color", [1.0, 0.0, 0.0, 1.0])),
        )

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
    if simple_preview:
        disable_scene_shadows(bpy.context.scene)

    labels_by_frame = resolve_action_labels(manifest, frame_count)
    hud_obj = None
    if show_action_label and labels_by_frame:
        hud_obj = add_action_name_hud(
            cam,
            bpy.context.scene,
            render_cfg,
            initial_text=labels_by_frame[0],
        )

    bpy.context.scene.frame_set(1)
    clear_arp_sequence_handlers()
    bpy.app.handlers.frame_change_pre.append(
        make_sequence_frame_handler(
            [{"object": obj, "vertices": verts}],
            hud_obj,
            labels_by_frame,
            source_clips=source_clip_objs or None,
            source_frame_map=source_frame_map,
        )
    )

    if args.dry_run:
        print(
            f"[render-arp-seq][dry-run] {sequence_id}: {out_mp4} "
            f"(show_source={show_source})"
        )
    else:
        bpy.ops.render.render(animation=True)
        print(f"[render-arp-seq] done: {sequence_id} -> {out_mp4}")


def main():
    args = parse_args(blender_argv())
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    render_sequence(manifest, args)


if __name__ == "__main__":
    main()
