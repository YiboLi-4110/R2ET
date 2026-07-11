#!/usr/bin/env python3
"""
Check whether FBX assets are ready for textured LBS rendering.

What this script verifies per case:
1) Source/target FBX can be imported and contain mesh objects.
2) Chosen template mesh vertex count matches npz vertex count.
3) UV layers/material slots/image texture nodes exist.

Run:
  blender --background --python visualization/debug_lbs_texture_ready_blender.py -- \
    --manifest_index visualization/videos/compare/fourway_manifest_index.json \
    --case_id attack_to_bomei3 \
    --out_json visualization/videos/compare/attack_to_bomei3/lbs_texture_ready.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
import numpy as np


def blender_argv():
    import sys

    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Check texture readiness for LBS rendering.")
    parser.add_argument("--manifest_index", type=str, required=True)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def imported_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def analyze_fbx_mesh(fbx_path: str):
    before = set(obj.name for obj in bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    imported = imported_objects(before)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        return {"error": f"no mesh in {fbx_path}"}

    mesh_obj = max(meshes, key=lambda o: len(o.data.vertices))
    mesh = mesh_obj.data
    uv_count = len(mesh.uv_layers)
    material_count = len(mesh.materials)
    texture_images = []
    for mat in mesh.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                texture_images.append(node.image.filepath)

    result = {
        "template_mesh_name": mesh_obj.name,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.polygons)),
        "uv_layer_count": int(uv_count),
        "material_count": int(material_count),
        "image_texture_count": int(len(texture_images)),
        "image_textures": texture_images,
    }
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    return result


def analyze_case(manifest):
    npz = np.load(manifest["npz_path"])
    source_npz_vertices = int(npz["source_vertices"].shape[1])
    target_npz_vertices = int(npz["ours_vertices"].shape[1])

    clean_scene()
    source_fbx = analyze_fbx_mesh(manifest["source_fbx_path"])
    clean_scene()
    target_fbx = analyze_fbx_mesh(manifest["target_fbx_path"])

    source_ok = (
        "error" not in source_fbx
        and source_fbx["vertex_count"] == source_npz_vertices
    )
    target_ok = (
        "error" not in target_fbx
        and target_fbx["vertex_count"] == target_npz_vertices
    )
    # We allow source to have no texture (as expected in your dataset), but target
    # should have materials + image textures if textured output is desired.
    target_has_texture = (
        "error" not in target_fbx
        and target_fbx["uv_layer_count"] > 0
        and target_fbx["material_count"] > 0
        and target_fbx["image_texture_count"] > 0
    )

    return {
        "case_id": manifest["case_id"],
        "source_npz_vertices": source_npz_vertices,
        "target_npz_vertices": target_npz_vertices,
        "source_fbx": source_fbx,
        "target_fbx": target_fbx,
        "source_vertex_count_match": bool(source_ok),
        "target_vertex_count_match": bool(target_ok),
        "target_has_texture_assets": bool(target_has_texture),
        "lbs_texture_ready": bool(source_ok and target_ok),
    }


def main():
    args = parse_args(blender_argv())
    manifests = json.loads(Path(args.manifest_index).read_text(encoding="utf-8"))
    if args.case_id is not None:
        manifests = [m for m in manifests if m.get("case_id") == args.case_id]
        if not manifests:
            raise SystemExit(f"case_id not found: {args.case_id}")

    reports = []
    for manifest in manifests:
        report = analyze_case(manifest)
        reports.append(report)
        print(
            f"[debug-lbs-texture] {report['case_id']}: "
            f"source_match={report['source_vertex_count_match']} "
            f"target_match={report['target_vertex_count_match']} "
            f"target_textures={report['target_has_texture_assets']}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[debug-lbs-texture] wrote: {out_path}")


if __name__ == "__main__":
    main()

