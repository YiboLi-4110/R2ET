#!/usr/bin/env python3
"""
Check whether FBX mesh vertex counts match LBS npz vertices.

This is a prerequisite for reusing FBX materials/textures on the LBS renderer.
If counts/order do not match, use the solid-color LBS renderer for correctness.
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
    parser = argparse.ArgumentParser(description="Check FBX/NPZ vertex compatibility.")
    parser.add_argument("--manifest_index", type=str, required=True)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh_vertex_counts(fbx_path):
    clean_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    return {
        "mesh_count": len(meshes),
        "mesh_vertices": [{"name": obj.name, "vertex_count": len(obj.data.vertices)} for obj in meshes],
        "total_vertices": int(sum(len(obj.data.vertices) for obj in meshes)),
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
        npz = np.load(manifest["npz_path"])
        source_npz_vertices = int(npz["source_vertices"].shape[1])
        target_npz_vertices = int(npz["ours_vertices"].shape[1])
        source_fbx = import_mesh_vertex_counts(manifest["source_fbx_path"])
        target_fbx = import_mesh_vertex_counts(manifest["target_fbx_path"])
        report = {
            "case_id": manifest["case_id"],
            "source_npz_vertices": source_npz_vertices,
            "target_npz_vertices": target_npz_vertices,
            "source_fbx": source_fbx,
            "target_fbx": target_fbx,
            "source_total_vertex_count_matches": source_fbx["total_vertices"] == source_npz_vertices,
            "target_total_vertex_count_matches": target_fbx["total_vertices"] == target_npz_vertices,
        }
        reports.append(report)
        print(
            f"[debug-vertex-compat] {manifest['case_id']}: "
            f"source {source_fbx['total_vertices']} vs {source_npz_vertices}, "
            f"target {target_fbx['total_vertices']} vs {target_npz_vertices}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[debug-vertex-compat] wrote: {out_path}")


if __name__ == "__main__":
    main()

