#!/usr/bin/env python3
"""Export source-character mesh caches via Blender depsgraph skinning.

Used for non-SMAL33 / bone-oriented armatures (e.g. sucaibao cat) where R2ET's
joint-ball LBS cannot reproduce the authored skin bind.

Writes Blender Z-up caches (same spaces as ``arp_export_mesh_blender.py``).
The Python wrapper converts them to ``lbs_y_up`` and applies size matching.

Run (usually via ``source_skin_mesh.export_source_mesh_cache`` / batch)::

  blender -b -P visualization/source_skin_mesh_blender.py -- \\
    --config /path/to/_source_blender_batch.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_blender_common import (  # noqa: E402
    blender_argv,
    clean_scene,
    get_action_frame_range,
    import_source_fbx_with_anim,
    load_cfg,
    reset_armature_object_transform,
)
from arp_export_mesh_blender import (  # noqa: E402
    sample_evaluated_mesh,
)


VALID_SPACES = {
    "blender_world_z_up",
    "mesh_local_zup",
    "armature_local_zup",
    "root_bone_local_zup",
    "world_root_h_locked_zup",
}


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Export source FBX evaluated mesh caches (depsgraph skin)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    return parser.parse_args(argv)


def pick_skinned_mesh(armature_obj):
    """Largest mesh deformed by ``armature_obj`` (Armature modifier or parent)."""
    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name == "Cube":
            continue
        uses_arm = False
        if obj.parent == armature_obj:
            uses_arm = True
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and mod.object == armature_obj:
                uses_arm = True
                break
        if uses_arm:
            candidates.append(obj)
    if not candidates:
        # Fallback: any mesh (single-character FBX).
        candidates = [
            obj for obj in bpy.data.objects if obj.type == "MESH" and obj.name != "Cube"
        ]
    if not candidates:
        raise RuntimeError(f"No mesh found for armature {armature_obj.name}")
    return max(candidates, key=lambda o: len(o.data.vertices))


def run_case(case_cfg, cfg):
    case_id = case_cfg["case_id"]
    source_cfg = cfg.get("source", {}) or {}
    fbx_path = Path(case_cfg["inp_fbx_path"])
    if not fbx_path.is_file():
        raise FileNotFoundError(f"[{case_id}] source FBX not found: {fbx_path}")

    out_path = Path(case_cfg["source_mesh_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_space = str(
        case_cfg.get("mesh_export_space")
        or source_cfg.get("mesh_export_space")
        or (cfg.get("arp", {}) or {}).get("mesh_export_space")
        or "world_root_h_locked_zup"
    ).lower()
    if export_space not in VALID_SPACES:
        print(
            f"[source-mesh-blender][{case_id}][warn] unknown mesh_export_space="
            f"'{export_space}', fallback to world_root_h_locked_zup"
        )
        export_space = "world_root_h_locked_zup"

    clean_scene()
    source_arm = import_source_fbx_with_anim(fbx_path)
    reset_armature_object_transform(source_arm)
    source_mesh = pick_skinned_mesh(source_arm)

    frame_start, frame_end = get_action_frame_range(source_arm)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    print(
        f"[source-mesh-blender][{case_id}] arm={source_arm.name} "
        f"mesh={source_mesh.name} V={len(source_mesh.data.vertices)} "
        f"frames={frame_start}-{frame_end} space={export_space}",
        flush=True,
    )

    verts, faces, root_bone_name = sample_evaluated_mesh(
        source_mesh,
        source_arm,
        frame_start,
        frame_end,
        export_space,
    )
    np.savez_compressed(
        out_path,
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        frame_start=np.int32(frame_start),
        frame_end=np.int32(frame_end),
        space=np.asarray(export_space),
        root_bone_name=np.asarray(root_bone_name),
        retarget_mode=np.asarray("source_blender"),
        inp_fbx_path=np.asarray(str(fbx_path)),
        source_backend=np.asarray("blender"),
    )
    span0 = verts[0].max(0) - verts[0].min(0)
    mid = len(verts) // 2
    spanm = verts[mid].max(0) - verts[mid].min(0)
    print(
        f"[source-mesh-blender][{case_id}] wrote {out_path} "
        f"T={len(verts)} span0={span0} span_mid={spanm} root={root_bone_name}",
        flush=True,
    )


def main():
    args = parse_args(blender_argv())
    cfg = load_cfg(Path(args.config))
    cases = list(cfg.get("cases") or [])
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [c for c in cases if c.get("case_id") in wanted]
    if not cases:
        raise SystemExit("No cases to export in source blender config.")

    for case_cfg in cases:
        run_case(case_cfg, cfg)
    print(f"[source-mesh-blender] done ({len(cases)} case(s))", flush=True)


if __name__ == "__main__":
    main()
