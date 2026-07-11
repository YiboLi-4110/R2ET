#!/usr/bin/env python3
"""
Debug FBX<->BVH binding compatibility for four-way compare lanes.

Run:
  blender --background --python visualization/debug_fourway_binding_blender.py -- \
    --manifest_index visualization/videos/compare/fourway_manifest_index.json \
    --case_id attack_to_bomei3 \
    --out_json visualization/videos/compare/attack_to_bomei3/binding_debug.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


def blender_argv():
    import sys

    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Debug fourway FBX/BVH binding consistency.")
    parser.add_argument("--manifest_index", type=str, required=True)
    parser.add_argument("--case_id", type=str, default=None)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def new_objects(before_names):
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def pick_single_armature(objects):
    arms = [obj for obj in objects if obj.type == "ARMATURE"]
    if not arms:
        return None
    return arms[0]


def normalize_name(name: str) -> str:
    return name.split(":")[-1]


def bone_lengths(arm_obj):
    return {b.name: float(b.length) for b in arm_obj.data.bones}


def bone_direction(arm_obj, bone_name):
    bone = arm_obj.data.bones[bone_name]
    vec = bone.tail_local - bone.head_local
    if vec.length < 1e-8:
        return None
    vec.normalize()
    return (float(vec.x), float(vec.y), float(vec.z))


def vec_dot(left, right):
    if left is None or right is None:
        return None
    return float(sum(a * b for a, b in zip(left, right)))


def match_bones(src_lengths, dst_lengths):
    dst_exact = set(dst_lengths.keys())
    dst_by_norm = {}
    for name in dst_lengths:
        dst_by_norm.setdefault(normalize_name(name), []).append(name)

    mapping = {}
    missing = []
    ambiguous = []
    for src_name in src_lengths:
        if src_name in dst_exact:
            mapping[src_name] = src_name
            continue
        key = normalize_name(src_name)
        cands = dst_by_norm.get(key, [])
        if len(cands) == 1:
            mapping[src_name] = cands[0]
        elif len(cands) == 0:
            missing.append(src_name)
        else:
            ambiguous.append({"src": src_name, "candidates": cands})
    return mapping, missing, ambiguous


def summarize_lane(case_id, lane_name, fbx_path, bvh_path):
    clean_scene()

    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    imported_fbx = new_objects(before)
    fbx_arm = pick_single_armature(imported_fbx)
    if fbx_arm is None:
        return {
            "case_id": case_id,
            "lane": lane_name,
            "error": f"no armature in FBX: {fbx_path}",
        }

    before = set(o.name for o in bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=str(bvh_path))
    imported_bvh = new_objects(before)
    bvh_arm = pick_single_armature(imported_bvh)
    if bvh_arm is None:
        return {
            "case_id": case_id,
            "lane": lane_name,
            "error": f"no armature in BVH: {bvh_path}",
        }

    fbx_l = bone_lengths(fbx_arm)
    bvh_l = bone_lengths(bvh_arm)
    mapping, missing, ambiguous = match_bones(fbx_l, bvh_l)

    zero_len_bvh = [name for name, val in bvh_l.items() if abs(val) < 1e-8]
    ratio_stats = []
    ratio_by_bone = []
    direction_dots = []
    for src_name, dst_name in mapping.items():
        src_len = fbx_l[src_name]
        dst_len = bvh_l[dst_name]
        if abs(src_len) < 1e-8:
            continue
        ratio = float(dst_len / src_len)
        ratio_stats.append(ratio)
        fbx_dir = bone_direction(fbx_arm, src_name)
        bvh_dir = bone_direction(bvh_arm, dst_name)
        dir_dot = vec_dot(fbx_dir, bvh_dir)
        if dir_dot is not None:
            direction_dots.append(dir_dot)
        ratio_by_bone.append(
            {
                "fbx_bone": src_name,
                "bvh_bone": dst_name,
                "fbx_length": float(src_len),
                "bvh_length": float(dst_len),
                "length_ratio": ratio,
                "direction_dot": dir_dot,
            }
        )

    ratio_abs_deviation = [abs(r - 1.0) for r in ratio_stats]
    ratio_bad = [r for r in ratio_stats if (r < 0.7 or r > 1.3)]
    ratio_outliers = [
        item for item in ratio_by_bone if item["length_ratio"] < 0.7 or item["length_ratio"] > 1.3
    ]
    direction_outliers = [
        item
        for item in ratio_by_bone
        if item["direction_dot"] is not None and abs(item["direction_dot"]) < 0.7
    ]

    return {
        "case_id": case_id,
        "lane": lane_name,
        "fbx_path": str(fbx_path),
        "bvh_path": str(bvh_path),
        "fbx_armature": fbx_arm.name,
        "bvh_armature": bvh_arm.name,
        "fbx_bone_count": len(fbx_l),
        "bvh_bone_count": len(bvh_l),
        "matched_bone_count": len(mapping),
        "missing_bones_in_bvh": missing,
        "ambiguous_matches": ambiguous,
        "zero_length_bones_bvh": zero_len_bvh,
        "length_ratio_mean": (sum(ratio_stats) / len(ratio_stats)) if ratio_stats else None,
        "length_ratio_max_abs_dev": max(ratio_abs_deviation) if ratio_abs_deviation else None,
        "length_ratio_outlier_count_0p7_1p3": len(ratio_bad),
        "direction_dot_mean": (sum(direction_dots) / len(direction_dots)) if direction_dots else None,
        "direction_dot_min": min(direction_dots) if direction_dots else None,
        "direction_dot_outlier_count_abs_lt_0p7": len(direction_outliers),
        "length_ratio_outliers": ratio_outliers,
        "direction_dot_outliers": direction_outliers,
        "length_ratio_samples": ratio_stats[:20],
    }


def main():
    args = parse_args(blender_argv())
    manifest_index = Path(args.manifest_index)
    manifests = json.loads(manifest_index.read_text(encoding="utf-8"))
    if not manifests:
        raise SystemExit("manifest index is empty")

    if args.case_id is not None:
        manifests = [m for m in manifests if m.get("case_id") == args.case_id]
        if not manifests:
            raise SystemExit(f"case_id not found in manifest index: {args.case_id}")

    all_reports = []
    for m in manifests:
        case_id = m["case_id"]
        lanes = [
            ("source", m["source_fbx_path"], m["source_bvh_path"]),
            ("copyquat", m["target_fbx_path"], m["copyquat_bvh_path"]),
            ("arp", m["target_fbx_path"], m["arp_bvh_path"]),
            ("ours", m["target_fbx_path"], m["ours_bvh_path"]),
        ]
        case_reports = []
        for lane_name, fbx, bvh in lanes:
            report = summarize_lane(case_id, lane_name, fbx, bvh)
            case_reports.append(report)
            print(
                f"[debug-binding] {case_id}/{lane_name}: "
                f"matched={report.get('matched_bone_count')} "
                f"missing={len(report.get('missing_bones_in_bvh', []))} "
                f"zero={len(report.get('zero_length_bones_bvh', []))}"
            )
        all_reports.append({"case_id": case_id, "lanes": case_reports})

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[debug-binding] wrote: {out_path}")


if __name__ == "__main__":
    main()

