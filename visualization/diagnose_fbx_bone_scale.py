#!/usr/bin/env python3
"""Probe FBX armatures for per-bone Scale (the channel BVH export drops).

Must be launched with Blender:

  blender -b -P visualization/diagnose_fbx_bone_scale.py -- \\
      --fbx path/a.fbx path/b.fbx \\
      --json_out /tmp/fbx_bone_scale.json

  blender -b -P visualization/diagnose_fbx_bone_scale.py -- \\
      --dir datasets/Planet_Zoo_FBX-smal2/train_char/sand_cat_juvenile \\
      --max_files 12 \\
      --json_out /tmp/fbx_bone_scale.json

Reports three different "scale" sources:

  object_scale     armature object (unit conversion; not a bone channel)
  rest_pose_scale  pose-bone scale on frame 0 / rest
  animated_scale   pose-bone scale over sampled frames
  fcurve_scale     whether the action actually has Scale fcurves

CopyQuat only cares about *animated* per-bone scale that is not identity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import bpy
except ImportError as exc:
    raise SystemExit(
        "Run with Blender, for example:\n"
        "  blender -b -P visualization/diagnose_fbx_bone_scale.py -- --fbx a.fbx\n"
    ) from exc


SCALE_EPS = 1e-4
VARY_EPS = 1e-4


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbx", type=Path, nargs="*", default=[], help="Explicit FBX files.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory of FBX files.")
    parser.add_argument("--glob", type=str, default="*.fbx")
    parser.add_argument("--max_files", type=int, default=0, help="0 = all matched files.")
    parser.add_argument("--max_samples", type=int, default=32, help="Max frames sampled per clip.")
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument(
        "--mocap_y_up",
        action="store_true",
        help="Import with axis_forward=-Z, axis_up=Y (same as fbx2bvh --mocap_y_up).",
    )
    return parser.parse_args(argv)


def purge_scene():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for name in ("actions", "armatures", "meshes", "materials", "images", "textures"):
        coll = getattr(bpy.data, name, None)
        if coll is None:
            continue
        for datablock in list(coll):
            coll.remove(datablock, do_unlink=True)
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def import_fbx(path: Path, mocap_y_up: bool):
    kwargs = {"filepath": str(path), "use_anim": True}
    if mocap_y_up:
        kwargs["axis_forward"] = "-Z"
        kwargs["axis_up"] = "Y"
    bpy.ops.import_scene.fbx(**kwargs)


def find_armatures():
    return [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]


def scene_frame_range():
    scene = bpy.context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    if bpy.data.actions:
        for act in bpy.data.actions:
            if act.frame_range:
                start = min(start, int(math.floor(act.frame_range[0])))
                end = max(end, int(math.ceil(act.frame_range[1])))
    if end < start:
        end = start
    return start, end


def sample_frames(start: int, end: int, max_samples: int):
    n = end - start + 1
    if n <= 0:
        return [start]
    k = min(int(max_samples), n)
    if k <= 1:
        return [start]
    if k == n:
        return list(range(start, end + 1))
    return [start + round(i * (n - 1) / (k - 1)) for i in range(k)]


def vec3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def max_abs_dev_from_one(scales):
    worst = 0.0
    for s in scales:
        for c in s:
            worst = max(worst, abs(float(c) - 1.0))
    return worst


def scale_varies(series):
    if not series:
        return False
    ref = series[0]
    for s in series[1:]:
        for a, b in zip(s, ref):
            if abs(float(a) - float(b)) > VARY_EPS:
                return True
    return False


def inspect_fcurves():
    """Return bones whose action fcurves animate scale."""
    hits = []
    for act in bpy.data.actions:
        by_bone = {}
        for fc in act.fcurves:
            path = fc.data_path or ""
            if ".scale" not in path and not path.endswith("scale"):
                continue
            bone = None
            if 'pose.bones["' in path:
                bone = path.split('pose.bones["', 1)[1].split('"]', 1)[0]
            key = bone or path
            xs = [kp.co[1] for kp in fc.keyframe_points]
            if not xs:
                continue
            info = by_bone.setdefault(
                key,
                {
                    "bone": bone or path,
                    "action": act.name,
                    "min": min(xs),
                    "max": max(xs),
                    "n_keys": 0,
                },
            )
            info["min"] = min(info["min"], min(xs))
            info["max"] = max(info["max"], max(xs))
            info["n_keys"] += len(xs)
        for info in by_bone.values():
            info["dev_from_1"] = max(abs(info["min"] - 1.0), abs(info["max"] - 1.0))
            info["varies"] = abs(info["max"] - info["min"]) > VARY_EPS
            hits.append(info)
    hits.sort(key=lambda h: h["dev_from_1"], reverse=True)
    return hits


def inspect_one(path: Path, mocap_y_up: bool, max_samples: int):
    purge_scene()
    import_fbx(path, mocap_y_up=mocap_y_up)
    arms = find_armatures()
    if not arms:
        return {
            "fbx": str(path),
            "ok": False,
            "error": "no armature",
        }

    arm = arms[0]
    bpy.context.view_layer.objects.active = arm
    start, end = scene_frame_range()
    frames = sample_frames(start, end, max_samples)

    bone_names = [pb.name for pb in arm.pose.bones]
    per_bone = {name: [] for name in bone_names}
    object_scales = []

    scene = bpy.context.scene
    for f in frames:
        scene.frame_set(int(f))
        bpy.context.view_layer.update()
        object_scales.append(vec3(arm.scale))
        for pb in arm.pose.bones:
            per_bone[pb.name].append(vec3(pb.scale))

    bone_rows = []
    n_non_identity = 0
    n_animated = 0
    worst_dev = 0.0
    worst_bone = None
    for name in bone_names:
        series = per_bone[name]
        dev = max_abs_dev_from_one(series)
        varies = scale_varies(series)
        identity = dev <= SCALE_EPS
        if not identity:
            n_non_identity += 1
        if varies:
            n_animated += 1
        if dev >= worst_dev:
            worst_dev = dev
            worst_bone = name
        mins = [min(s[i] for s in series) for i in range(3)]
        maxs = [max(s[i] for s in series) for i in range(3)]
        if (not identity) or varies:
            bone_rows.append(
                {
                    "bone": name,
                    "min": mins,
                    "max": maxs,
                    "dev_from_1": dev,
                    "varies": varies,
                    "rest": series[0],
                }
            )

    bone_rows.sort(key=lambda r: r["dev_from_1"], reverse=True)
    fcurves = inspect_fcurves()
    obj_dev = max_abs_dev_from_one(object_scales)
    obj_varies = scale_varies(object_scales)

    return {
        "fbx": str(path),
        "ok": True,
        "armature": arm.name,
        "n_bones": len(bone_names),
        "frame_start": start,
        "frame_end": end,
        "n_samples": len(frames),
        "object_scale": {
            "rest": object_scales[0] if object_scales else None,
            "dev_from_1": obj_dev,
            "varies": obj_varies,
            "identity": obj_dev <= SCALE_EPS,
        },
        "pose_bone_scale": {
            "n_non_identity": n_non_identity,
            "n_animated": n_animated,
            "worst_dev_from_1": worst_dev,
            "worst_bone": worst_bone,
            "outliers": bone_rows[:12],
        },
        "fcurve_scale": {
            "n_curves": len(fcurves),
            "n_non_identity": sum(1 for h in fcurves if h["dev_from_1"] > SCALE_EPS),
            "n_animated": sum(1 for h in fcurves if h["varies"]),
            "outliers": fcurves[:12],
        },
        "verdict": verdict_one(n_non_identity, n_animated, obj_dev, obj_varies, fcurves),
    }


def verdict_one(n_non_identity, n_animated, obj_dev, obj_varies, fcurves):
    if n_animated > 0 or any(h["varies"] for h in fcurves):
        return "animated_bone_scale"
    if n_non_identity > 0:
        return "static_nonidentity_bone_scale"
    if obj_dev > SCALE_EPS and obj_varies:
        return "animated_object_scale"
    if obj_dev > SCALE_EPS:
        return "static_object_scale"
    return "identity"


def collect_files(args) -> list[Path]:
    files = []
    for p in args.fbx:
        files.append(Path(p).resolve())
    if args.dir is not None:
        files.extend(sorted(Path(args.dir).resolve().glob(args.glob)))
    uniq = []
    seen = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    if args.max_files and len(uniq) > args.max_files:
        # Spread across the list instead of taking only the first N.
        n = args.max_files
        uniq = [uniq[round(i * (len(uniq) - 1) / max(n - 1, 1))] for i in range(n)]
    return uniq


def summarize(rows):
    counts = {}
    for r in rows:
        key = r.get("verdict", "error" if not r.get("ok") else "unknown")
        counts[key] = counts.get(key, 0) + 1
    animated = [r for r in rows if r.get("verdict") == "animated_bone_scale"]
    static_bone = [r for r in rows if r.get("verdict") == "static_nonidentity_bone_scale"]
    return {
        "n_files": len(rows),
        "verdict_counts": counts,
        "any_animated_bone_scale": bool(animated),
        "any_static_nonidentity_bone_scale": bool(static_bone),
        "copyquat_scale_worth_implementing": bool(animated),
    }


def main():
    args = parse_args()
    files = collect_files(args)
    if not files:
        raise SystemExit("No FBX files given. Use --fbx and/or --dir.")

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path}", flush=True)
        if not path.exists():
            rows.append({"fbx": str(path), "ok": False, "error": "missing"})
            continue
        try:
            row = inspect_one(path, mocap_y_up=args.mocap_y_up, max_samples=args.max_samples)
        except Exception as exc:
            row = {"fbx": str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"    verdict={row.get('verdict') or row.get('error')}", flush=True)
        if row.get("ok"):
            ps = row["pose_bone_scale"]
            print(
                f"    bones={row['n_bones']} frames={row['frame_start']}-{row['frame_end']} "
                f"non_id={ps['n_non_identity']} animated={ps['n_animated']} "
                f"worst={ps['worst_bone']} dev={ps['worst_dev_from_1']:.6g} "
                f"obj={row['object_scale']['rest']}",
                flush=True,
            )

    out = {"summary": summarize(rows), "files": rows}
    text = json.dumps(out, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
        print(f"[wrote] {args.json_out}", flush=True)
    print(json.dumps(out["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
