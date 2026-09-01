#!/usr/bin/env python3
"""Blender helper: evaluate sucaibao FBX armature skin as ground-truth mesh cache.

Run::

  $BLENDER -b -P datasets/diagnose_sucaibao_source_lbs_blender.py -- \\
    --fbx datasets/shepherd/cat_actions_sucaibao/train_char/JumpFw/JumpFw_IP.fbx \\
    --out /tmp/sucaibao_blender_skin.npz \\
    --mocap_y_up

Then compare with::

  python datasets/diagnose_sucaibao_source_lbs.py --blender_skin /tmp/sucaibao_blender_skin.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mocap_y_up", action="store_true")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def main():
    args = parse_args()
    clear_scene()
    kwargs = {"filepath": str(args.fbx.resolve()), "use_anim": True}
    if args.mocap_y_up:
        kwargs["axis_forward"] = "-Z"
        kwargs["axis_up"] = "Y"
    bpy.ops.import_scene.fbx(**kwargs)

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Cube"]
    if not arms or not meshes:
        raise SystemExit(f"Need armature+mesh in {args.fbx}")
    arm = arms[0]
    mesh_obj = max(meshes, key=lambda o: len(o.data.vertices))
    print(f"arm={arm.name} mesh={mesh_obj.name} verts={len(mesh_obj.data.vertices)}")

    # Rest bone orientations: if not near-identity (joint-ball), Python LBS will 拉皮.
    bone_rows = []
    for bone in arm.data.bones:
        ml = np.array(bone.matrix_local, dtype=np.float64)
        R = ml[:3, :3]
        # Deviation from pure translation rest (identity rotation in bone local)
        # Use Frobenius distance to nearest... simply ||R - I|| / ||I||
        eye = np.eye(3)
        rot_err = float(np.linalg.norm(R - eye))
        head = np.array(bone.head_local, dtype=np.float64)
        tail = np.array(bone.tail_local, dtype=np.float64)
        bone_rows.append(
            {
                "name": bone.name,
                "rot_err_vs_I": rot_err,
                "length": float(np.linalg.norm(tail - head)),
                "head_local": head.tolist(),
            }
        )
    bone_rows.sort(key=lambda r: -r["rot_err_vs_I"])
    print("\n== rest bone orientation (top 10 ||R-I||) ==")
    for row in bone_rows[:10]:
        print(
            f"  {row['name']:28s} ||R-I||={row['rot_err_vs_I']:.3f} "
            f"len={row['length']:.4f}"
        )
    n_oriented = sum(1 for r in bone_rows if r["rot_err_vs_I"] > 0.2)
    print(
        f"  bones with ||R-I||>0.2: {n_oriented}/{len(bone_rows)} "
        "(>0 means joint-ball LBS bind is wrong for this FBX)"
    )

    scene = bpy.context.scene
    if arm.animation_data and arm.animation_data.action:
        f0, f1 = [int(x) for x in arm.animation_data.action.frame_range]
    else:
        f0, f1 = int(scene.frame_start), int(scene.frame_end)
    if args.max_frames > 0:
        f1 = min(f1, f0 + args.max_frames - 1)

    frames = []
    faces = None
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh_obj.evaluated_get(dg)
        me = eval_obj.to_mesh()
        me.transform(eval_obj.matrix_world)
        me.calc_loop_triangles()
        verts = np.array([v.co[:] for v in me.vertices], dtype=np.float32)
        if faces is None:
            faces = np.array(
                [t.vertices[:] for t in me.loop_triangles], dtype=np.int32
            )
        frames.append(verts)
        eval_obj.to_mesh_clear()
        print(f"  frame {f}: aabb={verts.max(0)-verts.min(0)}")

    out = np.stack(frames, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vertices=out,
        faces=faces,
        frame_start=np.int32(f0),
        frame_end=np.int32(f1),
        space=np.asarray("blender_world_z_up"),
        source=np.asarray("depsgraph"),
        fbx=np.asarray(str(args.fbx)),
        bone_rot_err_vs_I=np.asarray(
            [r["rot_err_vs_I"] for r in bone_rows], dtype=np.float64
        ),
        bone_names=np.asarray([r["name"] for r in bone_rows]),
        n_oriented_bones=np.int32(n_oriented),
    )
    print(f"Wrote {args.out} shape={out.shape}")


if __name__ == "__main__":
    main()
