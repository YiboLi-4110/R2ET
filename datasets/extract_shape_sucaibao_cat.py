#!/usr/bin/env python3
"""Extract shape .npz from sucaibao-cat FBX (non-SMAL33 skeleton).

Same pipeline as extract_shape_smal33.py (rest mesh, LBS weights, joint_shape),
but uses the ~50-joint sucaibao hierarchy seen in
datasets/shepherd/cat_actions_sucaibao/train_char/*.bvh.
"""
import argparse
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def rm_prefix(name):
    if ":" in name:
        return name[name.index(":") + 1 :]
    return name


def rm_prefix_name(name):
    return name.split(":")[-1] if ":" in name else name


def get_width(vertices):
    box = np.zeros((2, 3))
    box[0, :] = vertices.min(axis=0)
    box[1, :] = vertices.max(axis=0)
    return box[1, :] - box[0, :]


AXIS_TRANSFORMS = {
    "none": np.eye(3, dtype=np.float64),
    "shepherd_y_negz_x": np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    "shepherd_y_z_x": np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
}


def axis_transform_matrix(name):
    key = "none" if name in (None, "", "none") else str(name)
    if key not in AXIS_TRANSFORMS:
        known = ", ".join(sorted(AXIS_TRANSFORMS))
        raise ValueError(f"Unknown axis transform '{name}'. Known: {known}")
    return AXIS_TRANSFORMS[key]


def apply_axis_transform_points(points, axis_transform="none"):
    matrix = axis_transform_matrix(axis_transform)
    return np.einsum("ij,...j->...i", matrix, points)


# Depth-first order matching Attack_Crouch_IP.bvh (ROOT + JOINT, including Helpers).
JOINT_NAME_SUCAIBAO_CAT = [
    "root_bone",
    "Spine_base",
    "spine_02",
    "spine_03",
    "spine_04",
    "spine_05",
    "neck",
    "head",
    "mouth",
    "tongue_01",
    "tongue_02",
    "tongue_03",
    "ear_1.L",
    "ear_2.L",
    "eyelid.L",
    "mustache.L",
    "ear_1.R",
    "ear_2.R",
    "eyelid.R",
    "mustache.R",
    "eye.L",
    "eye.R",
    "shoulder_blade.L",
    "hip_f.L",
    "leg_f.L",
    "foot_f.L",
    "claw_f.L",
    "shoulder_blade.R",
    "hip_f.R",
    "leg_f.R",
    "foot_f.R",
    "claw_f.R",
    "spine_01",
    "tail_01",
    "tail_02",
    "tail_03",
    "tail_04",
    "tail_05",
    "hip_b.L",
    "leg_b.L",
    "foot_b.L",
    "claw_b.L",
    "hip_b.R",
    "leg_b.R",
    "foot_b.R",
    "claw_b.R",
    "Helper_claw_b.L",
    "Helper_claw_b.R",
    "Helper_foot_f.L",
    "Helper_foot_f.R",
]

BODY_JOINT_NAMES = {
    "root_bone",
    "Spine_base",
    "spine_01",
    "spine_02",
    "spine_03",
    "spine_04",
    "spine_05",
    "neck",
    "head",
    "mouth",
}

LEG_NAME_TOKENS = (
    "shoulder_blade",
    "hip_f",
    "leg_f",
    "foot_f",
    "claw_f",
    "hip_b",
    "leg_b",
    "foot_b",
    "claw_b",
)


def is_leg_joint(name):
    if name.startswith("Helper_"):
        return False
    return any(token in name for token in LEG_NAME_TOKENS)


def parse_args():
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(
        description="Extract sucaibao-cat shape .npz files from rest-pose FBX meshes."
    )
    parser.add_argument("--fbx_root", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--one_npz_per_subdirectory", action="store_true")
    parser.add_argument("--all_fbx_per_subdirectory", action="store_true")
    parser.add_argument("--recursive_fbx_search", action="store_true", default=True)
    parser.add_argument(
        "--no_recursive_fbx_search",
        action="store_false",
        dest="recursive_fbx_search",
    )
    parser.add_argument("--mocap_y_up", action="store_true")
    parser.add_argument("--fbx_axis_forward", type=str, default="")
    parser.add_argument("--fbx_axis_up", type=str, default="")
    parser.add_argument("--axis_transform", default="none")
    parser.add_argument(
        "--dump_bones",
        action="store_true",
        help="Print armature bone names for the first job and exit.",
    )
    parser.add_argument(
        "--from_armature",
        action="store_true",
        help="Use all armature bones in DFS order instead of the canonical 50-joint list.",
    )
    parser.add_argument(
        "--skip_helpers",
        action="store_true",
        help="Drop Helper_* bones from the simplified skeleton (weights collapse to parent).",
    )
    return parser.parse_args(argv)


def resolve_fbx_import_axes(args):
    if args.fbx_axis_forward or args.fbx_axis_up:
        return args.fbx_axis_forward or "-Z", args.fbx_axis_up or "Y"
    if args.mocap_y_up:
        return "-Z", "Y"
    return None, None


def clear_scene():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    datablock_names = [
        "actions",
        "armatures",
        "meshes",
        "materials",
        "images",
        "textures",
        "cameras",
        "lights",
        "curves",
        "grease_pencils",
        "node_groups",
    ]
    for name in datablock_names:
        if not hasattr(bpy.data, name):
            continue
        collection = getattr(bpy.data, name)
        for datablock in list(collection):
            collection.remove(datablock, do_unlink=True)

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True,
            do_linked_ids=True,
            do_recursive=True,
        )
    except Exception:
        pass


def import_fbx(fbx_path, fbx_axes=None):
    kwargs = {"filepath": str(fbx_path), "use_anim": True}
    if fbx_axes is not None:
        axis_forward, axis_up = fbx_axes
        if axis_forward and axis_up:
            kwargs["axis_forward"] = axis_forward
            kwargs["axis_up"] = axis_up
    bpy.ops.import_scene.fbx(**kwargs)


def find_armature():
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found after FBX import.")


def dfs_bone_names(armature):
    roots = [b for b in armature.data.bones if b.parent is None]
    names = []

    def walk(bone):
        names.append(rm_prefix_name(bone.name))
        for child in bone.children:
            walk(child)

    for root in roots:
        walk(root)
    return names


def normalize_lookup(name):
    return rm_prefix_name(name).replace("_", ".").lower()


def build_bone_map(armature):
    bone_map = {}
    alias_map = {}
    for bone in armature.data.bones:
        raw = rm_prefix_name(bone.name)
        bone_map[raw] = bone
        alias_map[normalize_lookup(raw)] = raw
    return bone_map, alias_map


def resolve_joint_names(armature, canonical, from_armature, skip_helpers):
    bone_map, alias_map = build_bone_map(armature)
    if from_armature:
        names = dfs_bone_names(armature)
        if skip_helpers:
            names = [n for n in names if not n.startswith("Helper_")]
        return names, bone_map

    resolved = []
    missing = []
    for name in canonical:
        if skip_helpers and name.startswith("Helper_"):
            continue
        if name in bone_map:
            resolved.append(name)
            continue
        alias = alias_map.get(normalize_lookup(name))
        if alias is not None:
            resolved.append(alias)
        else:
            missing.append(name)
    if missing:
        present = ", ".join(sorted(bone_map))
        raise ValueError(
            f"Missing sucaibao joints in armature: {missing}\n"
            f"Armature bones: {present}"
        )
    return resolved, bone_map


def topology_from_armature(joint_names, bone_map):
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    topology = []
    for name in joint_names:
        bone = bone_map[name]
        parent = bone.parent
        while parent is not None and rm_prefix_name(parent.name) not in name_to_idx:
            parent = parent.parent
        if parent is None:
            topology.append(-1)
        else:
            topology.append(name_to_idx[rm_prefix_name(parent.name)])
    return topology


def build_simplified_joint_offsets(joint_names, topology, bone_map, rest_origin):
    offsets = np.zeros((len(joint_names), 3), dtype=np.float64)
    for idx, name in enumerate(joint_names):
        bone = bone_map[name]
        parent_idx = topology[idx]
        if parent_idx == -1:
            offsets[idx] = np.array(bone.head_local) - rest_origin
        else:
            parent_bone = bone_map[joint_names[parent_idx]]
            offsets[idx] = np.array(bone.head_local) - np.array(parent_bone.head_local)
    return offsets


def body_leg_indices(joint_names):
    body_ids = [i for i, n in enumerate(joint_names) if n in BODY_JOINT_NAMES]
    leg_ids = [i for i, n in enumerate(joint_names) if is_leg_joint(n)]
    return body_ids, leg_ids


def extract_data(
    fbx_path,
    subject_name,
    save_path,
    fbx_axes=None,
    axis_transform="none",
    from_armature=False,
    skip_helpers=False,
    dump_bones=False,
):
    clear_scene()
    import_fbx(fbx_path, fbx_axes=fbx_axes)
    source_arm = find_armature()
    source_arm.data.pose_position = "REST"

    if dump_bones:
        names = dfs_bone_names(source_arm)
        print(f"DUMP_BONES ({len(names)}) from {fbx_path}:")
        for i, name in enumerate(names):
            bone = source_arm.data.bones[name] if name in source_arm.data.bones else None
            parent = rm_prefix_name(bone.parent.name) if bone is not None and bone.parent else None
            print(f"  {i:02d} {name}  parent={parent}")
        clear_scene()
        return

    joint_names, bone_map = resolve_joint_names(
        source_arm,
        JOINT_NAME_SUCAIBAO_CAT,
        from_armature=from_armature,
        skip_helpers=skip_helpers,
    )
    topology = topology_from_armature(joint_names, bone_map)
    body_ids, leg_ids = body_leg_indices(joint_names)
    body_id_set = set(body_ids)
    leg_id_set = set(leg_ids)

    root_name = joint_names[0]
    rest_origin = np.array(bone_map[root_name].head_local, dtype=np.float64)
    rest_x, rest_y, rest_z = rest_origin.tolist()

    np_rest_verts = None
    np_rest_faces = None
    scene = bpy.context.scene
    mesh_count = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj.name == "Cube":
            continue
        mesh_count += 1
        bme_rest = bmesh.new()
        bme_rest.from_mesh(obj.data)
        bm_rest_faces_tri = bmesh.ops.triangulate(
            bme_rest,
            faces=bme_rest.faces,
            quad_method="BEAUTY",
            ngon_method="BEAUTY",
        )["faces"]
        rest_verts_lst = [
            (v.co.x - rest_x, v.co.y - rest_y, v.co.z - rest_z) for v in bme_rest.verts
        ]
        rest_faces_lst = [
            (f.verts[0].index, f.verts[1].index, f.verts[2].index) for f in bm_rest_faces_tri
        ]
        np_rest_verts = apply_axis_transform_points(
            np.array(rest_verts_lst), axis_transform
        )
        np_rest_faces = np.array(rest_faces_lst)
        bme_rest.free()

    if np_rest_verts is None:
        raise RuntimeError(f"No mesh found in {fbx_path}")
    if mesh_count > 1:
        print(f"WARN: {mesh_count} meshes in {fbx_path}; using the last one (same as SMAL33).")

    simplified_joint_offsets = apply_axis_transform_points(
        build_simplified_joint_offsets(joint_names, topology, bone_map, rest_origin),
        axis_transform,
    )
    root_orient_data = np.zeros((1, 3), dtype=np.single)

    verts = None
    vgrp_label = None
    np_skinning_weights = None
    for obj in scene.objects:
        if obj.type == "MESH" and obj.name != "Cube":
            verts = obj.data.vertices
            vgrps = obj.vertex_groups
            np_skinning_weights = np.zeros((len(verts), len(vgrps)))
            vgrp_label = list(vgrps.keys())
            for i, vert in enumerate(verts):
                for g in vert.groups:
                    np_skinning_weights[i, g.group] = g.weight
        if obj.type == "ARMATURE":
            source_arm = bpy.data.objects[obj.name]

    np_simplified_skinning_weights = np.zeros((len(verts), len(joint_names)))
    bone_lookup = {rm_prefix_name(b.name): b for b in source_arm.data.bones}
    for j, name in enumerate(vgrp_label):
        bone_key = rm_prefix_name(name)
        if bone_key not in bone_lookup:
            print(f"WARN: vertex group '{name}' has no matching bone, skip")
            continue
        bone = bone_lookup[bone_key]
        while bone.parent is not None and rm_prefix(bone.name) not in joint_names:
            bone = bone.parent
        mapped = rm_prefix(bone.name)
        if mapped not in joint_names:
            print(f"WARN: could not map vertex group '{name}' onto simplified joints")
            continue
        idx = joint_names.index(mapped)
        np_simplified_skinning_weights[:, idx] += np_skinning_weights[:, j]

    vertex_part = np.argmax(np_simplified_skinning_weights, axis=1)
    body_vid_lst = [i for i, p in enumerate(vertex_part) if p in body_id_set]
    arm_vid_lst = [i for i, p in enumerate(vertex_part) if p in leg_id_set]
    rest_body_vertices = np_rest_verts[body_vid_lst, :] if body_vid_lst else np.zeros((0, 3))
    rest_arm_vertices = np_rest_verts[arm_vid_lst, :] if arm_vid_lst else np.zeros((0, 3))
    body_width = get_width(rest_body_vertices) if len(rest_body_vertices) else np.zeros(3)
    full_width = get_width(np_rest_verts)

    shape_lst = []
    for i in range(len(joint_names)):
        joint_i = np.where(vertex_part == i)[0]
        if len(joint_i) == 0:
            shape_lst.append(np.array([0.0, 0.0, 0.0]))
        else:
            shape_lst.append(get_width(np_rest_verts[joint_i, :]))
    shape_lst_array = np.stack(shape_lst, axis=0)

    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{subject_name}.npz")
    np.savez(
        output_path,
        skinning_weights=np_simplified_skinning_weights.astype(np.single),
        joint_names=np.array(joint_names),
        root_orient=root_orient_data,
        rest_vertices=np_rest_verts.astype(np.single),
        rest_faces=np_rest_faces,
        skeleton=simplified_joint_offsets.astype(np.single),
        topology=np.array(topology, dtype=np.int32),
        subject=subject_name,
        vertex_part=vertex_part,
        rest_body_vertices=rest_body_vertices,
        rest_arm_vertices=rest_arm_vertices,
        body_width=body_width,
        full_width=full_width,
        joint_shape=shape_lst_array,
    )
    print(
        f"  joints={len(joint_names)} verts={len(np_rest_verts)} "
        f"body_vids={len(body_vid_lst)} leg_vids={len(arm_vid_lst)}"
    )
    clear_scene()


def list_fbx_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".fbx" and not path.name.startswith(".")
    )


def find_fbx_for_character(directory: Path, recursive: bool) -> Path | None:
    direct = list_fbx_files(directory)
    if direct:
        return direct[0]
    if not recursive:
        return None
    for subdir in sorted(
        path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        found = find_fbx_for_character(subdir, recursive=True)
        if found is not None:
            return found
    return None


def collect_all_fbx_under(directory: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(
            path
            for path in directory.rglob("*.fbx")
            if path.is_file() and not path.name.startswith(".")
        )
    return list_fbx_files(directory)


def collect_extraction_jobs(
    fbx_root: Path,
    one_npz_per_subdirectory: bool,
    all_fbx_per_subdirectory: bool,
    recursive_fbx_search: bool,
):
    if one_npz_per_subdirectory and all_fbx_per_subdirectory:
        raise ValueError(
            "Cannot use --one_npz_per_subdirectory together with --all_fbx_per_subdirectory."
        )

    jobs = []
    if one_npz_per_subdirectory or all_fbx_per_subdirectory:
        character_dirs = sorted(
            path
            for path in fbx_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        for character_dir in character_dirs:
            if all_fbx_per_subdirectory:
                fbx_paths = collect_all_fbx_under(character_dir, recursive_fbx_search)
                if not fbx_paths:
                    print(f"WARN: no .fbx found under {character_dir}, skipping")
                    continue
                for fbx_path in fbx_paths:
                    jobs.append(
                        {
                            "fbx_path": fbx_path,
                            "subject_name": fbx_path.stem,
                            "character_dir": character_dir,
                        }
                    )
            else:
                fbx_path = find_fbx_for_character(character_dir, recursive_fbx_search)
                if fbx_path is None:
                    print(f"WARN: no .fbx found under {character_dir}, skipping")
                    continue
                jobs.append(
                    {
                        "fbx_path": fbx_path,
                        "subject_name": character_dir.name,
                        "character_dir": character_dir,
                    }
                )
    else:
        for fbx_path in list_fbx_files(fbx_root):
            jobs.append(
                {
                    "fbx_path": fbx_path,
                    "subject_name": fbx_path.stem,
                    "character_dir": fbx_root,
                }
            )
    return jobs


if __name__ == "__main__":
    args = parse_args()
    fbx_root = args.fbx_root.resolve()
    save_path = args.save_path.resolve()
    if not fbx_root.exists():
        raise SystemExit(f"fbx_root does not exist: {fbx_root}")

    jobs = collect_extraction_jobs(
        fbx_root,
        args.one_npz_per_subdirectory,
        args.all_fbx_per_subdirectory,
        args.recursive_fbx_search,
    )
    if not jobs:
        raise SystemExit(f"No .fbx extraction jobs found under {fbx_root}")

    print(f"Found {len(jobs)} shape job(s)")
    fbx_axes = resolve_fbx_import_axes(args)
    if fbx_axes is not None:
        print(f"FBX import axes: forward={fbx_axes[0]!r}, up={fbx_axes[1]!r}")
    print(f"axis_transform={args.axis_transform}")

    if args.dump_bones:
        extract_data(
            jobs[0]["fbx_path"],
            jobs[0]["subject_name"],
            save_path,
            fbx_axes=fbx_axes,
            dump_bones=True,
        )
        raise SystemExit(0)

    for job in jobs:
        subject_name = job["subject_name"]
        fbx_path = job["fbx_path"]
        output_file = os.path.join(save_path, f"{subject_name}.npz")
        if os.path.exists(output_file) and not args.overwrite_existing:
            print(f"SKIP: {subject_name} (exists)")
            continue
        print(f"EXTRACT: {subject_name} <= {fbx_path}")
        extract_data(
            fbx_path,
            subject_name,
            save_path,
            fbx_axes=fbx_axes,
            axis_transform=args.axis_transform,
            from_armature=args.from_armature,
            skip_helpers=args.skip_helpers,
        )
        print(f"DONE: {subject_name}")