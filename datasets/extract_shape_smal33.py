import argparse
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

sys.path.append(".")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def rm_prefix(str):
    if ':' in str:
        return str[str.index(':') + 1 :]
    else:
        return str


def get_width(vertices):
    # vertices: (num_joint, 3)
    box = np.zeros((2, 3))
    box[0, :] = vertices.min(axis=0)
    box[1, :] = vertices.max(axis=0)
    width = box[1, :] - box[0, :]
    return width


AXIS_TRANSFORMS = {
    "none": np.eye(3, dtype=np.float64),
    # Shepherd bind-pose diagnostic result:
    #   new_x = old_y, new_y = -old_z, new_z = old_x
    "shepherd_y_negz_x": np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
    # A/B candidate when shepherd_y_negz_x appears vertically flipped:
    #   new_x = old_y, new_y = old_z, new_z = old_x
    "shepherd_y_z_x": np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
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


'''33 joints SMAL quadruped'''
JOINT_NAME_SMAL_33 = [
    "Root",
    "Spine1", "Spine2", "Spine3", "Spine4", "Spine5", "Spine6",
    "LeftScapula", "LeftUpperArm", "LeftForeLeg", "LeftFrontPaw",
    "RightScapula", "RightUpperArm", "RightForeLeg", "RightFrontPaw",
    "Neck", "Head", "Jaw",
    "LeftThigh", "LeftShin", "LeftHock", "LeftHindPaw",
    "RightThigh", "RightShin", "RightHock", "RightHindPaw",
    "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6", "Tail7",
]
body_joint_lst = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17]  # Spine1-6 + Neck + Head + Jaw
left_front_leg_lst = [7, 8, 9, 10]    # LeftScapula → LeftFrontPaw
right_front_leg_lst = [11, 12, 13, 14]  # RightScapula → RightFrontPaw
left_hind_leg_lst = [18, 19, 20, 21]   # LeftThigh → LeftHindPaw
right_hind_leg_lst = [22, 23, 24, 25]  # RightThigh → RightHindPaw
tail_lst = [26, 27, 28, 29, 30, 31, 32]  # Tail1-7
legs_joint_lst = left_front_leg_lst + right_front_leg_lst + left_hind_leg_lst + right_hind_leg_lst

SMAL_TOPOLOGY = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 16, 0, 18, 19, 20, 0, 22, 23, 24, 0, 26, 27, 28, 29, 30, 31]


def parse_args():
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(
        description="Extract SMAL 33 shape .npz files from rest-pose FBX meshes."
    )
    parser.add_argument(
        "--fbx_root",
        type=Path,
        default=Path("./Planet_Zoo_FBX-smal2/train_char"),
        help="Directory containing one rest-pose FBX per subject.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("./Planet_Zoo_FBX-smal2/train_shape"),
        help="Directory to save extracted .npz files.",
    )
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="If set, recompute .npz files even when they already exist.",
    )
    parser.add_argument(
        "--one_npz_per_subdirectory",
        action="store_true",
        help=(
            "Treat each immediate subdirectory of --fbx_root as one character. "
            "Pick one .fbx inside (see --recursive_fbx_search) and write "
            "<subdir_name>.npz to --save_path. Useful when motion FBX files "
            "for the same character live under per-pet folders."
        ),
    )
    parser.add_argument(
        "--all_fbx_per_subdirectory",
        action="store_true",
        help=(
            "Treat each immediate subdirectory of --fbx_root as a group and "
            "export every .fbx inside to --save_path/<fbx_stem>.npz. "
            "Useful for batch2_dogs where each subject has its own FBX."
        ),
    )
    parser.add_argument(
        "--recursive_fbx_search",
        action="store_true",
        default=True,
        help=(
            "With --one_npz_per_subdirectory or --all_fbx_per_subdirectory, "
            "also search nested folders under each character subdirectory "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--no_recursive_fbx_search",
        action="store_false",
        dest="recursive_fbx_search",
        help="Only look for .fbx files directly under each character subdirectory.",
    )
    parser.add_argument(
        "--mocap_y_up",
        action="store_true",
        help=(
            "Import FBX with mocap-style axes (-Z forward, Y up). "
            "Use the same setting as fbx2bvh_smal33.py for external assets."
        ),
    )
    parser.add_argument(
        "--fbx_axis_forward",
        type=str,
        default="",
        help="Optional FBX import axis_forward override (e.g. -Z).",
    )
    parser.add_argument(
        "--fbx_axis_up",
        type=str,
        default="",
        help="Optional FBX import axis_up override (e.g. Y).",
    )
    parser.add_argument(
        "--axis_transform",
        default="none",
        help="Optional post-import coordinate transform. Use shepherd_y_negz_x for smal@shepherd.",
    )
    return parser.parse_args(argv)


def resolve_fbx_import_axes(args):
    if args.fbx_axis_forward or args.fbx_axis_up:
        return args.fbx_axis_forward or "-Z", args.fbx_axis_up or "Y"
    if args.mocap_y_up:
        return "-Z", "Y"
    return None, None


def rm_prefix_name(name):
    return name.split(":")[-1] if ":" in name else name


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


def build_simplified_joint_offsets(source_arm, rest_origin):
    bone_map = {}
    for bone in source_arm.data.bones:
        bone_map[rm_prefix_name(bone.name)] = bone

    missing = [joint_name for joint_name in JOINT_NAME_SMAL_33 if joint_name not in bone_map]
    if missing:
        raise ValueError(f"Missing SMAL joints in armature: {missing}")

    simplified_joint_offsets = np.zeros((len(JOINT_NAME_SMAL_33), 3), dtype=np.float64)
    for idx, joint_name in enumerate(JOINT_NAME_SMAL_33):
        bone = bone_map[joint_name]
        parent_idx = SMAL_TOPOLOGY[idx]
        if parent_idx == -1:
            simplified_joint_offsets[idx] = np.array(bone.head_local) - rest_origin
        else:
            parent_name = JOINT_NAME_SMAL_33[parent_idx]
            parent_bone = bone_map[parent_name]
            simplified_joint_offsets[idx] = np.array(bone.head_local) - np.array(parent_bone.head_local)

    return simplified_joint_offsets


def extract_data(fbx_path, subject_name, save_path, fbx_axes=None, axis_transform="none"):
    clear_scene()
    kwargs = {"filepath": str(fbx_path), "use_anim": True}
    if fbx_axes is not None:
        axis_forward, axis_up = fbx_axes
        if axis_forward and axis_up:
            kwargs["axis_forward"] = axis_forward
            kwargs["axis_up"] = axis_up
    bpy.ops.import_scene.fbx(**kwargs)
    context = bpy.context
    scene = context.scene

    # find armature
    source_arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')

    # obtain mesh under rest pose
    source_arm.data.pose_position = 'REST'

    rest_x, rest_y, rest_z = source_arm.data.bones[0].head_local
    rest_origin = np.array([rest_x, rest_y, rest_z], dtype=np.float64)
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            bme_rest = bmesh.new()
            bme_rest.from_mesh(obj.data)

            bm_rest_verts = bme_rest.verts
            bm_rest_faces_ori = bme_rest.faces
            bm_rest_faces_tri = bmesh.ops.triangulate(
                bme_rest,
                faces=bm_rest_faces_ori,
                quad_method='BEAUTY',
                ngon_method='BEAUTY',
            )['faces']

            rest_verts_lst = []
            for v in bm_rest_verts:
                rest_verts_lst.append(
                    (v.co.x - rest_x, v.co.y - rest_y, v.co.z - rest_z)
                )

            rest_faces_lst = []
            for face in bm_rest_faces_tri:
                f_verts = face.verts
                rest_faces_lst.append(
                    (f_verts[0].index, f_verts[1].index, f_verts[2].index)
                )

            np_rest_verts = np.array(rest_verts_lst)
            np_rest_verts = apply_axis_transform_points(np_rest_verts, axis_transform)
            np_rest_faces = np.array(rest_faces_lst)

    simplified_joint_names = list(JOINT_NAME_SMAL_33)
    simplified_joint_offsets = build_simplified_joint_offsets(source_arm, rest_origin)
    simplified_joint_offsets = apply_axis_transform_points(
        simplified_joint_offsets,
        axis_transform,
    )
    root_orient_data = np.zeros((1, 3), dtype=np.single)

    # ====== extract data block ======
    # extract skinning weight and simplify it
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            verts = obj.data.vertices
            vgrps = obj.vertex_groups  # vertex groups correspond to the joints.

            np_skinning_weights = np.zeros((len(verts), len(vgrps)))
            mask = np.zeros(np_skinning_weights.shape, dtype=np.int32)
            vgrp_label = vgrps.keys()

            for i, vert in enumerate(verts):
                for g in vert.groups:
                    j = g.group
                    np_skinning_weights[i, j] = g.weight
                    mask[i, j] = 1

        if obj.type == 'ARMATURE':
            source_arm = bpy.data.objects[obj.name]

    np_simplified_skinning_weights = np.zeros(
        (len(verts), len(simplified_joint_names))
    )
    for j, name in enumerate(vgrp_label):
        bone = source_arm.data.bones[name]
        while (
            bone.parent is not None
            and rm_prefix(bone.name) not in simplified_joint_names
        ):
            bone = bone.parent

        idx = simplified_joint_names.index(rm_prefix(bone.name))
        np_simplified_skinning_weights[:, idx] += np_skinning_weights[:, j]

    vertex_part = np.argmax(np_simplified_skinning_weights, axis=1)
    num_face = np_rest_faces.shape[0]
    face_part = []
    for i in range(num_face):
        face_part.append(vertex_part[np_rest_faces[i][0]])

    face_part = np.array(face_part)
    body_vid_lst = []
    arm_vid_lst = []
    for i in range(vertex_part.shape[0]):
        if vertex_part[i] in body_joint_lst:
            body_vid_lst.append(i)
        if vertex_part[i] in legs_joint_lst:
            arm_vid_lst.append(i)

    rest_body_vertices = np_rest_verts[body_vid_lst, :]
    rest_arm_vertices = np_rest_verts[arm_vid_lst, :]

    body_width = get_width(rest_body_vertices)
    full_width = get_width(np_rest_verts)

    # detail shape
    joint_shape_lst = []
    for i in range(33):
        joint_i = []
        for j in range(vertex_part.shape[0]):
            if vertex_part[j] == i:
                joint_i.append(j)
        joint_shape_lst.append(joint_i)

    shape_lst = []
    for joint_i in joint_shape_lst:
        if len(joint_i) == 0:
            shape_lst.append(np.array([0, 0, 0]))
        else:
            joint_i_vertices = np_rest_verts[joint_i, :]
            joint_i_width = get_width(joint_i_vertices)
            shape_lst.append(joint_i_width)

    shape_lst_array = np.stack(shape_lst, axis=0)

    skinning_weights_data = np_simplified_skinning_weights.astype(np.single)
    joint_names_data = simplified_joint_names
    rest_vertices_data = np_rest_verts.astype(np.single)
    rest_faces_data = np_rest_faces
    skeleton_data = simplified_joint_offsets.astype(np.single)

    output_root = save_path
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, '%s.npz' % (subject_name))

    np.savez(
        output_path,
        skinning_weights=skinning_weights_data,
        joint_names=joint_names_data,
        root_orient=root_orient_data,
        rest_vertices=rest_vertices_data,
        rest_faces=rest_faces_data,
        skeleton=skeleton_data,
        subject=subject_name,
        vertex_part=vertex_part,
        rest_body_vertices=rest_body_vertices,
        rest_arm_vertices=rest_arm_vertices,
        body_width=body_width,
        full_width=full_width,
        joint_shape=shape_lst_array,
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


if __name__ == '__main__':
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
    for job in jobs:
        subject_name = job["subject_name"]
        fbx_path = job["fbx_path"]
        output_file = os.path.join(save_path, '%s.npz' % subject_name)
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
        )
        print(f"DONE: {subject_name}")
