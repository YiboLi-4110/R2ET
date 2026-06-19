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
    return parser.parse_args(argv)


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


def extract_data(fbx_path, subject_name, save_path):
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=True)
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
            np_rest_faces = np.array(rest_faces_lst)

    simplified_joint_names = list(JOINT_NAME_SMAL_33)
    simplified_joint_offsets = build_simplified_joint_offsets(source_arm, rest_origin)
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


if __name__ == '__main__':
    args = parse_args()
    fbx_root = args.fbx_root.resolve()
    save_path = args.save_path.resolve()

    fbx_name_lst = sorted(
        [path.name for path in fbx_root.iterdir() if path.is_file() and path.suffix.lower() == ".fbx"]
    )

    for fbx_name in fbx_name_lst:
        subject_name = fbx_name.split(".")[0]
        output_file = os.path.join(save_path, '%s.npz' % subject_name)
        if os.path.exists(output_file) and not args.overwrite_existing:
            print("SKIP:", subject_name)
            continue
        fbx_path = fbx_root / fbx_name
        extract_data(fbx_path, subject_name, save_path)
        print("DONE:", subject_name)
