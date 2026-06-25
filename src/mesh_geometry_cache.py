"""
Pre-load shape meshes and rest-pose convex-hull topology for RDF / ADF losses.

At runtime only hull *vertex positions* are updated via LBS-deformed coordinates;
face connectivity is fixed from rest pose, avoiding per-frame trimesh convex_hull.
"""

from __future__ import annotations

import numpy as np
import trimesh
import torch


def _convex_hull_mapping(rest_vertices, vertex_indices):
    """Build rest-pose convex hull; return global mesh indices + face list."""
    if len(vertex_indices) < 4:
        return None

    pts = rest_vertices[np.asarray(vertex_indices, dtype=np.int64)]
    hull = trimesh.points.PointCloud(vertices=pts).convex_hull
    if hull.vertices.shape[0] < 4 or hull.faces.shape[0] < 2:
        return None

    hull_global_indices = []
    for hv in np.asarray(hull.vertices):
        local_i = int(np.argmin(np.linalg.norm(pts - hv, axis=1)))
        hull_global_indices.append(int(vertex_indices[local_i]))

    return {
        "vertex_indices": hull_global_indices,
        "faces": np.asarray(hull.faces, dtype=np.int32),
    }


def build_mesh_geometry_cache(mesh_file_dic, mesh_groups):
    """
    Returns dict mesh_name -> {
        vertices: FloatTensor (V, 3),
        skin_weights: FloatTensor (V, J),
        hulls: {torso, head, hind, torso_adf} each optional hull dict
    }
    """
    cache = {}
    for mesh_name, fbx_data in mesh_file_dic.items():
        rest_vertices = np.asarray(fbx_data["rest_vertices"], dtype=np.float32)
        skin_weights = np.asarray(fbx_data["skinning_weights"], dtype=np.float32)

        torso_lst = mesh_groups["torso"][mesh_name]
        head_lst = mesh_groups["head"][mesh_name]
        hind_lst = list(mesh_groups["left_hind"][mesh_name]) + list(
            mesh_groups["right_hind"][mesh_name]
        )

        hulls = {}
        torso_hull = _convex_hull_mapping(rest_vertices, torso_lst)
        if torso_hull is not None:
            hulls["torso"] = torso_hull
            hulls["torso_adf"] = torso_hull
        head_hull = _convex_hull_mapping(rest_vertices, head_lst)
        if head_hull is not None:
            hulls["head"] = head_hull
        hind_hull = _convex_hull_mapping(rest_vertices, hind_lst)
        if hind_hull is not None:
            hulls["hind"] = hind_hull

        cache[mesh_name] = {
            "vertices": torch.from_numpy(rest_vertices),
            "skin_weights": torch.from_numpy(skin_weights),
            "hulls": hulls,
        }

    return cache
