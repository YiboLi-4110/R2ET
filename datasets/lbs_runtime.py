"""Shared linear-blend skinning for visualization.

Parents come from the shape npz ``topology`` when present. SMAL33 parents
are used only when the rest skeleton has 33 joints and no topology is given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.linear_blend_skin import linear_blend_skinning  # noqa: E402

from datasets.skeleton_io import parents_from_mesh_or_skel  # noqa: E402


@torch.no_grad()
def skin_mesh_sequence(quat_np, rest_skel_np, mesh_data, device, parents=None):
    parents_np = parents_from_mesh_or_skel(
        mesh_data=mesh_data,
        rest_skel=rest_skel_np,
        parents=parents,
    )
    quat_t = torch.from_numpy(np.asarray(quat_np)).float().to(device)
    rest_t = torch.from_numpy(np.asarray(rest_skel_np)).float().to(device)
    verts_t = torch.from_numpy(np.asarray(mesh_data["vertices"])).float().to(device)
    weights_t = torch.from_numpy(np.asarray(mesh_data["skin_weights"])).float().to(device)
    parent_t = torch.as_tensor(parents_np, dtype=torch.long, device=device)
    joint_count = int(rest_t.reshape(-1, 3).shape[0])
    if quat_t.shape[1] != joint_count:
        raise ValueError(
            f"LBS quat joints {quat_t.shape[1]} != rest skeleton {joint_count}"
        )
    if weights_t.shape[1] != joint_count:
        raise ValueError(
            f"LBS skin weights {weights_t.shape[1]} != rest skeleton {joint_count}"
        )
    if parent_t.numel() != joint_count:
        raise ValueError(
            f"LBS parents {parent_t.numel()} != rest skeleton {joint_count}"
        )
    out = linear_blend_skinning(parent_t, quat_t, rest_t, verts_t, weights_t)
    return out.detach().cpu().numpy()
