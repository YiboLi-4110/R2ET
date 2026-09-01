#!/usr/bin/env python3
"""
Diagnose why R2ET dog .blend packing (FBX + Ours BVH via mesh_visualize) looks
broken while the four-way LBS video path looks correct.

Checks (no Blender required for A–D; optional Blender for E):

  A) Ours BVH bone OFFSET lengths  -> zero-length leaf bones?
  B) Rest-bone direction agreement:
       raw target BVH (FBX-native)  vs  Ours BVH (as exported, often axis-transformed)
  C) Same as B after applying / undoing shepherd axis transforms
  D) Frame-0 joint positions:
       model-space FK/LBS rest skeleton  vs  Ours BVH global FK
  E) Optional: FBX armature vs Ours BVH bone directions inside Blender

Example (server, after batch_r2et_dog_blend --keep_intermediates):

  python visualization/diagnose_r2et_blend_binding.py \\
    --manifest temp/r2et_dog_blend_XXXX/work/博美_3/dog_actions_manifest.json \\
    --config config/visualization_blend_per_dog_smal33.yaml \\
    --output temp/diagnose_blend_binding_bomei3.json

  # Optional Blender FBX check (one action):
  blender --background --python visualization/diagnose_r2et_blend_binding.py -- \\
    --manifest .../dog_actions_manifest.json \\
    --action_id Attack_J \\
    --blender_fbx_check \\
    --output temp/diagnose_blend_binding_bomei3_blender.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTSIDE = _REPO_ROOT / "outside-code"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_OUTSIDE) not in sys.path:
    sys.path.insert(0, str(_OUTSIDE))

import Animation  # noqa: E402
import BVH  # noqa: E402

from datasets.smal33_motion_io import (  # noqa: E402
    NUM_JOINTS,
    SMAL33_PARENTS,
    apply_axis_transform_anim,
    axis_transform_matrix,
    get_inp_from_bvh,
    get_skel,
    load_mesh_from_npz,
    parse_bvh_joint_names,
    remap_bvh_anim,
)
from src.forward_kinematics import FK  # noqa: E402
from src.linear_blend_skin import linear_blend_skinning  # noqa: E402
import torch  # noqa: E402

# Full 33-joint names (Root + JOINTS_LIST).
BONE_NAMES = [
    "Root",
    "Spine1",
    "Spine2",
    "Spine3",
    "Spine4",
    "Spine5",
    "Spine6",
    "LeftScapula",
    "LeftUpperArm",
    "LeftForeLeg",
    "LeftFrontPaw",
    "RightScapula",
    "RightUpperArm",
    "RightForeLeg",
    "RightFrontPaw",
    "Neck",
    "Head",
    "Jaw",
    "LeftThigh",
    "LeftShin",
    "LeftHock",
    "LeftHindPaw",
    "RightThigh",
    "RightShin",
    "RightHock",
    "RightHindPaw",
    "Tail1",
    "Tail2",
    "Tail3",
    "Tail4",
    "Tail5",
    "Tail6",
    "Tail7",
]

FOCUS_BONES = [
    "Spine5",
    "Spine6",
    "Neck",
    "Head",
    "LeftForeLeg",
    "LeftFrontPaw",
    "RightForeLeg",
    "RightFrontPaw",
    "LeftHock",
    "LeftHindPaw",
    "RightHock",
    "RightHindPaw",
    "Jaw",
    "Tail7",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose R2ET blend FBX/BVH binding failures vs LBS path."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="dog_actions_manifest.json from export_r2et_dog_actions_smal33.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "config/visualization_blend_per_dog_smal33.yaml",
    )
    parser.add_argument(
        "--action_id",
        type=str,
        default=None,
        help="Diagnose only this action_id (default: first action in manifest).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index for FK/LBS comparison (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path.",
    )
    parser.add_argument(
        "--blender_fbx_check",
        action="store_true",
        default=False,
        help="Also compare FBX vs BVH bone dirs inside Blender (run under blender -P).",
    )
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def blender_argv():
    if "--" not in sys.argv:
        return sys.argv[1:]
    return sys.argv[sys.argv.index("--") + 1 :]


def norm_name(name: str) -> str:
    return name.split(":")[-1]


def safe_unit(vec: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return None
    return v / n


def vec_dot(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def load_bvh_raw(path: Path):
    anim, names, ftime = BVH.load(str(path))
    return anim, names, ftime


def load_bvh_remapped(path: Path, axis_transform: str | None = None):
    anim, names, ftime = BVH.load(str(path))
    joint_names = parse_bvh_joint_names(path)
    anim, to_keep = remap_bvh_anim(anim, joint_names)
    if axis_transform not in (None, "", "none"):
        anim = apply_axis_transform_anim(anim, axis_transform)
    return anim, names, ftime, to_keep


def offset_report(anim, names=None) -> dict[str, Any]:
    offsets = np.asarray(anim.offsets, dtype=np.float64)
    parents = np.asarray(anim.parents)
    rows = []
    zero = []
    for i in range(len(offsets)):
        length = float(np.linalg.norm(offsets[i]))
        name = None
        if names is not None and i < len(names):
            name = str(names[i])
        elif i < len(BONE_NAMES):
            name = BONE_NAMES[i]
        else:
            name = f"joint_{i}"
        entry = {
            "index": i,
            "name": name,
            "parent": int(parents[i]) if i < len(parents) else -1,
            "offset": [float(x) for x in offsets[i]],
            "length": length,
        }
        rows.append(entry)
        if i > 0 and length < 1e-8:
            zero.append(name)
    return {
        "joint_count": len(offsets),
        "zero_length_bones": zero,
        "bones": rows,
    }


def direction_compare(
    offsets_a: np.ndarray,
    offsets_b: np.ndarray,
    names: list[str],
    *,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    dots = []
    per_bone = []
    for i, name in enumerate(names):
        if i == 0:
            continue
        if i >= len(offsets_a) or i >= len(offsets_b):
            break
        ua = safe_unit(offsets_a[i])
        ub = safe_unit(offsets_b[i])
        d = vec_dot(ua, ub)
        item = {
            "index": i,
            "name": name,
            f"{label_a}_length": float(np.linalg.norm(offsets_a[i])),
            f"{label_b}_length": float(np.linalg.norm(offsets_b[i])),
            "direction_dot": d,
        }
        per_bone.append(item)
        if d is not None:
            dots.append(d)

    focus = [b for b in per_bone if norm_name(b["name"]) in FOCUS_BONES]
    outliers = [
        b
        for b in per_bone
        if b["direction_dot"] is not None and abs(b["direction_dot"]) < 0.7
    ]
    return {
        "label_a": label_a,
        "label_b": label_b,
        "direction_dot_mean": float(np.mean(dots)) if dots else None,
        "direction_dot_min": float(np.min(dots)) if dots else None,
        "direction_dot_p10": float(np.percentile(dots, 10)) if dots else None,
        "outlier_count_abs_dot_lt_0p7": len(outliers),
        "focus_bones": focus,
        "outliers": outliers,
    }


def rest_global_positions(anim) -> np.ndarray:
    """Global joint positions for frame-0 rest (identity-ish local orients)."""
    rest = anim.copy()
    # Use stored orients as rest local rotations (same trick as get_inp_from_bvh).
    rest.rotations.qs[...] = rest.orients.qs[None]
    if rest.positions.shape[0] < 1:
        raise RuntimeError("animation has no frames")
    rest.positions = get_skel(
        Animation.positions_global(rest)[0], rest.parents
    )[None]
    return Animation.positions_global(rest)[0]


def posed_global_positions(anim, frame: int) -> np.ndarray:
    frame = int(np.clip(frame, 0, anim.positions.shape[0] - 1))
    return Animation.positions_global(anim)[frame]


def pairwise_bone_gaps(positions: np.ndarray, parents: np.ndarray) -> list[dict[str, Any]]:
    gaps = []
    for i in range(1, len(positions)):
        p = int(parents[i])
        if p < 0:
            continue
        dist = float(np.linalg.norm(positions[i] - positions[p]))
        gaps.append(
            {
                "child": BONE_NAMES[i] if i < len(BONE_NAMES) else f"j{i}",
                "parent": BONE_NAMES[p] if p < len(BONE_NAMES) else f"j{p}",
                "gap": dist,
            }
        )
    return gaps


def summarize_focus_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_links = {
        ("Spine5", "Spine6"),
        ("Spine6", "Neck"),
        ("Neck", "Head"),
        ("LeftForeLeg", "LeftFrontPaw"),
        ("RightForeLeg", "RightFrontPaw"),
        ("LeftHock", "LeftHindPaw"),
        ("RightHock", "RightHindPaw"),
    }
    out = []
    for g in gaps:
        parent = norm_name(g["parent"])
        child = norm_name(g["child"])
        if (parent, child) in wanted_links or child in FOCUS_BONES:
            out.append(g)
    uniq = {g["child"]: g for g in out}
    return list(uniq.values())


def lbs_frame0_vertices(quat, rest_skel, mesh_data, device="cuda:0"):
    quat_t = torch.from_numpy(quat.astype(np.float32)).float().to(device)
    rest_t = torch.from_numpy(rest_skel.astype(np.float32)).float().to(device)
    verts_t = torch.from_numpy(mesh_data["vertices"]).float().to(device)
    weights_t = torch.from_numpy(mesh_data["skin_weights"]).float().to(device)
    parents = torch.as_tensor(SMAL33_PARENTS, dtype=torch.long, device=device)
    out = linear_blend_skinning(parents, quat_t, rest_t, verts_t, weights_t)
    return out.detach().cpu().numpy()


def fk_local_positions(quat, rest_skel, device="cuda:0"):
    parents = torch.as_tensor(SMAL33_PARENTS, dtype=torch.long, device=device)
    rest = np.repeat(rest_skel[None], len(quat), axis=0)
    local = FK.run(
        parents,
        torch.from_numpy(rest).float().to(device),
        torch.from_numpy(quat.astype(np.float32)).float().to(device),
    )
    return local.detach().cpu().numpy()


def diagnose_action(
    action: dict[str, Any],
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    frame: int,
) -> dict[str, Any]:
    motion_cfg = cfg.get("motion", {}) or {}
    tgt_axis = motion_cfg.get("tgt_axis_transform", "none")
    inp_axis = motion_cfg.get("inp_axis_transform", "none")
    tgt_forward = motion_cfg.get("tgt_forward_mode", "body")

    ours_bvh = Path(action["ours_bvh_path"])
    target_bvh = Path(manifest["target_bvh_path"])
    target_fbx = Path(manifest["target_fbx_path"])
    tgt_shape = Path(manifest.get("tgt_shape_path") or "")

    # Companion files written by retarget_to_bvh.
    action_dir = ours_bvh.parent
    target_rest_export = action_dir / ours_bvh.name.replace("_retarget.bvh", "_target_rest.bvh")
    if not target_rest_export.exists():
        # pair_tag is {action_id}_ours -> {action_id}_ours_target_rest.bvh
        stem = ours_bvh.name.replace("_retarget.bvh", "")
        target_rest_export = action_dir / f"{stem}_target_rest.bvh"

    report: dict[str, Any] = {
        "action_id": action.get("action_id"),
        "action_stem": action.get("action_stem"),
        "ours_bvh_path": str(ours_bvh),
        "target_bvh_path": str(target_bvh),
        "target_fbx_path": str(target_fbx),
        "config_tgt_axis_transform": tgt_axis,
        "config_inp_axis_transform": inp_axis,
        "frame": int(frame),
    }

    if not ours_bvh.exists():
        report["error"] = f"ours BVH missing: {ours_bvh}"
        return report
    if not target_bvh.exists():
        report["error"] = f"target BVH missing: {target_bvh}"
        return report

    # --- A) zero-length offsets in Ours BVH ---
    ours_raw, ours_names, _ = load_bvh_raw(ours_bvh)
    tgt_raw, tgt_names, _ = load_bvh_raw(target_bvh)
    report["A_ours_offset_lengths"] = offset_report(ours_raw, ours_names)
    report["A_target_raw_offset_lengths"] = offset_report(tgt_raw, tgt_names)
    report["A_verdict"] = {
        "ours_zero_length_bones": report["A_ours_offset_lengths"]["zero_length_bones"],
        "target_zero_length_bones": report["A_target_raw_offset_lengths"]["zero_length_bones"],
        "note": (
            "Blender BVH importer prints 'zero length node found' for these names "
            "and invents arbitrary bone orientations -> paw/jaw/tail tip flip."
        ),
    }

    # Remapped 33-joint views.
    ours_none, _, _, _ = load_bvh_remapped(ours_bvh, axis_transform="none")
    tgt_none, _, _, _ = load_bvh_remapped(target_bvh, axis_transform="none")
    tgt_model, _, _, _ = load_bvh_remapped(target_bvh, axis_transform=tgt_axis)

    names33 = BONE_NAMES[: ours_none.offsets.shape[0]]

    # --- B) raw target vs exported Ours (what Blender binds against FBX) ---
    report["B_raw_target_vs_ours_exported"] = direction_compare(
        tgt_none.offsets,
        ours_none.offsets,
        names33,
        label_a="raw_target",
        label_b="ours_exported",
    )
    report["B_verdict"] = {
        "mean_dot": report["B_raw_target_vs_ours_exported"]["direction_dot_mean"],
        "min_dot": report["B_raw_target_vs_ours_exported"]["direction_dot_min"],
        "interpretation": (
            "If mean_dot is low / many outliers, Ours BVH rest axes do NOT match "
            "the FBX-native target BVH. mesh_visualize binds Ours BVH onto FBX mesh "
            "-> broken Spine6/Neck gaps and twisted limbs."
        ),
    }

    # --- C) axis-transform hypotheses ---
    report["C_raw_target_vs_target_after_config_axis"] = direction_compare(
        tgt_none.offsets,
        tgt_model.offsets,
        names33,
        label_a="raw_target",
        label_b=f"target_after_{tgt_axis}",
    )
    report["C_ours_vs_target_after_config_axis"] = direction_compare(
        ours_none.offsets,
        tgt_model.offsets,
        names33,
        label_a="ours_exported",
        label_b=f"target_after_{tgt_axis}",
    )

    # Inverse of config axis on Ours: if this recovers raw target, export wrote model-space BVH.
    try:
        M = axis_transform_matrix(tgt_axis)
        Minv = np.linalg.inv(M)
        ours_inv_offsets = np.einsum("ij,...j->...i", Minv, ours_none.offsets)
        report["C_inverse_axis_on_ours_vs_raw_target"] = direction_compare(
            tgt_none.offsets,
            ours_inv_offsets,
            names33,
            label_a="raw_target",
            label_b=f"ours_inv_{tgt_axis}",
        )
    except Exception as exc:
        report["C_inverse_axis_on_ours_vs_raw_target"] = {"error": repr(exc)}

    c_inv = report.get("C_inverse_axis_on_ours_vs_raw_target", {})
    report["C_verdict"] = {
        "config_axis_changes_target_rest": report["C_raw_target_vs_target_after_config_axis"][
            "direction_dot_mean"
        ],
        "ours_matches_axis_transformed_target": report["C_ours_vs_target_after_config_axis"][
            "direction_dot_mean"
        ],
        "ours_after_inverse_matches_raw_target": c_inv.get("direction_dot_mean"),
        "interpretation": (
            "If ours_matches_axis_transformed_target ≈ 1 and "
            "ours_after_inverse_matches_raw_target ≈ 1, the exported Ours BVH is in "
            f"model coordinates ({tgt_axis}), while FBX/raw BVH are not."
        ),
    }

    # --- D) joint gaps + FK consistency ---
    # Model-space motion (what LBS video uses)
    tgt_motion = get_inp_from_bvh(
        str(target_bvh),
        axis_transform=tgt_axis,
        forward_mode=tgt_forward,
    )
    ours_as_file = get_inp_from_bvh(
        str(ours_bvh),
        axis_transform="none",  # already transformed when saved
        forward_mode=tgt_forward,
        canonicalize_bind_pose=False,
    )
    # Also parse Ours with same axis again (would double-transform if Ours is already transformed)
    ours_double = get_inp_from_bvh(
        str(ours_bvh),
        axis_transform=tgt_axis,
        forward_mode=tgt_forward,
        canonicalize_bind_pose=False,
    )

    report["D_parse"] = {
        "target_motion_ok": tgt_motion is not None,
        "ours_as_exported_parse_ok": ours_as_file is not None,
        "ours_with_config_axis_again_ok": ours_double is not None,
    }

    if tgt_motion is not None and ours_as_file is not None:
        rest_skel = tgt_motion["skel"][0].astype(np.float32)
        quat = ours_as_file["quat"].astype(np.float32)
        frame_i = int(np.clip(frame, 0, len(quat) - 1))

        fk_local = fk_local_positions(quat[frame_i : frame_i + 1], rest_skel)[0]
        bvh_global = posed_global_positions(ours_none, frame_i)
        # Align by root for fair distance compare
        fk_centered = fk_local - fk_local[0:1]
        bvh_centered = bvh_global - bvh_global[0:1]
        joint_err = np.linalg.norm(fk_centered - bvh_centered[:NUM_JOINTS], axis=-1)

        gaps_ours = pairwise_bone_gaps(bvh_global, ours_none.parents)
        gaps_tgt_raw = pairwise_bone_gaps(rest_global_positions(tgt_none), tgt_none.parents)
        gaps_tgt_model = pairwise_bone_gaps(rest_global_positions(tgt_model), tgt_model.parents)

        report["D_frame_joint_error_fk_model_vs_ours_bvh"] = {
            "rmse": float(np.sqrt(np.mean(joint_err**2))),
            "max": float(joint_err.max()),
            "per_focus": [
                {
                    "name": BONE_NAMES[i],
                    "err": float(joint_err[i]),
                }
                for i in range(min(len(joint_err), len(BONE_NAMES)))
                if BONE_NAMES[i] in FOCUS_BONES
            ],
        }
        report["D_parent_child_gaps"] = {
            "ours_exported_frame": summarize_focus_gaps(gaps_ours),
            "raw_target_rest": summarize_focus_gaps(gaps_tgt_raw),
            "target_after_axis_rest": summarize_focus_gaps(gaps_tgt_model),
            "note": (
                "Compare Spine6-Neck / ForeLeg-Paw gaps: raw_target_rest is what FBX "
                "expects; ours_exported_frame is what the blend armature uses."
            ),
        }

        if tgt_shape.exists():
            mesh = load_mesh_from_npz(str(tgt_shape))
            lbs_verts = lbs_frame0_vertices(quat[frame_i : frame_i + 1], rest_skel, mesh)[0]
            report["D_lbs"] = {
                "vertex_count": int(lbs_verts.shape[0]),
                "bbox_min": [float(x) for x in lbs_verts.min(axis=0)],
                "bbox_max": [float(x) for x in lbs_verts.max(axis=0)],
                "note": (
                    "LBS vertices are the same representation used by "
                    "render_fourway_lbs_blender.py (correct video path)."
                ),
            }
        else:
            report["D_lbs"] = {"skipped": f"shape npz missing: {tgt_shape}"}

        if ours_double is not None:
            q0 = ours_as_file["quat"][frame_i]
            q1 = ours_double["quat"][frame_i]
            # Quaternion absolute dot (sign ambiguity)
            dots = np.abs(np.sum(q0 * q1, axis=-1))
            report["D_double_axis_quat_abs_dot"] = {
                "mean": float(dots.mean()),
                "min": float(dots.min()),
                "note": (
                    "If this is clearly < 1, applying config axis again to Ours BVH "
                    "changes orientations -> Ours file is already axis-transformed."
                ),
            }

    report["D_verdict"] = {
        "interpretation": (
            "Large Spine6/Neck gap difference between raw_target_rest and "
            "ours_exported_frame, together with low B/C direction dots, confirms "
            "the blend path binds a model-space BVH onto an FBX-native mesh."
        )
    }

    report["paths_side"] = {
        "target_rest_export_exists": target_rest_export.exists(),
        "target_rest_export": str(target_rest_export),
    }
    return report


def blender_fbx_vs_bvh(fbx_path: Path, bvh_path: Path) -> dict[str, Any]:
    import bpy

    def clean():
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def new_objs(before):
        return [o for o in bpy.data.objects if o.name not in before]

    def pick_arm(objs):
        arms = [o for o in objs if o.type == "ARMATURE"]
        return arms[0] if arms else None

    clean()
    before = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), use_anim=False)
    fbx_arm = pick_arm(new_objs(before))
    if fbx_arm is None:
        return {"error": f"no armature in FBX: {fbx_path}"}

    before = {o.name for o in bpy.data.objects}
    bpy.ops.import_anim.bvh(filepath=str(bvh_path))
    bvh_arm = pick_arm(new_objs(before))
    if bvh_arm is None:
        return {"error": f"no armature in BVH: {bvh_path}"}

    def bone_dir(arm, name):
        bone = arm.data.bones.get(name)
        if bone is None:
            # try without / with prefix
            for b in arm.data.bones:
                if norm_name(b.name) == norm_name(name):
                    bone = b
                    break
        if bone is None:
            return None, 0.0
        vec = bone.tail_local - bone.head_local
        length = float(vec.length)
        if length < 1e-8:
            return None, length
        vec.normalize()
        return (float(vec.x), float(vec.y), float(vec.z)), length

    fbx_names = [b.name for b in fbx_arm.data.bones]
    bvh_names = [b.name for b in bvh_arm.data.bones]
    zero_bvh = []
    rows = []
    dots = []
    for bn in fbx_names:
        fd, fl = bone_dir(fbx_arm, bn)
        bd, bl = bone_dir(bvh_arm, bn)
        if bl < 1e-8 and norm_name(bn) in {
            "LeftFrontPaw",
            "RightFrontPaw",
            "LeftHindPaw",
            "RightHindPaw",
            "Jaw",
            "Tail7",
        }:
            zero_bvh.append(bn)
        d = None
        if fd is not None and bd is not None:
            d = float(sum(a * b for a, b in zip(fd, bd)))
            dots.append(d)
        if norm_name(bn) in FOCUS_BONES or (d is not None and abs(d) < 0.7):
            rows.append(
                {
                    "bone": bn,
                    "fbx_length": fl,
                    "bvh_length": bl,
                    "direction_dot": d,
                }
            )

    return {
        "fbx_armature": fbx_arm.name,
        "bvh_armature": bvh_arm.name,
        "fbx_bone_count": len(fbx_names),
        "bvh_bone_count": len(bvh_names),
        "zero_length_like_bvh_focus": zero_bvh,
        "direction_dot_mean": float(np.mean(dots)) if dots else None,
        "direction_dot_min": float(np.min(dots)) if dots else None,
        "focus_or_outlier_bones": rows,
        "note": (
            "This is exactly what mesh_visualize sees: FBX rest axes vs imported "
            "Ours BVH axes inside Blender."
        ),
    }


def overall_verdict(action_report: dict[str, Any]) -> dict[str, Any]:
    flags = []
    a_zero = action_report.get("A_verdict", {}).get("ours_zero_length_bones") or []
    if a_zero:
        flags.append(
            {
                "id": "zero_length_leaf_bones",
                "severity": True,
                "detail": a_zero,
            }
        )

    b_mean = action_report.get("B_verdict", {}).get("mean_dot")
    if b_mean is not None and b_mean < 0.85:
        flags.append(
            {
                "id": "rest_axis_mismatch_raw_target_vs_ours",
                "severity": True,
                "mean_dot": b_mean,
            }
        )

    c_match = action_report.get("C_verdict", {}).get("ours_matches_axis_transformed_target")
    c_inv = action_report.get("C_verdict", {}).get("ours_after_inverse_matches_raw_target")
    if c_match is not None and c_match > 0.95 and c_inv is not None and c_inv > 0.95:
        flags.append(
            {
                "id": "ours_bvh_is_model_axis_not_fbx_native",
                "severity": True,
                "ours_matches_transformed_target": c_match,
                "inverse_recovers_raw_target": c_inv,
            }
        )

    e = action_report.get("E_blender_fbx_vs_ours")
    if isinstance(e, dict) and e.get("direction_dot_mean") is not None:
        if e["direction_dot_mean"] < 0.85:
            flags.append(
                {
                    "id": "blender_fbx_vs_bvh_direction_mismatch",
                    "severity": True,
                    "mean_dot": e["direction_dot_mean"],
                    "min_dot": e.get("direction_dot_min"),
                }
            )

    if not flags:
        flags.append(
            {
                "id": "no_strong_mismatch_detected",
                "severity": False,
                "detail": "Check focus gaps manually; may need another action/frame.",
            }
        )
    return {"likely_causes": flags}


def main(argv=None):
    if argv is None:
        # blender -P this.py -- <args>
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        else:
            argv = sys.argv[1:]
    args = parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cfg = load_yaml(args.config.resolve()) if args.config.exists() else {}

    actions = list(manifest.get("actions") or [])
    if not actions:
        raise SystemExit("manifest has no actions[]")
    if args.action_id:
        actions = [a for a in actions if a.get("action_id") == args.action_id]
        if not actions:
            raise SystemExit(f"action_id not found: {args.action_id}")
    else:
        actions = actions[:1]

    reports = []
    for action in actions:
        print(f"[diagnose] action={action.get('action_id')}")
        rep = diagnose_action(action, manifest, cfg, args.frame)
        if args.blender_fbx_check:
            try:
                import bpy  # noqa: F401

                rep["E_blender_fbx_vs_ours"] = blender_fbx_vs_bvh(
                    Path(manifest["target_fbx_path"]),
                    Path(action["ours_bvh_path"]),
                )
            except ImportError:
                rep["E_blender_fbx_vs_ours"] = {
                    "error": "bpy not available; run this script under blender -P"
                }
        rep["overall_verdict"] = overall_verdict(rep)
        reports.append(rep)

        v = rep["overall_verdict"]["likely_causes"]
        print("[diagnose] likely causes:")
        for item in v:
            print(f"  - {item['id']}: {json.dumps(item, ensure_ascii=False)}")
        a_zero = rep.get("A_verdict", {}).get("ours_zero_length_bones")
        print(f"[diagnose] zero-length Ours bones: {a_zero}")
        print(
            "[diagnose] B mean_dot(raw_target vs ours): "
            f"{rep.get('B_verdict', {}).get('mean_dot')}"
        )
        print(
            "[diagnose] C ours~transformed_target / inv~raw: "
            f"{rep.get('C_verdict', {}).get('ours_matches_axis_transformed_target')} / "
            f"{rep.get('C_verdict', {}).get('ours_after_inverse_matches_raw_target')}"
        )
        gaps = rep.get("D_parent_child_gaps", {})
        if gaps:
            print("[diagnose] focus gaps ours_exported_frame:")
            for g in gaps.get("ours_exported_frame", [])[:12]:
                print(f"    {g['parent']} -> {g['child']}: {g['gap']:.5f}")
            print("[diagnose] focus gaps raw_target_rest:")
            for g in gaps.get("raw_target_rest", [])[:12]:
                print(f"    {g['parent']} -> {g['child']}: {g['gap']:.5f}")

    payload = {
        "dog_id": manifest.get("dog_id"),
        "manifest": str(args.manifest),
        "config": str(args.config),
        "stage2_mode": manifest.get("stage2_mode"),
        "gate_scale": manifest.get("gate_scale"),
        "actions": reports,
    }
    out = args.output
    if out is None:
        out = Path("temp") / f"diagnose_blend_binding_{manifest.get('dog_id', 'dog')}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diagnose] wrote {out}")


if __name__ == "__main__":
    main()
