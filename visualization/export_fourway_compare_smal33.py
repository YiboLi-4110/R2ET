#!/usr/bin/env python3
"""
Export four-way SMAL33 comparison payloads for Blender rendering.

Outputs per case:
  - fourway_compare.npz     (mesh vertex sequences + metadata)
  - fourway_manifest.json   (asset/action paths for Blender stage)
  - copyquat / ours BVH     (generated with shared retarget writer)

Four lanes:
  1) source mesh on source action
  2) target mesh driven by copy-quat baseline
  3) target mesh driven by Blender ARP output
  4) target mesh driven by our shape-stage model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import sys
# 将项目根目录加入 sys.path，以便导入 datasets / src 等顶级包。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# 将脚本所在目录加入 sys.path，以便导入同目录下的本地模块（如 compare_assets）。
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from compare_assets import enrich_case_assets
from compare_lanes import compare_lane_titles, include_copyquat_lane

from datasets.lbs_runtime import skin_mesh_sequence
from datasets.smal33_motion_io import (
    NUM_JOINTS,
    SMAL33_PARENTS,
    build_model_inputs,
    get_height_from_skel,
    get_inp_from_bvh,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
    load_shape_retnet,
    load_shape_vector,
    load_stats,
    report_lbs_rest_skel_mismatch,
    retarget_to_bvh,
    setup_cuda_device,
)
from src.forward_kinematics import FK

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Export four-way SMAL33 compare payloads.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/visualization_compare_smal33.yaml",
    )
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    parser.add_argument(
        "--skip_arp_if_missing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def motion_parse_options(motion_cfg: dict[str, Any], prefix: str):
    return {
        "axis_transform": motion_cfg.get(f"{prefix}_axis_transform", "none"),
        "forward_mode": motion_cfg.get(f"{prefix}_forward_mode", "body"),
        "post_axis_yaw_deg": float(motion_cfg.get(f"{prefix}_post_axis_yaw_deg", 0.0) or 0.0),
        "canonicalize_bind_pose": bool(
            motion_cfg.get(f"{prefix}_canonicalize_bind_pose", True)
        ),
    }


def mesh_load_options(motion_cfg: dict[str, Any], prefix: str):
    return {
        "canonicalize_bind_pose": bool(
            motion_cfg.get(f"{prefix}_canonicalize_bind_pose", True)
        ),
        "forward_mode": motion_cfg.get(f"{prefix}_forward_mode", "body"),
    }


@torch.no_grad()
def run_stage2_inference(
    model, inp_motion, tgt_motion, inp_shape, tgt_shape, stats, device, gate_scale=1.0
):
    inp_batch = build_model_inputs(inp_motion, stats, inp_shape, device)
    tgt_batch = build_model_inputs(tgt_motion, stats, tgt_shape, device)
    # gate_scale maps to the model's global balance-gate multiplier `k`
    # (qB = lerp(qB_base, qB_hat, gate*k)). >1 strengthens the shape/collision
    # correction at inference time, no retrain needed. See RetNet.forward.
    local_b, global_b, quat_b, _, _ = model(
        inp_batch["seq"],
        tgt_batch["seq"],
        inp_batch["skel"],
        tgt_batch["skel"],
        inp_batch["shape"],
        tgt_batch["shape"],
        inp_batch["quat"],
        inp_batch["height"],
        tgt_batch["height"],
        stats["local_mean"],
        stats["local_std"],
        stats["quat_mean"],
        stats["quat_std"],
        SMAL33_PARENTS,
        k=float(gate_scale),
        phase="test",
    )
    return (
        local_b[0].cpu().numpy(),
        global_b[0].cpu().numpy(),
        quat_b[0].cpu().numpy(),
    )


def align_frames(arr: np.ndarray, num_frames: int) -> np.ndarray:
    if len(arr) == num_frames:
        return arr
    if len(arr) <= 1:
        return np.repeat(arr, num_frames, axis=0)
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(np.int64)
    return arr[idx]


def resolve_arp_bvh_path(case_cfg: dict[str, Any], cfg: dict[str, Any]) -> Path:
    explicit = case_cfg.get("arp_bvh_path")
    if explicit:
        return Path(explicit)
    arp_cfg = cfg.get("arp", {})
    output_dir = Path(arp_cfg.get("output_dir", "./visualization/videos/compare/arp_outputs"))
    suffix = arp_cfg.get("output_suffix", "_arp_retarget.bvh")
    return output_dir / f"{case_cfg['case_id']}{suffix}"


def resolve_arp_mesh_path(case_cfg: dict[str, Any], cfg: dict[str, Any]) -> Path:
    explicit = case_cfg.get("arp_mesh_path")
    if explicit:
        return Path(explicit)
    arp_cfg = cfg.get("arp", {})
    output_dir = Path(
        arp_cfg.get("mesh_output_dir", "./visualization/videos/compare/arp_mesh_outputs")
    )
    return output_dir / f"{case_cfg['case_id']}_arp_mesh.npz"


def blender_z_up_to_lbs_y_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate Blender Z-up ARP verts into the LBS Y-up space of the other lanes.

    IMPORTANT: this must be a proper rotation (det=+1), NOT a bare axis swap.
    The old implementation used ``verts[..., [0, 2, 1]]``, which swaps the Y/Z
    axes -> determinant -1 -> a reflection that flips chirality. That forced the
    orientation auto-picker to select a ``mirror_*`` reflection to compensate,
    and a reflection swaps the animal's left/right limbs.

    Rotating -90 deg about X keeps handedness: (x, y, z) -> (x, z, -y).
    """
    verts = np.asarray(vertices, dtype=np.float32)
    out = verts[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    return out


def bbox_center_per_frame(vertices: np.ndarray) -> np.ndarray:
    mn = vertices.min(axis=1)
    mx = vertices.max(axis=1)
    return (mn + mx) * 0.5


def center_vertices_per_frame(vertices: np.ndarray) -> np.ndarray:
    centers = bbox_center_per_frame(vertices)
    return vertices - centers[:, None, :]


def apply_orientation_transform(vertices: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(vertices, dtype=np.float32).copy()
    if mode == "identity":
        return out
    if mode == "yaw_180":
        out[..., 0] *= -1.0
        out[..., 2] *= -1.0
        return out
    if mode == "mirror_x":
        out[..., 0] *= -1.0
        return out
    if mode == "mirror_z":
        out[..., 2] *= -1.0
        return out
    raise ValueError(f"Unknown ARP orientation transform: {mode}")


def auto_pick_orientation_transform(
    arp_vertices: np.ndarray,
    reference_vertices: np.ndarray | None,
) -> str:
    if reference_vertices is None:
        return "identity"
    if arp_vertices.shape != reference_vertices.shape:
        return "identity"

    # Only proper rotations are allowed: a mirror_* reflection can score lower on
    # a near-symmetric quadruped, but it swaps left/right limbs. Handedness is
    # already fixed correctly in blender_z_up_to_lbs_y_up.
    candidates = ("identity", "yaw_180")
    arp_centered = center_vertices_per_frame(arp_vertices).astype(np.float64)
    ref_centered = center_vertices_per_frame(reference_vertices).astype(np.float64)

    scores = {}
    for name in candidates:
        transformed = apply_orientation_transform(arp_centered, name)
        rmse = np.sqrt(np.mean((transformed - ref_centered) ** 2, axis=(1, 2)))
        scores[name] = float(rmse.mean())
    best = min(scores, key=scores.get)
    print(f"[export][arp] orientation auto selected={best} scores={scores}")
    return best


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove global horizontal translation while keeping local deformation.

    ARP evaluated mesh cache is exported in Blender world-space. That preserves
    physically meaningful root motion, but in lane-locked comparison videos this
    can look like panel-to-panel teleporting. We therefore subtract only per-frame
    horizontal center drift and keep the vertical channel unchanged.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) <= 1:
        return verts, np.zeros((len(verts), 3), dtype=np.float32)

    centers = bbox_center_per_frame(verts)
    if anchor == "median":
        ref = np.median(centers, axis=0, keepdims=True).astype(np.float32)
    else:
        # Default: keep frame-0 as canonical lane center.
        ref = centers[:1].astype(np.float32)

    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    corrected = verts - drift[:, None, :]
    return corrected, drift


def load_arp_mesh_cache(
    path: Path,
    num_frames: int,
    arp_cfg: dict[str, Any] | None = None,
    reference_vertices: np.ndarray | None = None,
) -> np.ndarray:
    arp_cfg = arp_cfg or {}
    payload = np.load(str(path))
    verts = payload["vertices"].astype(np.float32)
    space = str(payload["space"]) if "space" in payload.files else "blender_world_z_up"
    if space in {
        "blender_world_z_up",
        "mesh_local_zup",
        "armature_local_zup",
        "root_bone_local_zup",
        "world_root_h_locked_zup",
    }:
        verts = blender_z_up_to_lbs_y_up(verts)
    else:
        print(f"[export][arp][warn] unknown mesh space '{space}', assuming blender_world_z_up")
        verts = blender_z_up_to_lbs_y_up(verts)
    verts = align_frames(verts, num_frames)

    orientation_mode = str(arp_cfg.get("mesh_orientation_correction", "auto")).lower()
    if orientation_mode == "auto":
        orientation_mode = auto_pick_orientation_transform(verts, reference_vertices)
    if orientation_mode not in ("identity", "yaw_180", "mirror_x", "mirror_z"):
        print(
            f"[export][arp][warn] unknown mesh_orientation_correction='{orientation_mode}', "
            "fallback to identity"
        )
        orientation_mode = "identity"
    verts = apply_orientation_transform(verts, orientation_mode)
    print(f"[export][arp] orientation_correction={orientation_mode}")

    lock_cfg = arp_cfg.get("mesh_lock_horizontal_translation", None)
    if lock_cfg is None:
        # Spaces that already cancel root horizontal translation at the source
        # (root-local / root-horizontal-locked) must NOT be bbox-locked again: a
        # bbox-center lock shifts with limb/tail extension and reintroduces
        # limb-driven wobble. Only the raw world/local spaces need the lock.
        source_locked = space in ("root_bone_local_zup", "world_root_h_locked_zup")
        lock_enabled = not source_locked
    else:
        lock_enabled = bool(lock_cfg)

    if lock_enabled:
        anchor = str(arp_cfg.get("mesh_lock_reference", "first")).lower()
        if anchor not in ("first", "median"):
            anchor = "first"
        verts, drift = remove_horizontal_drift(
            verts,
            vertical_axis=1,
            anchor=anchor,
        )
        drift_norm = np.linalg.norm(drift, axis=-1)
        print(
            "[export][arp] lock_horizontal_translation="
            f"on anchor={anchor} drift_max={float(drift_norm.max()):.5f} "
            f"drift_p95={float(np.percentile(drift_norm, 95.0)):.5f}"
        )
    else:
        print("[export][arp] lock_horizontal_translation=off")
    return verts


def compute_copyquat_outputs(inp_motion, tgt_motion, device, rest_skel_tgt=None):
    source_quat = inp_motion["quat"].astype(np.float32)
    num_frames = len(source_quat)
    global_in = inp_motion["seq"][:, -8:-4].astype(np.float32)
    h_in = float(get_height_from_skel(inp_motion["skel"][0]))
    rest_skel_tgt = (
        np.asarray(rest_skel_tgt, dtype=np.float32)
        if rest_skel_tgt is not None
        else tgt_motion["skel"][0].astype(np.float32)
    )
    h_tgt = float(get_height_from_skel(rest_skel_tgt))
    scale = 1.0 if abs(h_in) < 1e-8 else (h_tgt / h_in)

    global_out = np.zeros_like(global_in, dtype=np.float32)
    global_out[:, :3] = global_in[:, :3] * scale
    global_out[:, 3] = global_in[:, 3]

    rest_repeat = np.repeat(rest_skel_tgt[None], num_frames, axis=0)
    parents = torch.as_tensor(SMAL33_PARENTS, dtype=torch.long, device=device)
    local_out = (
        FK.run(
            parents,
            torch.from_numpy(rest_repeat).float().to(device),
            torch.from_numpy(source_quat).float().to(device),
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return local_out, global_out.astype(np.float32), source_quat


def export_case(case_cfg: dict[str, Any], cfg: dict[str, Any], stage2_model, stats, device, output_root: Path):
    case_id = case_cfg["case_id"]
    case_out = output_root / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    motion_cfg = cfg.get("motion", {})
    inp_motion = get_inp_from_bvh(
        case_cfg["inp_bvh_path"],
        **motion_parse_options(motion_cfg, "inp"),
    )
    tgt_motion = get_inp_from_bvh(
        case_cfg["tgt_bvh_path"],
        **motion_parse_options(motion_cfg, "tgt"),
    )
    if inp_motion is None or tgt_motion is None:
        raise RuntimeError(f"[{case_id}] failed to parse input/target BVH.")

    inp_shape = load_shape_vector(case_cfg["inp_shape_path"])
    tgt_shape = load_shape_vector(case_cfg["tgt_shape_path"])
    inp_mesh = load_mesh_from_npz(
        case_cfg["inp_shape_path"], **mesh_load_options(motion_cfg, "inp")
    )
    tgt_mesh = load_mesh_from_npz(
        case_cfg["tgt_shape_path"], **mesh_load_options(motion_cfg, "tgt")
    )
    inp_rest_skel = lbs_rest_skel_from_mesh(
        inp_mesh, fallback_skel=inp_motion["skel"][0]
    )
    tgt_rest_skel = lbs_rest_skel_from_mesh(
        tgt_mesh, fallback_skel=tgt_motion["skel"][0]
    )
    report_lbs_rest_skel_mismatch(
        inp_rest_skel,
        inp_motion["skel"][0],
        label=f"{case_id}/inp",
        mesh_canonicalized=inp_mesh.get("bind_pose_canonicalized"),
    )
    report_lbs_rest_skel_mismatch(
        tgt_rest_skel,
        tgt_motion["skel"][0],
        label=f"{case_id}/tgt",
        mesh_canonicalized=tgt_mesh.get("bind_pose_canonicalized"),
    )

    num_frames = len(inp_motion["quat"])
    render_cfg = cfg.get("render", {})
    camera_cfg = cfg.get("camera", {})
    export_copyquat = include_copyquat_lane(render_cfg)

    src_quat = inp_motion["quat"].astype(np.float32)
    src_quat = align_frames(src_quat, num_frames)
    source_verts = skin_mesh_sequence(
        src_quat,
        inp_rest_skel,
        inp_mesh,
        device,
    )

    copy_bvh = None
    copy_verts = None
    if export_copyquat:
        copy_local, copy_global, copy_quat = compute_copyquat_outputs(
            inp_motion, tgt_motion, device, rest_skel_tgt=tgt_rest_skel
        )
        copy_local = align_frames(copy_local, num_frames)
        copy_global = align_frames(copy_global, num_frames)
        copy_quat = align_frames(copy_quat, num_frames)
        copy_verts = skin_mesh_sequence(
            copy_quat,
            tgt_rest_skel,
            tgt_mesh,
            device,
        )

    stage2_gate_scale = float(cfg.get("stage2", {}).get("gate_scale", 1.0))
    ours_local, ours_global, ours_quat = run_stage2_inference(
        stage2_model,
        inp_motion,
        tgt_motion,
        inp_shape,
        tgt_shape,
        stats,
        device,
        gate_scale=stage2_gate_scale,
    )
    ours_local = align_frames(ours_local.astype(np.float32), num_frames)
    ours_global = align_frames(ours_global.astype(np.float32), num_frames)
    ours_quat = align_frames(ours_quat.astype(np.float32), num_frames)
    ours_verts = skin_mesh_sequence(
        ours_quat,
        tgt_rest_skel,
        tgt_mesh,
        device,
    )

    pair_tag = case_id
    if export_copyquat:
        _, _, copy_bvh = retarget_to_bvh(
            inp_motion,
            tgt_motion,
            copy_local,
            copy_global,
            copy_quat,
            stats,
            case_out,
            f"{pair_tag}_copyquat",
            inp_bvh_path=case_cfg["inp_bvh_path"],
            tgt_bvh_path=case_cfg["tgt_bvh_path"],
            local_is_normalized=False,
        )
    _, _, ours_bvh = retarget_to_bvh(
        inp_motion,
        tgt_motion,
        ours_local,
        ours_global,
        ours_quat,
        stats,
        case_out,
        f"{pair_tag}_ours",
        inp_bvh_path=case_cfg["inp_bvh_path"],
        tgt_bvh_path=case_cfg["tgt_bvh_path"],
    )

    arp_bvh = resolve_arp_bvh_path(case_cfg, cfg)
    arp_mesh = resolve_arp_mesh_path(case_cfg, cfg)
    arp_verts = None
    arp_ok = False
    if arp_mesh.exists():
        arp_ref = copy_verts if (export_copyquat and copy_verts is not None) else ours_verts
        arp_verts = load_arp_mesh_cache(
            arp_mesh,
            num_frames,
            cfg.get("arp", {}),
            reference_vertices=arp_ref,
        )
        arp_ok = True
        print(f"[export]   arp_mesh: {arp_mesh}")
    elif arp_bvh.exists():
        print(
            f"[export][warn] ARP mesh cache missing ({arp_mesh}); "
            "falling back to BVH-driven LBS, which may be less reliable."
        )
        arp_motion = get_inp_from_bvh(
            str(arp_bvh),
            axis_transform=motion_cfg.get("arp_axis_transform", "none"),
            forward_mode=motion_cfg.get("arp_forward_mode", "body"),
            canonicalize_bind_pose=False,
        )
        if arp_motion is not None:
            arp_quat = align_frames(arp_motion["quat"].astype(np.float32), num_frames)
            # Target mesh weights are bound to the (canonicalized) shape skeleton.
            arp_verts = skin_mesh_sequence(
                arp_quat,
                tgt_rest_skel,
                tgt_mesh,
                device,
            )
            arp_ok = True

    if not arp_ok:
        if cfg.get("skip_arp_if_missing", False):
            arp_verts = np.zeros_like(ours_verts, dtype=np.float32)
        else:
            raise FileNotFoundError(
                f"[{case_id}] ARP output missing or invalid: {arp_bvh}. "
                "Run ARP batch first or enable skip_arp_if_missing."
            )

    npz_path = case_out / "fourway_compare.npz"
    npz_payload = {
        "source_vertices": source_verts.astype(np.float32),
        "arp_vertices": arp_verts.astype(np.float32),
        "ours_vertices": ours_verts.astype(np.float32),
        "source_faces": inp_mesh["faces"].astype(np.int32),
        "target_faces": tgt_mesh["faces"].astype(np.int32),
        "fps": np.int32(int(render_cfg.get("fps", 30))),
        "frame_count": np.int32(num_frames),
        "camera_zoom": np.float32(float(camera_cfg.get("camera_zoom", 1.0))),
        "view_elev": np.float32(float(camera_cfg.get("view_elev", 18.0))),
        "view_azim": np.float32(float(camera_cfg.get("view_azim", -75.0))),
    }
    if export_copyquat and copy_verts is not None:
        npz_payload["copyquat_vertices"] = copy_verts.astype(np.float32)
    np.savez_compressed(npz_path, **npz_payload)

    manifest = {
        "case_id": case_id,
        "npz_path": str(npz_path),
        "lane_titles": compare_lane_titles(render_cfg),
        "source_bvh_path": case_cfg["inp_bvh_path"],
        "target_bvh_path": case_cfg["tgt_bvh_path"],
        "copyquat_bvh_path": str(copy_bvh) if copy_bvh is not None else None,
        "arp_bvh_path": str(arp_bvh),
        "arp_mesh_path": str(arp_mesh) if arp_mesh.exists() else None,
        "ours_bvh_path": str(ours_bvh),
        "source_fbx_path": case_cfg["inp_fbx_path"],
        "target_fbx_path": case_cfg["tgt_fbx_path"],
        "inp_shape_path": case_cfg["inp_shape_path"],
        "tgt_shape_path": case_cfg["tgt_shape_path"],
        "camera": camera_cfg,
        "render": render_cfg,
        "arp_ok": arp_ok,
    }
    manifest_path = case_out / "fourway_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = parse_args()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root)
    if args.device is not None:
        cfg["device"] = int(args.device)
    if args.skip_arp_if_missing is not None:
        cfg["skip_arp_if_missing"] = bool(args.skip_arp_if_missing)

    output_root = Path(cfg.get("output_root", "./visualization/videos/compare"))
    output_root.mkdir(parents=True, exist_ok=True)

    case_ids = set(args.case_ids) if args.case_ids else None
    all_cases = cfg.get("cases", [])
    cases = [c for c in all_cases if case_ids is None or c.get("case_id") in case_ids]
    if not cases:
        raise SystemExit("No cases to process (check --case_ids or config.cases).")

    device = setup_cuda_device(int(cfg.get("device", 0)))
    model_cfg = cfg.get("model", {})
    stage2_weights = model_cfg.get("stage2_weights")
    stats_path = model_cfg.get("stats_path")
    if not stage2_weights or not stats_path:
        raise SystemExit("Missing model.stage2_weights or model.stats_path in config.")

    stats = load_stats(stats_path)
    stage2_model = load_shape_retnet(stage2_weights, cfg["ret_model_args"], device)

    manifests = []
    assets_cfg = cfg.get("assets", {})
    for case in cases:
        case_id = case["case_id"]
        case = enrich_case_assets(case, assets_cfg)
        print(f"[export] {case_id}")
        print(f"[export]   inp_fbx: {case['inp_fbx_path']}")
        print(f"[export]   tgt_fbx: {case['tgt_fbx_path']}")
        manifest = export_case(case, cfg, stage2_model, stats, device, output_root)
        manifests.append(manifest)
        print(f"[export] done: {case_id}")

    index_path = output_root / "fourway_manifest_index.json"
    index_path.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[export] wrote index: {index_path}")


if __name__ == "__main__":
    main()