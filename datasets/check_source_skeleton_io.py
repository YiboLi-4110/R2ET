#!/usr/bin/env python3
"""Sanity-check Source LBS I/O for SMAL33 and sucaibao (~50 joint) clips.

Does not require GPU. Prints joint counts, skeleton_mode, and landmark indices.

Examples (from repo root):
  python datasets/check_source_skeleton_io.py
  python datasets/check_source_skeleton_io.py --skin --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.skeleton_io import (  # noqa: E402
    is_smal33_names,
    parse_bvh_hierarchy_names,
    resolve_landmark_indices,
)
from datasets.smal33_motion_io import (  # noqa: E402
    NUM_JOINTS,
    get_inp_from_bvh,
    load_mesh_from_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check generic vs SMAL33 source skeleton I/O.")
    parser.add_argument(
        "--sucaibao_bvh",
        type=Path,
        default=REPO_ROOT
        / "datasets/shepherd/cat_actions_sucaibao/train_char/Attack/Attack_Crouch_IP.bvh",
    )
    parser.add_argument(
        "--sucaibao_npz",
        type=Path,
        default=REPO_ROOT / "datasets/shepherd/cat_actions_sucaibao/train_shape/Attack.npz",
    )
    parser.add_argument(
        "--smal33_bvh",
        type=Path,
        default=REPO_ROOT
        / "datasets/shepherd/smal@shepherd/train_char/shepherd@Attack/shepherd@Attack_R_smal_dog-foot_on_ground.bvh",
    )
    parser.add_argument("--skin", action="store_true", help="Also run one-frame LBS (needs GPU).")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def report_motion(label: str, bvh_path: Path, npz_path: Path | None):
    print(f"\n== {label} ==")
    print(f"bvh: {bvh_path}")
    hier = parse_bvh_hierarchy_names(bvh_path)
    print(f"hierarchy joints: {len(hier)} smal33_names={is_smal33_names(hier)}")

    keep_names = None
    mesh = None
    if npz_path is not None and npz_path.is_file():
        mesh = load_mesh_from_npz(
            str(npz_path), canonicalize_bind_pose=True, forward_mode="body"
        )
        keep_names = mesh.get("joint_names")
        print(
            f"npz: {npz_path.name} weights={None if mesh['skin_weights'] is None else mesh['skin_weights'].shape} "
            f"skel={None if mesh['skeleton'] is None else mesh['skeleton'].shape} "
            f"names={None if keep_names is None else len(keep_names)}"
        )

    motion = get_inp_from_bvh(
        str(bvh_path),
        axis_transform="shepherd_y_z_x",
        forward_mode="body",
        keep_joint_names=keep_names,
    )
    quat = motion["quat"]
    skel = motion["skel"]
    print(
        f"mode={motion.get('_skeleton_mode')} quat={quat.shape} skel={skel.shape} "
        f"seq={motion['seq'].shape} kept={len(motion.get('joint_names') or [])}"
    )
    names = motion.get("joint_names") or []
    if names:
        marks = resolve_landmark_indices(names)
        print(f"landmarks: {marks}")
    if mesh is not None and mesh["skin_weights"] is not None:
        wj = mesh["skin_weights"].shape[1]
        qj = quat.shape[1]
        if wj != qj:
            raise SystemExit(f"FAIL: skin weights J={wj} != quat J={qj}")
        print("weights/quat joint count: OK")
    return motion, mesh


def maybe_skin(motion, mesh, device: int):
    from datasets.lbs_runtime import skin_mesh_sequence
    from datasets.smal33_motion_io import setup_cuda_device

    torch_device = setup_cuda_device(device)
    rest = mesh["skeleton"][0] if mesh["skeleton"].ndim == 3 else mesh["skeleton"]
    verts = skin_mesh_sequence(
        motion["quat"][:1],
        rest,
        mesh,
        torch_device,
        parents=mesh.get("topology"),
    )
    print(f"lbs frame0 verts={verts.shape} finite={bool(np.isfinite(verts).all())}")
    if not np.isfinite(verts).all():
        raise SystemExit("FAIL: LBS produced non-finite vertices")


def main():
    args = parse_args()
    sucaibao_motion, sucaibao_mesh = report_motion(
        "sucaibao", args.sucaibao_bvh, args.sucaibao_npz
    )
    if sucaibao_motion.get("_skeleton_mode") != "generic":
        raise SystemExit("FAIL: sucaibao clip should use generic skeleton_mode")
    if sucaibao_motion["quat"].shape[1] == NUM_JOINTS:
        raise SystemExit("FAIL: sucaibao clip was remapped to 33 joints")

    smal_npz = None
    smal_shape_dir = REPO_ROOT / "datasets/shepherd/smal@shepherd/train_shape"
    if smal_shape_dir.is_dir():
        cands = sorted(smal_shape_dir.glob("*.npz"))
        if cands:
            smal_npz = cands[0]
    smal_motion, smal_mesh = report_motion("smal33", args.smal33_bvh, smal_npz)
    if smal_motion.get("_skeleton_mode") != "smal33":
        raise SystemExit("FAIL: SMAL33 clip should use smal33 skeleton_mode")
    if smal_motion["quat"].shape[1] != NUM_JOINTS:
        raise SystemExit(
            f"FAIL: SMAL33 quat joints {smal_motion['quat'].shape[1]} != {NUM_JOINTS}"
        )

    if args.skin:
        if sucaibao_mesh is None:
            raise SystemExit("FAIL: --skin needs sucaibao npz")
        print("\n== LBS smoke ==")
        maybe_skin(sucaibao_motion, sucaibao_mesh, args.device)
        if smal_mesh is not None:
            maybe_skin(smal_motion, smal_mesh, args.device)

    print("\nOK")


if __name__ == "__main__":
    main()
