"""
Shared SMAL33 motion I/O utilities for eval / inference.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage.filters as filters
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTSIDE_CODE = REPO_ROOT / "outside-code"
if str(OUTSIDE_CODE) not in sys.path:
    sys.path.insert(0, str(OUTSIDE_CODE))

import Animation  # noqa: E402
import BVH  # noqa: E402
from Quaternions import Quaternions  # noqa: E402
from Pivots import Pivots  # noqa: E402

NUM_JOINTS = 33
GLOBAL_DIM = 4
SEQ_TAIL_DIM = 8

SMAL33_PARENTS = np.array(
    [
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 16,
        0, 18, 19, 20, 0, 22, 23, 24, 0, 26, 27, 28, 29, 30, 31,
    ],
    dtype=np.int64,
)

JOINTS_LIST = [
    "Spine1", "Spine2", "Spine3", "Spine4", "Spine5", "Spine6",
    "LeftScapula", "LeftUpperArm", "LeftForeLeg", "LeftFrontPaw",
    "RightScapula", "RightUpperArm", "RightForeLeg", "RightFrontPaw",
    "Neck", "Head", "Jaw",
    "LeftThigh", "LeftShin", "LeftHock", "LeftHindPaw",
    "RightThigh", "RightShin", "RightHock", "RightHindPaw",
    "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6", "Tail7",
]

ATTENTION_JOINTS = [9, 10, 13, 14, 19, 20, 21, 23, 24, 25]
BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17]
LIMB_JOINTS = ATTENTION_JOINTS

STATS_FILES = {
    "local_mean": "smal33_local_motion_mean.npy",
    "local_std": "smal33_local_motion_std.npy",
    "global_mean": "smal33_global_motion_mean.npy",
    "global_std": "smal33_global_motion_std.npy",
    "quat_mean": "smal33_quat_mean.npy",
    "quat_std": "smal33_quat_std.npy",
}


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def get_skel(joints, parents):
    c_offsets = []
    for j in range(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[j, :] - joints[parents[j], :])
        else:
            c_offsets.append(joints[j, :])
    return np.stack(c_offsets, axis=0)


def process_positions(positions):
    fid_l, fid_r = np.array([10, 21]), np.array([14, 25])
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    positions = positions.copy()
    positions[:, :, 1] -= floor_height

    reference = positions[:, 0]
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)

    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (
        ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    ).astype(np.float32)

    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()
    positions[:, :, 0] = positions[:, :, 0] - positions[:, :1, 0]
    positions[1:, 1:, 1] = positions[1:, 1:, 1] - (
        positions[1:, :1, 1] - positions[:1, :1, 1]
    )
    positions[:, :, 2] = positions[:, :, 2] - positions[:, :1, 2]

    sdr_l, sdr_r, hip_l, hip_r = 9, 13, 19, 23
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions

    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 1]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    return positions, rotation


def remap_bvh_anim(anim, joint_names_in_file):
    to_keep = [0]
    for jname in JOINTS_LIST:
        for k, name in enumerate(joint_names_in_file):
            if jname == name[-len(jname) :]:
                to_keep.append(k + 1)
                break

    anim.parents = anim.parents[to_keep]
    for i in range(1, len(anim.parents)):
        if anim.parents[i] not in to_keep:
            anim.parents[i] = anim.parents[i] - 1
        anim.parents[i] = to_keep.index(anim.parents[i])

    anim.positions = anim.positions[:, to_keep, :]
    anim.rotations.qs = anim.rotations.qs[:, to_keep, :]
    anim.orients.qs = anim.orients.qs[to_keep, :]
    return anim, to_keep


def parse_bvh_joint_names(bvh_path):
    bvh_file = Path(bvh_path).read_text().split("JOINT")
    return [f.split("\n")[0].strip() for f in bvh_file[1:]]


def get_inp_from_bvh(bvh_path):
    anim, names, ftime = BVH.load(str(bvh_path))
    joint_names = parse_bvh_joint_names(bvh_path)
    anim, to_keep = remap_bvh_anim(anim, joint_names)
    if anim.positions.shape[0] <= 1:
        return None

    joints = Animation.positions_global(anim)
    joints = np.concatenate([joints, joints[-1:]], axis=0)
    new_joints, rotation = process_positions(joints)
    new_joints = new_joints[:, 3:]
    rotation = rotation[:-1]
    anim.rotations[:, 0, :] = rotation[:, 0, :] * anim.rotations[:, 0, :]
    angle = anim.rotations.qs.copy()

    anim.rotations.qs[...] = anim.orients.qs[None]
    tjoints = Animation.positions_global(anim)
    anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
    anim.positions[:, 0, :] = new_joints[:, :3]
    skel = anim.positions.copy()
    return {
        "quat": angle,
        "seq": new_joints,
        "skel": skel,
        "anim": anim,
        "names": names,
        "ftime": ftime,
        "to_keep": to_keep,
    }


def load_shape_vector(shape_npz_path):
    payload = np.load(str(shape_npz_path))
    full_width = payload["full_width"].astype(np.single)
    joint_shape = payload["joint_shape"].astype(np.single)
    return np.divide(joint_shape, full_width[None, :]).reshape(-1)


def load_stats(stats_path):
    stats_path = Path(stats_path)
    out = {key: np.load(stats_path / fname) for key, fname in STATS_FILES.items()}
    out["local_std"] = out["local_std"].copy()
    out["local_std"][out["local_std"] == 0] = 1
    out["quat_std"] = out["quat_std"].copy()
    out["quat_std"][out["quat_std"] == 0] = 1
    return out


def get_height_from_skel(skel):
    diffs = np.sqrt((skel ** 2).sum(axis=-1))
    return (diffs[1:7].sum() + diffs[19:22].sum()) / 100.0


def normalize_motion(quat, local, skel, stats):
    local = (local - stats["local_mean"]) / stats["local_std"]
    skel = (skel - stats["local_mean"]) / stats["local_std"]
    quat = (quat - stats["quat_mean"]) / stats["quat_std"]
    return quat, local, skel


def build_model_inputs(motion, stats, shape_vec, device):
    quat = motion["quat"].astype(np.float32)
    seq = motion["seq"].astype(np.float32)
    skel = motion["skel"].astype(np.float32)

    local = np.reshape(seq[:, :-SEQ_TAIL_DIM], (seq.shape[0], NUM_JOINTS, 3))
    global_part = seq[:, -SEQ_TAIL_DIM:-4]
    height = np.array([get_height_from_skel(skel[0])], dtype=np.float32)

    quat, local, skel = normalize_motion(quat, local, skel, stats)

    seq_feat = np.concatenate([local.reshape(len(local), -1), global_part], axis=-1)
    return {
        "seq": torch.from_numpy(seq_feat)[None].float().to(device),
        "skel": torch.from_numpy(skel.reshape(len(skel), -1))[None].float().to(device),
        "quat": torch.from_numpy(quat)[None].float().to(device),
        "shape": torch.from_numpy(shape_vec.astype(np.float32))[None].float().to(device),
        "height": torch.from_numpy(height)[None].float().to(device),
        "global_raw": global_part,
        "local_norm": local,
    }


def load_retnet(weights_path, ret_model_args, device):
    from src.model_skeleton_aware_smal33 import RetNet

    model = RetNet(**ret_model_args).to(device)
    weights = torch.load(str(weights_path), map_location=device)
    cleaned = OrderedDict()
    for key, val in weights.items():
        cleaned[key.split("module.")[-1]] = val
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def load_shape_retnet(weights_path, ret_model_args, device):
    """Load shape-aware RetNet (stage-2); strict=False for shared stage-1 keys."""
    from src.model_shape_aware_smal33 import RetNet

    model = RetNet(**ret_model_args).to(device)
    weights = torch.load(str(weights_path), map_location=device)
    cleaned = OrderedDict()
    for key, val in weights.items():
        cleaned[key.split("module.")[-1]] = val
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def setup_cuda_device(device_id):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return torch.device("cuda:0")


def expand_anim_frames(anim, num_frames):
    """Resize animation frame count to match model output length."""
    anim = anim.copy()
    cur = anim.positions.shape[0]
    if cur == num_frames:
        return anim
    if cur == 1:
        anim.positions = np.repeat(anim.positions, num_frames, axis=0)
        anim.rotations.qs = np.repeat(anim.rotations.qs, num_frames, axis=0)
        return anim
    idx = np.linspace(0, cur - 1, num_frames).round().astype(int)
    anim.positions = anim.positions[idx]
    anim.rotations.qs = anim.rotations.qs[idx]
    return anim


def get_orient_start_smal33(reference):
    sdr_l, sdr_r, hip_l, hip_r = 8, 12, 18, 22
    across1 = reference[0:1, hip_l] - reference[0:1, hip_r]
    across0 = reference[0:1, sdr_l] - reference[0:1, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    return -rotation


def retarget_to_bvh(
    inp_motion,
    tgt_motion,
    local_rt,
    global_rt,
    quat_rt,
    stats,
    save_dir,
    pair_tag,
    inp_bvh_path=None,
    tgt_bvh_path=None,
):
    from src.utils import put_in_world_bvh

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]

    ours_l = local_rt * local_std + local_mean
    ours_g = global_rt
    ours_total = np.concatenate([ours_l.reshape(len(local_rt), -1), ours_g], axis=-1)
    num_frames = len(ours_total)

    tgt_to_keep = tgt_motion["to_keep"]
    if inp_bvh_path is not None:
        inp_anim, inp_names, inp_ftime = BVH.load(str(inp_bvh_path))
    else:
        inp_anim = inp_motion["anim"].copy()
        inp_names = inp_motion["names"]
        inp_ftime = inp_motion["ftime"]

    if tgt_bvh_path is not None:
        tgt_anim, tgt_names, tgt_ftime = BVH.load(str(tgt_bvh_path))
    else:
        tgt_anim = tgt_motion["anim"].copy()
        tgt_names = tgt_motion["names"]
        tgt_ftime = tgt_motion["ftime"]

    tgt_anim_rest = tgt_anim.copy()
    tgt_anim = expand_anim_frames(tgt_anim, num_frames)

    # Orientation uses remapped SMAL33 joint indices, not the full BVH topology.
    tmp_gt = Animation.positions_global(tgt_motion["anim"])
    start_rots = get_orient_start_smal33(tmp_gt)
    tjoints = tgt_motion["skel"][0:1] * local_std + local_mean
    tjoints = np.repeat(tjoints, num_frames, axis=0)

    output_bvh = ours_total.copy()
    output_bvh[:, -4:] = output_bvh[:, -4:] * (
        np.sign(inp_motion["seq"][:, -8:-4]) * np.sign(output_bvh[:, -4:])
    )
    output_bvh[:, -3][np.abs(inp_motion["seq"][:, -8:-4][:, 2]) <= 1e-2] = 0.0
    output_bvh[:, :3] = tgt_anim.positions[:1, 0, :].copy()

    wjs, rots = put_in_world_bvh(output_bvh.copy(), start_rots)
    tjoints[:, 0, :] = wjs[0, :, 0].copy()

    cquat = quat_rt[:, :NUM_JOINTS].copy()

    inp_copy = save_dir / f"{pair_tag}_input.bvh"
    tgt_copy = save_dir / f"{pair_tag}_target_rest.bvh"
    out_copy = save_dir / f"{pair_tag}_retarget.bvh"

    BVH.save(str(inp_copy), inp_anim, inp_names, inp_ftime)
    BVH.save(str(tgt_copy), tgt_anim_rest, tgt_names, tgt_ftime)

    tgt_anim.positions[:, tgt_to_keep] = tjoints
    tgt_anim.offsets[tgt_to_keep[1:]] = tjoints[0, 1:]
    cquat[:, 0:1, :] = (rots * Quaternions(cquat[:, 0:1, :])).qs
    tgt_anim.rotations.qs[:, tgt_to_keep] = cquat
    BVH.save(str(out_copy), tgt_anim, tgt_names, tgt_ftime)
    return inp_copy, tgt_copy, out_copy


def world_joints_from_motion(local_rt, global_rt, stats, start_rots):
    from src.utils import put_in_world_bvh

    local_mean = stats["local_mean"]
    local_std = stats["local_std"]
    ours_l = local_rt * local_std + local_mean
    ours_total = np.concatenate([ours_l.reshape(len(local_rt), -1), global_rt], axis=-1)
    wjs, _ = put_in_world_bvh(ours_total.copy(), start_rots)
    return wjs[0]


def save_skeleton_video(animations, parents, save_path, interval=33.33):
    from src.utils import animation_plot

    animation_plot(animations, str(save_path), parents, interval=interval)


def list_sequences(data_path, min_frames=32):
    data_path = Path(data_path)
    samples = []
    for char_dir in sorted(p for p in data_path.iterdir() if p.is_dir()):
        for seq_path in sorted(char_dir.glob("*_seq.npy")):
            stem = seq_path.name[: -len("_seq.npy")]
            skel_path = char_dir / f"{stem}_skel.npy"
            quat_path = char_dir / f"{stem}_quat.npy"
            if not skel_path.exists() or not quat_path.exists():
                continue
            seq = np.load(seq_path)
            if seq.shape[0] < min_frames:
                continue
            samples.append(
                {
                    "character": char_dir.name,
                    "sequence": stem,
                    "seq_path": seq_path,
                    "skel_path": skel_path,
                    "quat_path": quat_path,
                    "num_frames": int(seq.shape[0]),
                }
            )
    return samples


def split_characters(samples, val_ratio=0.1, seed=3047):
    chars = sorted({s["character"] for s in samples})
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(len(chars) * val_ratio)))
    val_chars = set(rng.choice(chars, size=n_val, replace=False).tolist())
    train_chars = [c for c in chars if c not in val_chars]
    return train_chars, sorted(val_chars)


def load_window_from_sample(sample, stats, max_length, start_idx=None, rng=None):
    seq = np.load(sample["seq_path"])
    skel = np.load(sample["skel_path"])
    quat = np.load(sample["quat_path"])
    seq_len = seq.shape[0]
    if seq_len > max_length:
        if start_idx is None:
            start_idx = 0 if rng is None else int(rng.randint(0, seq_len - max_length))
        end = start_idx + max_length
    else:
        start_idx = 0
        end = seq_len
    mask_len = end - start_idx

    seq_w = seq[start_idx:end]
    skel_w = skel[start_idx:end]
    quat_w = quat[start_idx:end]

    local = np.reshape(seq_w[:, :-SEQ_TAIL_DIM], (seq_w.shape[0], NUM_JOINTS, 3))
    global_part = seq_w[:, -SEQ_TAIL_DIM:-4]
    quat_n, local_n, skel_n = normalize_motion(quat_w, local, skel_w, stats)

    height = get_height_from_skel(skel_w[0])

    seq_feat = np.concatenate([local_n.reshape(len(local_n), -1), global_part], axis=-1)
    local_gt = local_n.copy()

    return {
        "seq": seq_feat.astype(np.float32),
        "skel": skel_n.reshape(len(skel_n), -1).astype(np.float32),
        "quat": quat_n.astype(np.float32),
        "local_gt": local_gt.astype(np.float32),
        "global_in": global_part.astype(np.float32),
        "height": np.array([height], dtype=np.float32),
        "mask_len": mask_len,
        "character": sample["character"],
        "sequence": sample["sequence"],
    }


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
