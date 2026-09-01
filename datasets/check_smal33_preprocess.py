import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTSIDE_CODE = REPO_ROOT / "outside-code"
if str(OUTSIDE_CODE) not in sys.path:
    sys.path.append(str(OUTSIDE_CODE))

import BVH  # noqa: E402
import Animation  # noqa: E402

from smal33_motion_io import (  # noqa: E402
    AXIS_TRANSFORMS,
    apply_axis_transform_anim,
    estimate_forward,
    get_inp_from_bvh,
)


EXPECTED_PARENTS = np.array(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        6,
        11,
        12,
        13,
        6,
        15,
        16,
        0,
        18,
        19,
        20,
        0,
        22,
        23,
        24,
        0,
        26,
        27,
        28,
        29,
        30,
        31,
    ],
    dtype=np.int64,
)

JOINT_NAMES = [
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

PAW_IDS = np.array([10, 14, 21, 25], dtype=np.int64)
HIND_CONTACT_IDS = np.array([20, 21, 24, 25], dtype=np.int64)
AXIS_NAMES = ["x", "y", "z"]
CONTACT_LEFT_IDS_RAW = np.array([10, 21], dtype=np.int64)
CONTACT_RIGHT_IDS_RAW = np.array([14, 25], dtype=np.int64)
CONTACT_LEFT_IDS_CORRECT = CONTACT_LEFT_IDS_RAW + 1
CONTACT_RIGHT_IDS_CORRECT = CONTACT_RIGHT_IDS_RAW + 1
CONTACT_LEFT_IDS_SHIFTED_BUG = np.array([10, 21], dtype=np.int64)
CONTACT_RIGHT_IDS_SHIFTED_BUG = np.array([14, 25], dtype=np.int64)


def finite_stats(arr):
    return {
        "finite": bool(np.isfinite(arr).all()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


def basic_stats(arr):
    arr = np.asarray(arr)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "mean": float(np.mean(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def per_axis_stats(points):
    flat = points.reshape(-1, 3)
    return {
        axis: {
            "min": float(flat[:, i].min()),
            "max": float(flat[:, i].max()),
            "range": float(flat[:, i].max() - flat[:, i].min()),
        }
        for i, axis in enumerate(AXIS_NAMES)
    }


def infer_floor_axis(global_positions):
    paw_positions = global_positions[:, PAW_IDS, :]
    per_frame_lowest_paw = paw_positions.min(axis=1)
    scores = {}
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        values = per_frame_lowest_paw[:, axis_idx]
        scores[axis_name] = {
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p95_minus_p05": float(np.percentile(values, 95) - np.percentile(values, 5)),
        }
    candidate = min(scores, key=lambda k: scores[k]["p95_minus_p05"])
    return candidate, scores


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def prepare_positions_for_contact(global_positions):
    positions = np.asarray(global_positions, dtype=np.float64).copy()
    paw_positions = positions[:, PAW_IDS, 1]
    floor_height = softmin(np.min(paw_positions, axis=1), softness=0.5, axis=0)
    positions[:, :, 1] -= floor_height
    reference = positions[:, 0]
    return np.concatenate([reference[:, np.newaxis], positions], axis=1)


def compute_contact_channels(prepared_positions, left_ids, right_ids):
    velfactor = 0.15
    heightfactor = np.array([9.0, 6.0], dtype=np.float64)

    feet_l_x = (prepared_positions[1:, left_ids, 0] - prepared_positions[:-1, left_ids, 0]) ** 2
    feet_l_y = (prepared_positions[1:, left_ids, 1] - prepared_positions[:-1, left_ids, 1]) ** 2
    feet_l_z = (prepared_positions[1:, left_ids, 2] - prepared_positions[:-1, left_ids, 2]) ** 2
    feet_l_h = prepared_positions[:-1, left_ids, 1]
    feet_l = (
        ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
    ).astype(np.float32)

    feet_r_x = (prepared_positions[1:, right_ids, 0] - prepared_positions[:-1, right_ids, 0]) ** 2
    feet_r_y = (prepared_positions[1:, right_ids, 1] - prepared_positions[:-1, right_ids, 1]) ** 2
    feet_r_z = (prepared_positions[1:, right_ids, 2] - prepared_positions[:-1, right_ids, 2]) ** 2
    feet_r_h = prepared_positions[:-1, right_ids, 1]
    feet_r = (
        ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
    ).astype(np.float32)

    return np.concatenate([feet_l, feet_r], axis=1)


def compare_contact_semantics(saved_contacts, raw_global):
    if raw_global.shape[0] == 0:
        return {
            "preferred_candidate": "ambiguous",
            "error": "raw_global has zero frames",
        }

    raw_global_ext = np.concatenate([raw_global, raw_global[-1:]], axis=0)
    prepared = prepare_positions_for_contact(raw_global_ext)
    expected_correct = compute_contact_channels(
        prepared, CONTACT_LEFT_IDS_CORRECT, CONTACT_RIGHT_IDS_CORRECT
    )
    expected_shifted_bug = compute_contact_channels(
        prepared, CONTACT_LEFT_IDS_SHIFTED_BUG, CONTACT_RIGHT_IDS_SHIFTED_BUG
    )

    saved_contacts = saved_contacts.astype(np.float32)
    
    if saved_contacts.shape != expected_correct.shape or saved_contacts.shape != expected_shifted_bug.shape:
        return {
            "preferred_candidate": "ambiguous",
            "shape_mismatch": {
                "saved_contacts": list(saved_contacts.shape),
                "expected_correct": list(expected_correct.shape),
                "expected_shifted_bug": list(expected_shifted_bug.shape),
            },
        }
    
    correct_match = saved_contacts == expected_correct
    shifted_match = saved_contacts == expected_shifted_bug

    correct_score = float(correct_match.mean())
    shifted_score = float(shifted_match.mean())

    result = {
        "channel_order": ["left_front", "left_hind", "right_front", "right_hind"],
        "agreement_with_correct_paws": {
            "overall": correct_score,
            "per_channel": [float(x) for x in correct_match.mean(axis=0)],
        },
        "agreement_with_shifted_bug_candidate": {
            "overall": shifted_score,
            "per_channel": [float(x) for x in shifted_match.mean(axis=0)],
        },
        "mean_activation_saved": [float(x) for x in saved_contacts.mean(axis=0)],
        "mean_activation_correct_paws": [float(x) for x in expected_correct.mean(axis=0)],
        "mean_activation_shifted_bug_candidate": [
            float(x) for x in expected_shifted_bug.mean(axis=0)
        ],
        "saved_matches_correct_exactly": bool(np.array_equal(saved_contacts, expected_correct)),
        "saved_matches_shifted_exactly": bool(np.array_equal(saved_contacts, expected_shifted_bug)),
    }

    if shifted_score > correct_score + 0.05:
        result["preferred_candidate"] = "shifted_bug_candidate"
    elif correct_score > shifted_score + 0.05:
        result["preferred_candidate"] = "correct_paws"
    else:
        result["preferred_candidate"] = "ambiguous"

    return result


def canonical_forward_alignment(local_positions, forward_mode="across"):
    """Check whether saved canonical local joints face +Z (Y-up convention)."""
    if local_positions.shape[0] == 0:
        return None
    forward = estimate_forward(local_positions, mode=forward_mode)
    forward = forward / np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
    forward_xz = forward[:, [0, 2]]
    forward_xz = forward_xz / np.maximum(
        np.linalg.norm(forward_xz, axis=-1, keepdims=True), 1e-8
    )
    target_xz = np.array([0.0, 1.0])
    dots = np.sum(forward_xz * target_xz[None, :], axis=-1)
    return {
        "forward_mode": forward_mode,
        "frame0_forward_xz": [float(x) for x in forward_xz[0]],
        "dot_mean": float(dots.mean()),
        "dot_median": float(np.median(dots)),
        "dot_p10": float(np.percentile(dots, 10)),
        "dot_p90": float(np.percentile(dots, 90)),
        "frames_facing_plus_z": int((dots > 0.9).sum()),
        "num_frames": int(dots.shape[0]),
    }


def compare_roundtrip(saved_seq, saved_quat, saved_skel, motion):
    """Compare saved npy arrays with a fresh get_inp_from_bvh() result."""
    diffs = {}
    for name, saved, fresh in (
        ("seq", saved_seq, motion["seq"]),
        ("quat", saved_quat, motion["quat"]),
        ("skel", saved_skel, motion["skel"]),
    ):
        if saved.shape != fresh.shape:
            diffs[name] = {
                "shape_match": False,
                "saved_shape": list(saved.shape),
                "fresh_shape": list(fresh.shape),
            }
            continue
        err = np.abs(saved.astype(np.float64) - fresh.astype(np.float64))
        diffs[name] = {
            "shape_match": True,
            "max": float(err.max()),
            "mean": float(err.mean()),
            "p99": float(np.percentile(err, 99)),
        }
    return diffs


def forward_velocity_alignment(global_positions):
    if global_positions.shape[0] < 3:
        return None

    # Raw BVH indices, before preprocess_q_smal_33.py adds the reference joint.
    left_scapula, right_scapula = 7, 11
    left_thigh, right_thigh = 18, 22
    across = (
        global_positions[:, left_thigh] - global_positions[:, right_thigh]
        + global_positions[:, left_scapula] - global_positions[:, right_scapula]
    )
    norm = np.linalg.norm(across, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-8
    across = np.divide(across, np.maximum(norm, 1e-8))
    forward = np.cross(across, np.array([[0.0, 1.0, 0.0]]))
    forward_norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    valid &= forward_norm[..., 0] > 1e-8
    forward = np.divide(forward, np.maximum(forward_norm, 1e-8))

    root_velocity = global_positions[1:, 0] - global_positions[:-1, 0]
    root_velocity[:, 1] = 0.0
    vel_norm = np.linalg.norm(root_velocity, axis=-1)
    valid_vel = vel_norm > np.percentile(vel_norm, 50)
    valid_pair = valid[1:] & valid_vel
    if valid_pair.sum() == 0:
        return {
            "valid_frames": 0,
            "note": "root motion is too small to judge forward direction",
        }

    velocity_dir = root_velocity[valid_pair] / vel_norm[valid_pair, None]
    dots = np.sum(forward[1:][valid_pair] * velocity_dir, axis=-1)
    return {
        "valid_frames": int(valid_pair.sum()),
        "dot_mean": float(dots.mean()),
        "dot_median": float(np.median(dots)),
        "dot_p10": float(np.percentile(dots, 10)),
        "dot_p90": float(np.percentile(dots, 90)),
    }


def find_bvh_for_seq(seq_path, npy_root, bvh_root):
    if bvh_root is None:
        return None
    rel = seq_path.relative_to(npy_root)
    stem = seq_path.name[: -len("_seq.npy")]
    return bvh_root / rel.parent / f"{stem}.bvh"


def load_triplet(seq_path):
    stem = seq_path.name[: -len("_seq.npy")]
    quat_path = seq_path.with_name(f"{stem}_quat.npy")
    skel_path = seq_path.with_name(f"{stem}_skel.npy")
    missing = [str(p) for p in [quat_path, skel_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing paired files for {seq_path}: {missing}")
    return np.load(seq_path), np.load(quat_path), np.load(skel_path)


def check_one(
    seq_path,
    npy_root,
    bvh_root,
    axis_transform="none",
    forward_mode="across",
    roundtrip=False,
    post_axis_yaw_deg=0.0,
):
    seq, quat, skel = load_triplet(seq_path)
    local_dim = 33 * 3
    expected_seq_dim = local_dim + 4 + 4
    result = {
        "seq_path": str(seq_path),
        "status": "ok",
        "errors": [],
        "warnings": [],
        "shapes": {
            "seq": list(seq.shape),
            "quat": list(quat.shape),
            "skel": list(skel.shape),
        },
        "finite": {
            "seq": finite_stats(seq),
            "quat": finite_stats(quat),
            "skel": finite_stats(skel),
        },
    }

    if seq.ndim != 2 or seq.shape[1] != expected_seq_dim:
        result["errors"].append(f"seq shape should be (T, {expected_seq_dim})")
    if quat.ndim != 3 or quat.shape[1:] != (33, 4):
        result["errors"].append("quat shape should be (T, 33, 4)")
    if skel.ndim != 3 or skel.shape[1:] != (33, 3):
        result["errors"].append("skel shape should be (T, 33, 3)")
    if seq.shape[0] != quat.shape[0] or seq.shape[0] != skel.shape[0]:
        result["errors"].append("seq/quat/skel frame counts do not match")

    if result["errors"]:
        result["status"] = "error"
        return result

    local = seq[:, :local_dim].reshape(seq.shape[0], 33, 3)
    root_motion = seq[:, local_dim : local_dim + 4]
    contacts = seq[:, local_dim + 4 :]

    quat_norm = np.linalg.norm(quat, axis=-1)
    quat_norm_abs_dev = np.abs(quat_norm - 1.0)
    result["quat_norm"] = {
        **basic_stats(quat_norm),
        "abs_dev_max": float(quat_norm_abs_dev.max()),
        "abs_dev_p99": float(np.percentile(quat_norm_abs_dev, 99)),
    }
    if quat_norm_abs_dev.max() > 1e-3:
        result["warnings"].append("quaternion norm deviates from 1 by more than 1e-3")

    result["local_axis"] = per_axis_stats(local)
    result["skel_axis"] = per_axis_stats(skel)
    result["root_motion"] = {
        "vel_x": basic_stats(root_motion[:, 0]),
        "vel_y": basic_stats(root_motion[:, 1]),
        "vel_z": basic_stats(root_motion[:, 2]),
        "rot_y": basic_stats(root_motion[:, 3]),
    }
    result["contacts"] = {
        "shape": list(contacts.shape),
        "mean_per_channel": [float(x) for x in contacts.mean(axis=0)],
        "active_ratio": float(contacts.mean()),
        "unique_values": sorted(float(x) for x in np.unique(contacts)),
    }
    if contacts.shape[1] != 4:
        result["errors"].append("contact channels should be 4")
    if not set(np.unique(contacts)).issubset({0.0, 1.0}):
        result["warnings"].append("contact channels are not binary")

    result["canonical_forward"] = canonical_forward_alignment(local, forward_mode=forward_mode)
    if result["canonical_forward"] is not None:
        cf = result["canonical_forward"]
        if cf["dot_median"] < 0.9:
            result["warnings"].append(
                f"canonical forward (mode={forward_mode}) does not align with +Z; "
                f"frame0 forward_xz={cf['frame0_forward_xz']}"
            )

    # Reconstruct saved local joints from saved skeleton offsets and quaternions.
    bvh_path = find_bvh_for_seq(seq_path, npy_root, bvh_root)
    if bvh_path is not None:
        result["bvh_path"] = str(bvh_path)
        if not bvh_path.exists():
            result["warnings"].append("paired BVH file not found")
        else:
            anim, _, _ = BVH.load(str(bvh_path))
            anim = apply_axis_transform_anim(anim, axis_transform)
            result["bvh"] = {
                "parents_match": bool(np.array_equal(anim.parents, EXPECTED_PARENTS)),
                "parents": anim.parents.tolist(),
                "frames": int(anim.positions.shape[0]),
            }
            if not result["bvh"]["parents_match"]:
                result["errors"].append("BVH parents do not match expected SMAL 33 topology")

            raw_global = Animation.positions_global(anim)
            floor_axis, floor_axis_scores = infer_floor_axis(raw_global)
            result["bvh"]["raw_global_axis"] = per_axis_stats(raw_global)
            result["bvh"]["floor_axis_candidate"] = floor_axis
            result["bvh"]["floor_axis_scores"] = floor_axis_scores
            if floor_axis != "y":
                result["warnings"].append(
                    "lowest-paw stability does not select y as floor axis; verify BVH coordinate system"
                )

            result["bvh"]["forward_velocity_alignment"] = forward_velocity_alignment(raw_global)
            result["contact_semantics"] = compare_contact_semantics(contacts, raw_global)
            preferred = result["contact_semantics"]["preferred_candidate"]
            correct_score = result["contact_semantics"]["agreement_with_correct_paws"]["overall"]
            shifted_score = result["contact_semantics"]["agreement_with_shifted_bug_candidate"][
                "overall"
            ]
            if preferred == "shifted_bug_candidate":
                result["warnings"].append(
                    "saved contact channels align more strongly with the shifted-index bug "
                    "candidate than with the true paw candidate; verify contact indices after "
                    "adding the reference joint"
                )
            elif correct_score < 0.9 and shifted_score < 0.9:
                result["warnings"].append(
                    "saved contact channels do not match either the paw candidate or the "
                    "shifted-index bug candidate very well; inspect contact computation manually"
                )

            recon_anim = anim.copy()
            recon_anim.rotations.qs = quat.copy()
            recon_anim.positions = skel.copy()
            recon_global = Animation.positions_global(recon_anim)
            if recon_global.shape == local.shape:
                err = np.abs(recon_global - local)
                result["fk_reconstruction_error"] = {
                    "max": float(err.max()),
                    "mean": float(err.mean()),
                    "p95": float(np.percentile(err, 95)),
                    "p99": float(np.percentile(err, 99)),
                }
                if err.max() > 1e-3:
                    result["warnings"].append("FK reconstruction max error is larger than 1e-3")
            else:
                result["warnings"].append(
                    f"FK reconstruction shape mismatch: {recon_global.shape} vs {local.shape}"
                )

            if roundtrip:
                try:
                    motion = get_inp_from_bvh(
                        str(bvh_path),
                        axis_transform=axis_transform,
                        forward_mode=forward_mode,
                        post_axis_yaw_deg=post_axis_yaw_deg,
                    )
                    if motion is None:
                        result["warnings"].append(
                            "round-trip get_inp_from_bvh returned None (<=1 frame BVH)"
                        )
                    else:
                        result["preprocess_params"] = {
                            "axis_transform": axis_transform,
                            "forward_mode": forward_mode,
                            "post_axis_yaw_deg": post_axis_yaw_deg,
                            "_axis_transform": motion.get("_axis_transform"),
                            "_forward_mode": motion.get("_forward_mode"),
                            "_post_axis_yaw_deg": motion.get("_post_axis_yaw_deg"),
                        }
                        result["roundtrip_diff"] = compare_roundtrip(
                            seq, quat, skel, motion
                        )
                        for key, diff in result["roundtrip_diff"].items():
                            if not diff.get("shape_match", True):
                                result["errors"].append(
                                    f"round-trip {key} shape mismatch: "
                                    f"{diff.get('saved_shape')} vs {diff.get('fresh_shape')}"
                                )
                            elif diff.get("max", 0.0) > 1e-5:
                                result["warnings"].append(
                                    f"round-trip {key} max diff {diff['max']:.6g} "
                                    f"(preprocess params may not match saved npy)"
                                )
                except Exception as exc:  # noqa: BLE001
                    result["warnings"].append(f"round-trip get_inp_from_bvh failed: {exc!r}")

    if result["errors"]:
        result["status"] = "error"
    elif result["warnings"]:
        result["status"] = "warning"
    return result


def summarize(results):
    total = len(results)
    errors = sum(1 for r in results if r["status"] == "error")
    warnings = sum(1 for r in results if r["status"] == "warning")
    ok = sum(1 for r in results if r["status"] == "ok")
    summary = {
        "total": total,
        "ok": ok,
        "warning": warnings,
        "error": errors,
    }  # type: ignore[assignment]

    if total:
        quat_dev = [
            r.get("quat_norm", {}).get("abs_dev_max")
            for r in results
            if r.get("quat_norm", {}).get("abs_dev_max") is not None
        ]
        if quat_dev:
            summary["quat_abs_dev_max"] = float(max(quat_dev))
        fk_max = [
            r.get("fk_reconstruction_error", {}).get("max")
            for r in results
            if r.get("fk_reconstruction_error", {}).get("max") is not None
        ]
        if fk_max:
            summary["fk_error_max"] = float(max(fk_max))
        floor_axes = [
            r.get("bvh", {}).get("floor_axis_candidate")
            for r in results
            if r.get("bvh", {}).get("floor_axis_candidate") is not None
        ]
        if floor_axes:
            summary["floor_axis_counts"] = {
                axis: floor_axes.count(axis) for axis in sorted(set(floor_axes))
            }
        contact_pref = [
            r.get("contact_semantics", {}).get("preferred_candidate")
            for r in results
            if r.get("contact_semantics", {}).get("preferred_candidate") is not None
        ]
        if contact_pref:
            summary["contact_preferred_candidate_counts"] = {
                pref: contact_pref.count(pref) for pref in sorted(set(contact_pref))
            }
        contact_correct = [
            r.get("contact_semantics", {})
            .get("agreement_with_correct_paws", {})
            .get("overall")
            for r in results
            if r.get("contact_semantics", {})
            .get("agreement_with_correct_paws", {})
            .get("overall")
            is not None
        ]
        if contact_correct:
            summary["contact_agreement_correct_paws"] = basic_stats(contact_correct)
        contact_shifted = [
            r.get("contact_semantics", {})
            .get("agreement_with_shifted_bug_candidate", {})
            .get("overall")
            for r in results
            if r.get("contact_semantics", {})
            .get("agreement_with_shifted_bug_candidate", {})
            .get("overall")
            is not None
        ]
        if contact_shifted:
            summary["contact_agreement_shifted_bug_candidate"] = basic_stats(contact_shifted)
        forward_dots = [
            r.get("canonical_forward", {}).get("dot_median")
            for r in results
            if r.get("canonical_forward", {}).get("dot_median") is not None
        ]
        if forward_dots:
            summary["canonical_forward_dot_median"] = basic_stats(forward_dots)
        roundtrip_max = [
            r.get("roundtrip_diff", {}).get("seq", {}).get("max")
            for r in results
            if r.get("roundtrip_diff", {}).get("seq", {}).get("max") is not None
        ]
        if roundtrip_max:
            summary["roundtrip_seq_diff_max"] = float(max(roundtrip_max))

    return summary


def print_human_report(summary, results, max_items):
    print("=== SMAL 33 preprocess check summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for result in results[:max_items]:
        print(f"[{result['status'].upper()}] {result['seq_path']}")
        print(f"  shapes: {result['shapes']}")
        if result["errors"]:
            print(f"  errors: {result['errors']}")
        if result["warnings"]:
            print(f"  warnings: {result['warnings']}")
        if "quat_norm" in result:
            q = result["quat_norm"]
            print(f"  quat abs dev max/p99: {q['abs_dev_max']:.6g} / {q['abs_dev_p99']:.6g}")
        if "fk_reconstruction_error" in result:
            e = result["fk_reconstruction_error"]
            print(f"  fk error max/mean/p99: {e['max']:.6g} / {e['mean']:.6g} / {e['p99']:.6g}")
        if "contacts" in result:
            print(f"  contact mean: {result['contacts']['mean_per_channel']}")
        if "contact_semantics" in result:
            contact_sem = result["contact_semantics"]
            print(
                "  contact agreement correct/shifted: "
                f"{contact_sem['agreement_with_correct_paws']['overall']:.6g} / "
                f"{contact_sem['agreement_with_shifted_bug_candidate']['overall']:.6g}"
            )
            print(f"  contact preferred candidate: {contact_sem['preferred_candidate']}")
        if "canonical_forward" in result and result["canonical_forward"] is not None:
            cf = result["canonical_forward"]
            print(
                f"  canonical forward ({cf['forward_mode']}): "
                f"frame0_xz={cf['frame0_forward_xz']}, dot_median={cf['dot_median']:.4f}"
            )
        if "roundtrip_diff" in result:
            rt = result["roundtrip_diff"]
            seq_diff = rt.get("seq", {})
            if seq_diff.get("shape_match"):
                print(
                    f"  round-trip seq max/mean: "
                    f"{seq_diff.get('max', float('nan')):.6g} / "
                    f"{seq_diff.get('mean', float('nan')):.6g}"
                )
        if "bvh" in result:
            bvh = result["bvh"]
            print(f"  bvh parents match: {bvh['parents_match']}")
            print(f"  floor axis candidate: {bvh['floor_axis_candidate']}")
            print(f"  forward/root alignment: {bvh['forward_velocity_alignment']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Check SMAL 33 BVH-to-NPY preprocess outputs."
    )
    parser.add_argument(
        "--npy-root",
        required=True,
        type=Path,
        help="Root directory containing *_seq.npy, *_quat.npy, *_skel.npy files.",
    )
    parser.add_argument(
        "--bvh-root",
        type=Path,
        default=None,
        help="Optional root directory containing paired .bvh files with the same relative layout.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N sequences.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable JSON instead of the short human report.",
    )
    parser.add_argument(
        "--max-report-items",
        type=int,
        default=20,
        help="Maximum per-sequence entries shown in the human report.",
    )
    known_axes = ", ".join(sorted(AXIS_TRANSFORMS))
    parser.add_argument(
        "--axis_transform",
        default="none",
        help=(
            "Coordinate transform used during preprocessing (for BVH checks and round-trip). "
            f"Known: {known_axes}. "
            "Not needed for npy-only FK consistency (quat+skel vs seq)."
        ),
    )
    parser.add_argument(
        "--forward_mode",
        choices=["across", "body"],
        default="across",
        help=(
            "Forward estimator used during preprocessing. "
            "Use body for smal@shepherd (matches preprocess_q_smal33.py). "
            "Affects canonical +Z alignment check and round-trip only."
        ),
    )
    parser.add_argument(
        "--post_axis_yaw_deg",
        type=float,
        default=0.0,
        help=(
            "Must match preprocess_q_smal33.py --post_axis_yaw_deg used to build npy "
            "(e.g. 90 for ARP cat_actions / batch2_dogs). Default 0."
        ),
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help=(
            "Re-run get_inp_from_bvh on paired BVH with axis_transform/forward_mode "
            "and compare to saved npy. Requires --bvh-root."
        ),
    )
    args = parser.parse_args()

    npy_root = args.npy_root.resolve()
    bvh_root = args.bvh_root.resolve() if args.bvh_root is not None else None
    seq_paths = sorted(npy_root.rglob("*_seq.npy"))
    if args.limit is not None:
        seq_paths = seq_paths[: args.limit]

    if args.roundtrip and bvh_root is None:
        parser.error("--roundtrip requires --bvh-root")

    results = [
        check_one(
            path,
            npy_root,
            bvh_root,
            axis_transform=args.axis_transform,
            forward_mode=args.forward_mode,
            roundtrip=args.roundtrip,
            post_axis_yaw_deg=args.post_axis_yaw_deg,
        )
        for path in seq_paths
    ]
    summary = summarize(results)
    payload = {"summary": summary, "results": results}

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human_report(summary, results, args.max_report_items)

    if summary["error"] > 0:
        raise SystemExit(2)
    if summary["warning"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
