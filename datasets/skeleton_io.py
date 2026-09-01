"""Skeleton-agnostic BVH helpers for Source LBS / visualization.

SMAL33 keeps its original remap + hardcoded landmark indices in
``smal33_motion_io``. This module is the non-33 path (sucaibao cat, etc.)
and the shared name/landmark utilities used to decide which path to take.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Duplicated from smal33_motion_io so this module can be imported there
# without a circular import. Keep in sync with JOINTS_LIST.
SMAL33_BODY_JOINT_NAMES = [
    "Spine1", "Spine2", "Spine3", "Spine4", "Spine5", "Spine6",
    "LeftScapula", "LeftUpperArm", "LeftForeLeg", "LeftFrontPaw",
    "RightScapula", "RightUpperArm", "RightForeLeg", "RightFrontPaw",
    "Neck", "Head", "Jaw",
    "LeftThigh", "LeftShin", "LeftHock", "LeftHindPaw",
    "RightThigh", "RightShin", "RightHock", "RightHindPaw",
    "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6", "Tail7",
]
SMAL33_JOINT_NAMES = ["Root"] + list(SMAL33_BODY_JOINT_NAMES)
NUM_JOINTS_SMAL33 = 33

# Bind-pose / facing landmarks. SMAL33 names are listed first so a generic
# lookup on a remapped 33-joint clip still hits LeftUpperArm/LeftThigh
# (the same indices the historical SMAL33 code uses after dummy-root insert).
SHOULDER_L_ALIASES = (
    "LeftUpperArm",
    "LeftScapula",
    "shoulder_blade.L",
    "hip_f.L",
)
SHOULDER_R_ALIASES = (
    "RightUpperArm",
    "RightScapula",
    "shoulder_blade.R",
    "hip_f.R",
)
HIP_L_ALIASES = ("LeftThigh", "hip_b.L")
HIP_R_ALIASES = ("RightThigh", "hip_b.R")
FOOT_L_ALIASES = (
    ("LeftFrontPaw", "foot_f.L", "claw_f.L"),
    ("LeftHindPaw", "foot_b.L", "claw_b.L"),
)
FOOT_R_ALIASES = (
    ("RightFrontPaw", "foot_f.R", "claw_f.R"),
    ("RightHindPaw", "foot_b.R", "claw_b.R"),
)


def decode_name_list(values) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return [str(values)]
    arr = np.asarray(values)
    if arr.dtype == object or arr.ndim == 1:
        return [str(x) for x in arr.tolist()]
    return [str(x) for x in arr.reshape(-1).tolist()]


def normalize_joint_name(name: str) -> str:
    text = str(name).split(":")[-1].strip().lower()
    return text.replace(" ", "")


def parse_bvh_hierarchy_names(bvh_path: str | Path) -> list[str]:
    """ROOT + JOINT names in file order, including dotted names like ear_1.L."""
    names: list[str] = []
    for line in Path(bvh_path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
            names.append(stripped.split(None, 1)[1])
    return names


def is_smal33_names(names: Sequence[str] | None) -> bool:
    if not names:
        return False
    hits = 0
    for jname in SMAL33_BODY_JOINT_NAMES:
        if any(jname == name[-len(jname) :] for name in names):
            hits += 1
    return hits == len(SMAL33_BODY_JOINT_NAMES)


def joint_names_from_npz(npz_path: str | Path) -> list[str] | None:
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "joint_names" not in data.files:
            return None
        return decode_name_list(data["joint_names"])


def topology_from_npz(npz_path: str | Path) -> np.ndarray | None:
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "topology" not in data.files and "parents" not in data.files:
            return None
        key = "topology" if "topology" in data.files else "parents"
        return np.asarray(data[key], dtype=np.int64)


def find_joint_index(
    names: Sequence[str],
    candidates: Iterable[str],
    *,
    used: set[int] | None = None,
) -> int | None:
    used = used or set()
    normalized = [normalize_joint_name(n) for n in names]
    for cand in candidates:
        want = str(cand)
        want_norm = normalize_joint_name(want)
        for idx, name in enumerate(names):
            if idx in used:
                continue
            if name == want or name[-len(want) :] == want:
                return idx
        for idx, name_norm in enumerate(normalized):
            if idx in used:
                continue
            if name_norm == want_norm or name_norm.replace("_", ".") == want_norm.replace(
                "_", "."
            ):
                return idx
    return None


def resolve_landmark_indices(names: Sequence[str]) -> dict[str, np.ndarray | int]:
    """Original (pre dummy-root) landmark indices for facing / floor."""
    sdr_l = find_joint_index(names, SHOULDER_L_ALIASES)
    sdr_r = find_joint_index(names, SHOULDER_R_ALIASES)
    hip_l = find_joint_index(names, HIP_L_ALIASES)
    hip_r = find_joint_index(names, HIP_R_ALIASES)
    missing = [
        label
        for label, idx in (
            ("shoulder_l", sdr_l),
            ("shoulder_r", sdr_r),
            ("hip_l", hip_l),
            ("hip_r", hip_r),
        )
        if idx is None
    ]
    if missing:
        raise ValueError(
            f"Cannot resolve facing landmarks {missing} from joints: {list(names)}"
        )

    def resolve_feet(alias_pairs) -> np.ndarray:
        found: list[int] = []
        used: set[int] = set()
        for aliases in alias_pairs:
            idx = find_joint_index(names, aliases, used=used)
            if idx is not None:
                found.append(idx)
                used.add(idx)
        if not found:
            raise ValueError(f"Cannot resolve foot landmarks from joints: {list(names)}")
        if len(found) == 1:
            found = found + found
        return np.asarray(found[:2], dtype=np.int64)

    return {
        "sdr_l": int(sdr_l),
        "sdr_r": int(sdr_r),
        "hip_l": int(hip_l),
        "hip_r": int(hip_r),
        "foot_l": resolve_feet(FOOT_L_ALIASES),
        "foot_r": resolve_feet(FOOT_R_ALIASES),
    }


def match_keep_indices(file_names: Sequence[str], keep_names: Sequence[str]) -> list[int]:
    used: set[int] = set()
    to_keep: list[int] = []
    for want in keep_names:
        idx = find_joint_index(file_names, [want], used=used)
        if idx is None:
            raise ValueError(
                f"Joint {want!r} not found in BVH joints: {list(file_names)}"
            )
        to_keep.append(idx)
        used.add(idx)
    return to_keep


def subset_animation_joints(anim, to_keep: Sequence[int]):
    """Keep joints in ``to_keep`` order; parents of dropped bones walk up."""
    keep = [int(i) for i in to_keep]
    old_parents = np.asarray(anim.parents).copy()
    old_to_new = {old: new for new, old in enumerate(keep)}
    new_parents = np.full(len(keep), -1, dtype=old_parents.dtype)
    for new_i, old_i in enumerate(keep):
        if new_i == 0:
            new_parents[new_i] = -1
            continue
        parent = int(old_parents[old_i])
        while parent >= 0 and parent not in old_to_new:
            parent = int(old_parents[parent])
        new_parents[new_i] = old_to_new.get(parent, -1) if parent >= 0 else -1
        if int(new_parents[new_i]) == new_i:
            new_parents[new_i] = -1

    anim.parents = new_parents
    anim.positions = anim.positions[:, keep, :]
    anim.rotations.qs = anim.rotations.qs[:, keep, :]
    anim.orients.qs = anim.orients.qs[keep, :]
    if getattr(anim, "offsets", None) is not None:
        offsets = np.asarray(anim.offsets)
        if offsets.shape[0] == len(old_parents):
            anim.offsets = offsets[keep]
    return anim, keep


def remap_anim_to_names(anim, file_names: Sequence[str], keep_names: Sequence[str]):
    to_keep = match_keep_indices(file_names, keep_names)
    return subset_animation_joints(anim, to_keep)


def parents_from_mesh_or_skel(mesh_data=None, rest_skel=None, parents=None) -> np.ndarray:
    if parents is not None:
        return np.asarray(parents, dtype=np.int64)
    if mesh_data is not None:
        for key in ("topology", "parents"):
            value = mesh_data.get(key)
            if value is not None:
                return np.asarray(value, dtype=np.int64)
    try:
        from smal33_motion_io import NUM_JOINTS, SMAL33_PARENTS
    except ImportError:
        from datasets.smal33_motion_io import NUM_JOINTS, SMAL33_PARENTS

    if rest_skel is None:
        return np.asarray(SMAL33_PARENTS, dtype=np.int64)
    joint_count = int(np.asarray(rest_skel).reshape(-1, 3).shape[0])
    if joint_count == NUM_JOINTS:
        return np.asarray(SMAL33_PARENTS, dtype=np.int64)
    raise ValueError(
        f"Need mesh topology/parents for a {joint_count}-joint skeleton "
        f"(SMAL33 fallback only applies when J={NUM_JOINTS})."
    )
