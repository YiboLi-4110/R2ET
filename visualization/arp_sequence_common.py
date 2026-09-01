"""Helpers for ARP-only multi-clip sequence export / stitch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def blender_z_up_to_lbs_y_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate Blender Z-up ARP verts into LBS Y-up (proper rotation, det=+1)."""
    verts = np.asarray(vertices, dtype=np.float32)
    out = verts[..., [0, 2, 1]].copy()
    out[..., 2] *= -1.0
    return out


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


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    centers = (verts.min(axis=1) + verts.max(axis=1)) * 0.5
    if anchor == "median":
        ref = np.median(centers, axis=0)
    else:
        ref = centers[0]
    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    return verts - drift[:, None, :], drift


def load_arp_mesh_for_sequence(
    path: Path,
    arp_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ARP mesh cache -> LBS Y-up verts [T,V,3] and faces [F,3]."""
    arp_cfg = arp_cfg or {}
    payload = np.load(str(path))
    if "vertices" not in payload.files:
        raise KeyError(f"ARP mesh cache missing 'vertices': {path}")
    if "faces" not in payload.files:
        raise KeyError(f"ARP mesh cache missing 'faces': {path}")

    verts = payload["vertices"].astype(np.float32)
    faces = payload["faces"].astype(np.int32)
    space = str(payload["space"]) if "space" in payload.files else "blender_world_z_up"
    if space == "lbs_y_up":
        # Already in R2ET/LBS Y-up (CopyQuat / Python direct path).
        pass
    elif space in {
        "blender_world_z_up",
        "mesh_local_zup",
        "armature_local_zup",
        "root_bone_local_zup",
        "world_root_h_locked_zup",
    }:
        verts = blender_z_up_to_lbs_y_up(verts)
    else:
        # Unknown: keep previous behavior for ARP caches.
        verts = blender_z_up_to_lbs_y_up(verts)

    orientation_mode = str(arp_cfg.get("mesh_orientation_correction", "identity")).lower()
    if orientation_mode == "auto":
        # Sequence pipeline has no CopyQuat/Ours reference.
        orientation_mode = "identity"
    if orientation_mode not in ("identity", "yaw_180", "mirror_x", "mirror_z"):
        orientation_mode = "identity"
    verts = apply_orientation_transform(verts, orientation_mode)

    lock_cfg = arp_cfg.get("mesh_lock_horizontal_translation", None)
    if lock_cfg is None:
        # LBS Y-up caches (Source / CopyQuat / R2ET) are already model-space.
        # Default bbox lock here causes limb-driven horizontal jitter (Idle etc.).
        # World-space ARP meshes still default to locked unless already root-locked.
        if space == "lbs_y_up":
            lock_enabled = False
        else:
            source_locked = space in ("root_bone_local_zup", "world_root_h_locked_zup")
            lock_enabled = not source_locked
    else:
        lock_enabled = bool(lock_cfg)
    if lock_enabled:
        anchor = str(arp_cfg.get("mesh_lock_reference", "first")).lower()
        if anchor not in ("first", "median"):
            anchor = "first"
        verts, _ = remove_horizontal_drift(verts, vertical_axis=1, anchor=anchor)

    return verts, faces


def stitch_clips_with_pause(
    clips: list[np.ndarray],
    *,
    pause_frames: int,
    pause_mode: str = "hold_last",
) -> np.ndarray:
    if not clips:
        raise ValueError("No clips to stitch.")
    pause_frames = max(int(pause_frames), 0)
    pause_mode = str(pause_mode).lower()
    if pause_mode not in ("hold_last", "hold_first"):
        raise ValueError(f"Unknown pause_mode '{pause_mode}'")

    parts: list[np.ndarray] = []
    for index, clip in enumerate(clips):
        clip_arr = np.asarray(clip, dtype=np.float32)
        if clip_arr.ndim != 3:
            raise ValueError(f"Clip {index} must be [T,V,3], got {clip_arr.shape}")
        if index > 0 and pause_frames > 0:
            if pause_mode == "hold_last":
                hold = np.repeat(parts[-1][-1:], pause_frames, axis=0)
            else:
                hold = np.repeat(clip_arr[:1], pause_frames, axis=0)
            parts.append(hold)
        parts.append(clip_arr)
    return np.concatenate(parts, axis=0)


def action_label_from_clip(clip: dict[str, Any], index: int = 0) -> str:
    """BVH stem (no suffix) used as on-screen action name."""
    bvh = clip.get("inp_bvh_path")
    if bvh:
        return Path(str(bvh)).stem
    clip_id = clip.get("clip_id")
    if clip_id:
        return Path(str(clip_id)).stem
    return f"clip_{index}"


def build_action_label_segments(
    clips: list[dict[str, Any]],
    *,
    pause_frames: int,
    pause_mode: str = "hold_last",
) -> list[dict[str, Any]]:
    """
    Map stitched timeline to action labels.

    Frame indices are 1-based (Blender). Pause frames inherit the held clip's
    label: previous for hold_last, next for hold_first.
    """
    pause_frames = max(int(pause_frames), 0)
    pause_mode = str(pause_mode).lower()
    if pause_mode not in ("hold_last", "hold_first"):
        raise ValueError(f"Unknown pause_mode '{pause_mode}'")

    segments: list[dict[str, Any]] = []
    cursor = 0
    for index, clip in enumerate(clips):
        label = action_label_from_clip(clip, index)
        n = int(clip["frame_count"])
        if index > 0 and pause_frames > 0:
            if pause_mode == "hold_first":
                pause_label = label
            else:
                pause_label = action_label_from_clip(clips[index - 1], index - 1)
            segments.append(
                {
                    "label": pause_label,
                    "start_frame": cursor + 1,
                    "end_frame": cursor + pause_frames,
                    "kind": "pause",
                }
            )
            cursor += pause_frames
        segments.append(
            {
                "label": label,
                "clip_id": clip.get("clip_id"),
                "inp_bvh_path": clip.get("inp_bvh_path"),
                "start_frame": cursor + 1,
                "end_frame": cursor + n,
                "kind": "clip",
            }
        )
        cursor += n
    return segments


def action_labels_per_frame(
    segments: list[dict[str, Any]],
    frame_count: int,
) -> list[str]:
    labels = [""] * max(int(frame_count), 0)
    for seg in segments:
        text = str(seg.get("label", ""))
        start = max(int(seg["start_frame"]), 1)
        end = min(int(seg["end_frame"]), len(labels))
        for frame in range(start, end + 1):
            labels[frame - 1] = text
    return labels


def build_clip_local_frame_map(
    clip_frame_counts: list[int],
    *,
    pause_frames: int,
    pause_mode: str = "hold_last",
) -> list[tuple[int, int]]:
    """
    Map each stitched timeline frame (0-based) to (clip_index, local_frame).

    Pause frames inherit the held clip: previous for hold_last, next for hold_first.
    """
    pause_frames = max(int(pause_frames), 0)
    pause_mode = str(pause_mode).lower()
    if pause_mode not in ("hold_last", "hold_first"):
        raise ValueError(f"Unknown pause_mode '{pause_mode}'")

    mapping: list[tuple[int, int]] = []
    counts = [int(n) for n in clip_frame_counts]
    for index, n in enumerate(counts):
        if n <= 0:
            raise ValueError(f"Clip {index} has non-positive frame_count={n}")
        if index > 0 and pause_frames > 0:
            if pause_mode == "hold_last":
                hold_clip = index - 1
                hold_local = counts[hold_clip] - 1
            else:
                hold_clip = index
                hold_local = 0
            for _ in range(pause_frames):
                mapping.append((hold_clip, hold_local))
        for local in range(n):
            mapping.append((index, local))
    return mapping


def slugify(text: str) -> str:
    import re

    raw = re.sub(r'[\\/:*?"<>|]+', "_", str(text).strip())
    raw = re.sub(r"\s+", "_", raw)
    return raw or "clip"
