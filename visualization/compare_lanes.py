"""Shared compare-lane helpers (no Blender/bpy dependency)."""

from __future__ import annotations

from typing import Any


def include_copyquat_lane(render_cfg: dict[str, Any] | None) -> bool:
    render_cfg = render_cfg or {}
    return bool(render_cfg.get("include_copyquat", True))


def compare_lane_titles(render_cfg: dict[str, Any] | None) -> list[str]:
    titles = ["Source", "R2ET", "ARP"]
    if include_copyquat_lane(render_cfg):
        titles.append("CopyQuat")
    return titles


def lane_layout_offset(lane_index: int, num_lanes: int, lane_spacing: float) -> float:
    return (lane_index - (num_lanes - 1) / 2.0) * float(lane_spacing)
