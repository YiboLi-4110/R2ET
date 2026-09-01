#!/usr/bin/env python3
"""
Export source-action mesh caches for ARP-sequence side-by-side preview.

Two backends:
  * ``lbs`` — R2ET joint-ball LBS (SMAL33 / identity-rest armatures).
  * ``blender`` — Blender depsgraph skin on the source FBX (bone-oriented
    armatures such as sucaibao cat; joint-ball LBS would 拉皮).

``skin_backend: auto`` picks ``lbs`` for SMAL33 shapes and ``blender`` otherwise.
Final caches are written as ``space=lbs_y_up``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_sequence_common import blender_z_up_to_lbs_y_up  # noqa: E402
from compare_assets import resolve_fbx_from_bvh  # noqa: E402
from datasets.lbs_runtime import skin_mesh_sequence  # noqa: E402
from datasets.skeleton_io import (  # noqa: E402
    is_smal33_names,
    joint_names_from_npz,
    parse_bvh_hierarchy_names,
)
from datasets.smal33_motion_io import (  # noqa: E402
    get_inp_from_bvh,
    lbs_rest_skel_from_mesh,
    load_mesh_from_npz,
    report_lbs_rest_skel_mismatch,
)
from export_r2et_dog_actions_smal33 import (  # noqa: E402
    mesh_load_options,
    motion_parse_options,
    resolve_inp_shape_path,
)


def _abs_path(path_like: str | Path, *, base: Path = _PROJECT_ROOT) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def merge_inp_motion_cfg(
    base_motion: dict[str, Any] | None,
    clip_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply optional per-clip inp_* overrides on top of global motion config."""
    motion = dict(base_motion or {})
    clip_entry = clip_entry or {}
    for key in (
        "inp_axis_transform",
        "inp_forward_mode",
        "inp_post_axis_yaw_deg",
        "inp_canonicalize_bind_pose",
    ):
        if key in clip_entry and clip_entry[key] is not None:
            motion[key] = clip_entry[key]
    return motion


def resolve_source_shape_path(
    clip: dict[str, Any],
    cfg: dict[str, Any],
) -> Path:
    explicit = clip.get("inp_shape_path")
    if explicit:
        path = _abs_path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"inp_shape_path not found: {path}")

    source_cfg = cfg.get("source", {}) or {}
    r2et_cfg = cfg.get("r2et", {}) or {}
    raw_root = (
        source_cfg.get("train_shape")
        or source_cfg.get("shape_root")
        or r2et_cfg.get("train_shape")
    )
    if not raw_root:
        raise FileNotFoundError(
            f"Cannot resolve source shape for clip '{clip.get('clip_id')}': "
            "set sources[].inp_shape_path or source.train_shape."
        )
    train_shape = _abs_path(raw_root)
    if not train_shape.is_dir():
        raise FileNotFoundError(f"source shape root not found: {train_shape}")

    bvh_path = Path(clip["inp_bvh_path"])
    shape_path = resolve_inp_shape_path(bvh_path.stem, bvh_path.parent.name, train_shape)
    if shape_path is None:
        raise FileNotFoundError(
            f"Cannot resolve source shape for {bvh_path.stem} "
            f"(folder={bvh_path.parent.name}, train_shape={train_shape}). "
            "Set sources[].inp_shape_path or source.train_shape."
        )
    return Path(shape_path)


def resolve_source_fbx_path(
    clip: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Resolve the animated source FBX used for Blender depsgraph skinning."""
    explicit = clip.get("inp_fbx_path")
    if explicit:
        path = _abs_path(explicit)
        if path.is_file():
            return path
        raise FileNotFoundError(f"inp_fbx_path not found: {path}")

    assets_cfg = (cfg or {}).get("assets", {}) or {}
    return resolve_fbx_from_bvh(
        Path(clip["inp_bvh_path"]),
        recursive_search=bool(assets_cfg.get("recursive_fbx_search", False)),
    )


def resolve_source_skin_backend(
    clip: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    """
    Return ``lbs`` or ``blender``.

    ``source.skin_backend``:
      auto     — SMAL33 shape/BVH → lbs, else blender
      lbs      — force joint-ball LBS
      blender  — force Blender depsgraph skin
    """
    source_cfg = cfg.get("source", {}) or {}
    raw = str(source_cfg.get("skin_backend", "auto") or "auto").strip().lower()
    if raw in ("python", "lbs", "joint_ball"):
        return "lbs"
    if raw in ("blender", "fbx", "depsgraph"):
        return "blender"
    if raw not in ("auto", ""):
        raise ValueError(
            f"Unknown source.skin_backend={raw!r}; expected auto | lbs | blender."
        )

    names = None
    try:
        shape_path = resolve_source_shape_path(clip, cfg)
        names = joint_names_from_npz(shape_path)
    except FileNotFoundError:
        names = None
    if not names:
        bvh = clip.get("inp_bvh_path")
        if bvh:
            names = parse_bvh_hierarchy_names(bvh)
    if is_smal33_names(names):
        return "lbs"
    return "blender"


def remove_horizontal_drift(
    vertices: np.ndarray,
    *,
    vertical_axis: int = 1,
    anchor: str = "first",
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    centers = (verts.min(axis=1) + verts.max(axis=1)) * 0.5
    if anchor == "median":
        ref = np.median(centers, axis=0)
    else:
        ref = centers[0]
    drift = centers - ref
    drift[:, int(vertical_axis)] = 0.0
    return verts - drift[:, None, :]


def rest_mesh_height(vertices: np.ndarray, *, vertical_axis: int = 1) -> float:
    """AABB height of a rest mesh in LBS Y-up (vertical = Y)."""
    return rest_mesh_size(vertices, mode="height", vertical_axis=vertical_axis)


def rest_mesh_size(
    vertices: np.ndarray,
    *,
    mode: str = "length",
    vertical_axis: int = 1,
    forward_axis: int = 2,
) -> float:
    """
    Characteristic rest size for Source↔Target matching.

    Modes (LBS Y-up, +Z forward):
      height     — AABB along vertical (Y)
      length     — AABB along body forward (Z); closest to prior mesh_scale≈3.3
      max_extent — max AABB axis
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] < 3:
        raise ValueError(f"rest vertices must be [V,3], got {verts.shape}")
    span = verts.max(axis=0) - verts.min(axis=0)
    key = str(mode or "length").strip().lower()
    if key in ("height", "y", "vertical"):
        return float(span[int(vertical_axis)])
    if key in ("length", "z", "body", "body_length"):
        return float(span[int(forward_axis)])
    if key in ("max", "max_extent", "extent"):
        return float(span.max())
    raise ValueError(
        f"Unknown match_target_size_mode={mode!r}. "
        "Expected height | length | max_extent."
    )


def scale_verts_about_feet(
    vertices: np.ndarray,
    scale: float,
    *,
    vertical_axis: int = 1,
) -> np.ndarray:
    """
    Uniform scale about frame-0 feet: horizontal pivot = frame-0 bbox center,
    vertical pivot = frame-0 lowest point. Does not use the full-clip AABB, so
    root travel distance is not inflated by the character-size scale.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return verts
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    axis = int(vertical_axis)
    frame0 = verts[0]
    pivot = 0.5 * (frame0.min(axis=0) + frame0.max(axis=0))
    pivot[axis] = float(frame0[:, axis].min())
    return (verts - pivot.astype(np.float32)) * np.float32(scale) + pivot.astype(
        np.float32
    )


def resolve_target_shape_path_for_source(cfg: dict[str, Any]) -> Path:
    """Resolve target shape.npz the same way direct CopyQuat does."""
    from direct_copy_mesh import resolve_target_shape_path  # noqa: E402

    target = cfg.get("target", {}) or {}
    case_cfg = {
        "tgt_bvh_path": target.get("tgt_bvh_path"),
        "tgt_shape_path": target.get("tgt_shape_path"),
    }
    if not case_cfg["tgt_bvh_path"] and not case_cfg["tgt_shape_path"]:
        raise FileNotFoundError(
            "match_target_height needs target.tgt_bvh_path or target.tgt_shape_path "
            "(and usually direct.shape_root)."
        )
    return resolve_target_shape_path(case_cfg, cfg)


def _apply_match_target_height(
    verts: np.ndarray,
    *,
    rest_verts_yup: np.ndarray,
    cfg: dict[str, Any],
    clip_id: str,
    motion_cfg: dict[str, Any],
) -> tuple[np.ndarray, float]:
    source_cfg = cfg.get("source", {}) or {}
    if not bool(source_cfg.get("match_target_height", False)):
        return verts, 1.0

    tgt_shape_path = resolve_target_shape_path_for_source(cfg)
    tgt_mesh = load_mesh_from_npz(
        str(tgt_shape_path), **mesh_load_options(motion_cfg, "tgt")
    )
    size_mode = str(source_cfg.get("match_target_size_mode", "length") or "length")
    h_src = rest_mesh_size(rest_verts_yup, mode=size_mode)
    h_tgt = rest_mesh_size(tgt_mesh["vertices"], mode=size_mode)
    if h_src < 1e-8:
        raise RuntimeError(
            f"Source rest size too small ({h_src}) for match_target_height "
            f"(mode={size_mode}, clip={clip_id})."
        )
    height_scale = float(h_tgt / h_src)
    verts = scale_verts_about_feet(verts, height_scale, vertical_axis=1)
    print(
        f"[source-skin][{clip_id}] match_target_height mode={size_mode} "
        f"scale={height_scale:.4f} (src={h_src:.4f} tgt={h_tgt:.4f} "
        f"file={tgt_shape_path.name})"
    )
    return verts, height_scale


def _finalize_lbs_y_up_cache(
    *,
    output_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    clip: dict[str, Any],
    cfg: dict[str, Any],
    backend: str,
    height_scale: float,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    source_cfg = cfg.get("source", {}) or {}
    lock_cfg = source_cfg.get("mesh_lock_horizontal_translation", False)
    if bool(lock_cfg):
        anchor = str(
            source_cfg.get(
                "mesh_lock_reference",
                (cfg.get("direct", {}) or {}).get("mesh_lock_reference", "first"),
            )
        ).lower()
        if anchor not in ("first", "median"):
            anchor = "first"
        verts = remove_horizontal_drift(verts, vertical_axis=1, anchor=anchor)

    payload = {
        "vertices": np.asarray(verts, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "frame_start": np.int32(0),
        "frame_end": np.int32(max(len(verts) - 1, 0)),
        "space": np.asarray("lbs_y_up"),
        "retarget_mode": np.asarray("source"),
        "source_backend": np.asarray(backend),
        "height_scale": np.float32(height_scale),
        "match_target_height": np.bool_(
            bool(source_cfg.get("match_target_height", False))
        ),
    }
    bvh_path = clip.get("inp_bvh_path")
    if bvh_path:
        payload["inp_bvh_path"] = np.asarray(str(bvh_path))
    if extra_meta:
        for key, value in extra_meta.items():
            payload[key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


@torch.no_grad()
def export_source_mesh_cache_lbs(
    clip: dict[str, Any],
    cfg: dict[str, Any],
    *,
    output_path: Path,
    device,
) -> Path:
    """Skin source BVH onto source shape mesh with joint-ball LBS."""
    motion_cfg = merge_inp_motion_cfg(cfg.get("motion", {}) or {}, clip)
    bvh_path = Path(clip["inp_bvh_path"])
    shape_path = resolve_source_shape_path(clip, cfg)
    clip_id = str(clip.get("clip_id", bvh_path.stem))

    parse_opts = motion_parse_options(motion_cfg, "inp")
    inp_mesh = load_mesh_from_npz(str(shape_path), **mesh_load_options(motion_cfg, "inp"))
    if inp_mesh.get("joint_names"):
        parse_opts["keep_joint_names"] = list(inp_mesh["joint_names"])

    inp_motion = get_inp_from_bvh(str(bvh_path), **parse_opts)
    if inp_motion is None:
        raise RuntimeError(f"Failed to parse source BVH: {bvh_path}")

    rest_skel = lbs_rest_skel_from_mesh(inp_mesh, fallback_skel=inp_motion["skel"][0])
    report_lbs_rest_skel_mismatch(
        rest_skel,
        inp_motion["skel"][0],
        label=f"{clip_id}/source",
        mesh_canonicalized=inp_mesh.get("bind_pose_canonicalized"),
    )

    src_quat = inp_motion["quat"].astype(np.float32)
    verts = skin_mesh_sequence(
        src_quat,
        rest_skel,
        inp_mesh,
        device,
        parents=inp_mesh.get("topology", inp_motion.get("parents")),
    ).astype(np.float32)

    verts, height_scale = _apply_match_target_height(
        verts,
        rest_verts_yup=inp_mesh["vertices"],
        cfg=cfg,
        clip_id=clip_id,
        motion_cfg=motion_cfg,
    )

    source_cfg = cfg.get("source", {}) or {}
    direct_cfg = cfg.get("direct", {}) or {}
    from direct_copy_mesh import (  # noqa: E402
        _parse_apply_root_translation,
        _parse_apply_root_yaw,
        apply_root_motion_to_vertices,
    )

    root_axes = _parse_apply_root_translation(
        source_cfg.get(
            "apply_root_translation",
            direct_cfg.get("apply_root_translation", False),
        )
    )
    apply_yaw = _parse_apply_root_yaw(
        source_cfg.get("apply_root_yaw", direct_cfg.get("apply_root_yaw", False))
    )
    if root_axes is not None or apply_yaw:
        global_vel = inp_motion["seq"][:, -8:-4].astype(np.float32).copy()
        if abs(height_scale - 1.0) > 1e-6:
            global_vel[:, :3] *= np.float32(height_scale)
        verts = apply_root_motion_to_vertices(
            verts, global_vel, axes=root_axes, apply_yaw=apply_yaw
        )

    return _finalize_lbs_y_up_cache(
        output_path=output_path,
        verts=verts,
        faces=inp_mesh["faces"],
        clip=clip,
        cfg=cfg,
        backend="lbs",
        height_scale=height_scale,
        extra_meta={"inp_shape_path": np.asarray(str(shape_path))},
    )


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _postprocess_blender_raw_cache(
    *,
    raw_path: Path,
    output_path: Path,
    clip: dict[str, Any],
    cfg: dict[str, Any],
) -> Path:
    """Convert Blender Z-up raw cache → lbs_y_up + optional size match."""
    motion_cfg = merge_inp_motion_cfg(cfg.get("motion", {}) or {}, clip)
    clip_id = str(clip.get("clip_id", Path(clip.get("inp_bvh_path", "clip")).stem))
    payload = np.load(str(raw_path), allow_pickle=True)
    verts = np.asarray(payload["vertices"], dtype=np.float32)
    faces = np.asarray(payload["faces"], dtype=np.int32)
    space = str(payload["space"]) if "space" in payload.files else "blender_world_z_up"

    if space == "lbs_y_up":
        verts_yup = verts
    else:
        verts_yup = blender_z_up_to_lbs_y_up(verts)

    # Prefer authored rest mesh for size matching when available.
    rest_yup = verts_yup[0]
    shape_meta = {}
    try:
        shape_path = resolve_source_shape_path(clip, cfg)
        inp_mesh = load_mesh_from_npz(
            str(shape_path), **mesh_load_options(motion_cfg, "inp")
        )
        rest_yup = np.asarray(inp_mesh["vertices"], dtype=np.float32)
        shape_meta["inp_shape_path"] = np.asarray(str(shape_path))
        if int(rest_yup.shape[0]) != int(verts_yup.shape[1]):
            print(
                f"[source-skin][{clip_id}][warn] shape V={rest_yup.shape[0]} != "
                f"blender mesh V={verts_yup.shape[1]}; using frame-0 for size match."
            )
            rest_yup = verts_yup[0]
    except FileNotFoundError:
        pass

    verts_yup, height_scale = _apply_match_target_height(
        verts_yup,
        rest_verts_yup=rest_yup,
        cfg=cfg,
        clip_id=clip_id,
        motion_cfg=motion_cfg,
    )

    # Blender FBX already contains authored root motion / lock from export_space.
    # Do NOT re-apply BVH root velocity (would double-count or fight locked space).
    fbx_path = None
    try:
        fbx_path = str(resolve_source_fbx_path(clip, cfg))
    except Exception:
        fbx_path = clip.get("inp_fbx_path")

    extra = dict(shape_meta)
    extra["blender_raw_space"] = np.asarray(space)
    if fbx_path:
        extra["inp_fbx_path"] = np.asarray(str(fbx_path))
    if "root_bone_name" in payload.files:
        extra["root_bone_name"] = payload["root_bone_name"]

    return _finalize_lbs_y_up_cache(
        output_path=output_path,
        verts=verts_yup,
        faces=faces,
        clip=clip,
        cfg=cfg,
        backend="blender",
        height_scale=height_scale,
        extra_meta=extra,
    )


def export_source_mesh_caches_blender(
    clips: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    work_dir: Path,
    blender: str | None = None,
    blender_threads: int | None = None,
    assets_cfg: dict[str, Any] | None = None,
) -> list[Path]:
    """Batch-export Blender-skinned source caches (one Blender process)."""
    if not clips:
        return []

    blender_bin = blender or os.environ.get("BLENDER", "blender")
    work_dir = Path(work_dir)
    raw_dir = work_dir / "source_mesh_raw_blender"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "source_mesh_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = assets_cfg if assets_cfg is not None else (cfg.get("assets", {}) or {})
    source_cfg = dict(cfg.get("source", {}) or {})
    arp_cfg = cfg.get("arp", {}) or {}
    export_space = str(
        source_cfg.get("mesh_export_space")
        or arp_cfg.get("mesh_export_space")
        or "world_root_h_locked_zup"
    )

    cases = []
    planned: list[tuple[dict[str, Any], Path, Path]] = []
    for clip in clips:
        clip = dict(clip)
        clip_id = str(clip["clip_id"])
        fbx_path = resolve_source_fbx_path(clip, {**cfg, "assets": assets})
        clip["inp_fbx_path"] = str(fbx_path)
        raw_path = raw_dir / f"{clip_id}_source_raw.npz"
        out_path = out_dir / f"{clip_id}_source_mesh.npz"
        cases.append(
            {
                "case_id": clip_id,
                "inp_fbx_path": str(fbx_path),
                "inp_bvh_path": clip.get("inp_bvh_path"),
                "source_mesh_path": str(raw_path),
                "mesh_export_space": export_space,
            }
        )
        planned.append((clip, raw_path, out_path))

    runtime_cfg = {
        "source": {
            **source_cfg,
            "mesh_export_space": export_space,
        },
        "arp": {"mesh_export_space": export_space},
        "cases": cases,
    }
    runtime_cfg_path = work_dir / "_source_blender_batch.yaml"
    _write_yaml(runtime_cfg_path, runtime_cfg)

    cmd = [blender_bin]
    if blender_threads is not None and int(blender_threads) > 0:
        cmd.extend(["--threads", str(int(blender_threads))])
    cmd.extend(
        [
            "--background",
            "--python",
            str(_SCRIPT_DIR / "source_skin_mesh_blender.py"),
            "--",
            "--config",
            str(runtime_cfg_path),
        ]
    )
    print(
        f"[source-skin] blender depsgraph export ({len(cases)} clip(s), "
        f"space={export_space}): {' '.join(cmd)}",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)

    outputs = []
    for clip, raw_path, out_path in planned:
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"Blender source raw cache missing for {clip.get('clip_id')}: {raw_path}"
            )
        outputs.append(
            _postprocess_blender_raw_cache(
                raw_path=raw_path,
                output_path=out_path,
                clip=clip,
                cfg=cfg,
            )
        )
        print(
            f"[source-skin][{clip.get('clip_id')}] blender → {out_path}",
            flush=True,
        )
    return outputs


def export_source_mesh_cache(
    clip: dict[str, Any],
    cfg: dict[str, Any],
    *,
    output_path: Path,
    device=None,
    blender: str | None = None,
    blender_threads: int | None = None,
    work_dir: Path | None = None,
) -> Path:
    """Dispatch Source cache export to LBS or Blender backend."""
    backend = resolve_source_skin_backend(clip, cfg)
    clip_id = clip.get("clip_id", Path(clip.get("inp_bvh_path", "clip")).stem)
    print(f"[source-skin][{clip_id}] backend={backend}", flush=True)
    if backend == "lbs":
        if device is None:
            from datasets.smal33_motion_io import setup_cuda_device

            device = setup_cuda_device(int(cfg.get("device", 0)))
        return export_source_mesh_cache_lbs(
            clip, cfg, output_path=output_path, device=device
        )

    wd = Path(work_dir) if work_dir is not None else Path(output_path).parent.parent
    paths = export_source_mesh_caches_blender(
        [clip],
        cfg,
        work_dir=wd,
        blender=blender,
        blender_threads=blender_threads,
    )
    # Ensure the caller's requested path exists (same as default naming).
    final = paths[0]
    if Path(final).resolve() != Path(output_path).resolve():
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.load(str(final), allow_pickle=True)
        np.savez_compressed(output_path, **{k: data[k] for k in data.files})
        return output_path
    return Path(final)


def export_source_mesh_caches(
    clips: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    work_dir: Path,
    device=None,
    blender: str | None = None,
    blender_threads: int | None = None,
) -> dict[str, Path]:
    """Export many Source caches, grouping Blender clips into one process."""
    work_dir = Path(work_dir)
    out: dict[str, Path] = {}
    lbs_clips: list[dict[str, Any]] = []
    blender_clips: list[dict[str, Any]] = []
    for clip in clips:
        backend = resolve_source_skin_backend(clip, cfg)
        if backend == "lbs":
            lbs_clips.append(clip)
        else:
            blender_clips.append(clip)

    if lbs_clips:
        if device is None:
            from datasets.smal33_motion_io import setup_cuda_device

            device = setup_cuda_device(int(cfg.get("device", 0)))
        for clip in lbs_clips:
            clip_id = str(clip["clip_id"])
            path = work_dir / "source_mesh_outputs" / f"{clip_id}_source_mesh.npz"
            out[clip_id] = export_source_mesh_cache_lbs(
                clip, cfg, output_path=path, device=device
            )

    if blender_clips:
        paths = export_source_mesh_caches_blender(
            blender_clips,
            cfg,
            work_dir=work_dir,
            blender=blender,
            blender_threads=blender_threads,
        )
        for clip, path in zip(blender_clips, paths):
            out[str(clip["clip_id"])] = path
    return out
