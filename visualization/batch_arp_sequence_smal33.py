#!/usr/bin/env python3
"""
Batch multi-clip sequence videos via ARP, same-skeleton direct copy, or R2ET.

For one target SMAL33 character and an ordered list of source motions:
  1) Per-clip retarget + mesh cache
       - retarget_mode=arp:    Blender Auto-Rig Pro
       - retarget_mode=direct: CopyQuat (model-space quat copy + LBS; fourway-compatible)
       - retarget_mode=r2et:   R2ET stage1 / stage2 / blend + LBS
  2) Stitch clips with configurable pause frames
  3) LBS render (optional side-by-side source skinned mesh via --show_source)

Example:
  python visualization/batch_arp_sequence_smal33.py \\
    --config config/visualization_arp_sequence_smal33.yaml \\
    --blender blender \\
    --arp_addon_modules auto_rig_pro-master

  # Same-skeleton CopyQuat (fourway-compatible, no ARP addon / no Blender retarget):
  python visualization/batch_arp_sequence_smal33.py \\
    --config config/visualization_arp_sequence_smal33.yaml \\
    --retarget_mode direct \\
    --blender blender

  # R2ET stage-1 (skeleton RetNet):
  python visualization/batch_arp_sequence_smal33.py \\
    --config config/visualization_arp_sequence_smal33.yaml \\
    --retarget_mode r2et \\
    --stage2_mode stage1 \\
    --blender blender

  python visualization/batch_arp_sequence_smal33.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_sequence_common import (  # noqa: E402
    action_label_from_clip,
    build_action_label_segments,
    load_arp_mesh_for_sequence,
    slugify,
    stitch_clips_with_pause,
)
from compare_assets import enrich_case_assets, resolve_fbx_from_bvh  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "config/visualization_arp_sequence_smal33.yaml"
VALID_RETARGET_MODES = ("arp", "direct", "r2et")
VALID_STAGE2_MODES = ("stage1", "stage2", "blend")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-source sequence retarget (ARP / direct / R2ET) + single-animal render."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blender", type=str, default=os.environ.get("BLENDER", "blender"))
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
    )
    parser.add_argument(
        "--retarget_mode",
        type=str,
        default=None,
        choices=list(VALID_RETARGET_MODES),
        help="Override sequence.retarget_mode (arp | direct | r2et).",
    )
    parser.add_argument(
        "--stage2_mode",
        type=str,
        default=None,
        choices=list(VALID_STAGE2_MODES),
        help="R2ET only: override stage2.mode (stage1 | stage2 | blend).",
    )
    parser.add_argument(
        "--gate_scale",
        type=float,
        default=None,
        help="R2ET only: override stage2.gate_scale.",
    )
    parser.add_argument(
        "--stage1_weights",
        type=Path,
        default=None,
        help="R2ET only: override model.stage1_weights.",
    )
    parser.add_argument(
        "--stage2_weights",
        type=Path,
        default=None,
        help="R2ET only: override model.stage2_weights.",
    )
    parser.add_argument(
        "--render_engine",
        type=str,
        default=None,
        choices=["eevee", "cycles"],
        help="Override render.render_engine from config.",
    )
    parser.add_argument(
        "--pause_frames",
        type=int,
        default=None,
        help="Override sequence.pause_frames.",
    )
    parser.add_argument(
        "--pause_mode",
        type=str,
        default=None,
        choices=["hold_last", "hold_first"],
        help="Override sequence.pause_mode.",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Override top-level sequence_id.",
    )
    parser.add_argument(
        "--skip_arp",
        action="store_true",
        default=False,
        help="Skip retarget Blender step (reuse existing mesh caches). Alias for --skip_retarget.",
    )
    parser.add_argument(
        "--skip_retarget",
        action="store_true",
        default=False,
        help="Skip retarget step (reuse existing mesh caches).",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        default=False,
        help="Only run retarget + stitch; do not launch the renderer.",
    )
    parser.add_argument(
        "--show_source",
        action="store_true",
        default=False,
        help="Side-by-side source skinned mesh (overrides sequence/render.show_source=true).",
    )
    parser.add_argument(
        "--no_show_source",
        action="store_true",
        default=False,
        help="Disable source lane even if config enables it.",
    )
    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        default=False,
        help="Keep per-clip mesh/BVH caches under output_root.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=None,
        help="Optional Blender --threads value.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def blender_prefix(blender: str, blender_threads: int | None) -> list[str]:
    cmd = [blender]
    if blender_threads is not None and blender_threads > 0:
        cmd.extend(["--threads", str(blender_threads)])
    return cmd


def run_command(cmd: list[str], *, cwd: Path, step: str):
    print(f"[arp-seq][run] {step}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_retarget_mode(args, seq_cfg: dict[str, Any]) -> str:
    raw = args.retarget_mode or seq_cfg.get("retarget_mode", "arp")
    mode = str(raw).strip().lower()
    if mode not in VALID_RETARGET_MODES:
        raise SystemExit(
            f"Unknown sequence.retarget_mode={raw!r}; expected one of {VALID_RETARGET_MODES}."
        )
    return mode


def apply_r2et_cli_overrides(cfg: dict[str, Any], args) -> dict[str, Any]:
    """Apply R2ET CLI overrides onto a config copy (mutates nothing upstream)."""
    out = dict(cfg)
    if args.stage2_mode is not None:
        out["stage2"] = dict(out.get("stage2", {}) or {})
        out["stage2"]["mode"] = args.stage2_mode
    if args.gate_scale is not None:
        out["stage2"] = dict(out.get("stage2", {}) or {})
        out["stage2"]["gate_scale"] = float(args.gate_scale)
    if args.stage1_weights is not None:
        out["model"] = dict(out.get("model", {}) or {})
        out["model"]["stage1_weights"] = str(args.stage1_weights)
    if args.stage2_weights is not None:
        out["model"] = dict(out.get("model", {}) or {})
        out["model"]["stage2_weights"] = str(args.stage2_weights)
    return out


def resolve_source_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sources = cfg.get("sources") or []
    if not sources:
        raise SystemExit("Config 'sources' list is empty.")
    resolved = []
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            raise SystemExit(f"sources[{index}] must be a mapping.")
        inp_bvh = entry.get("inp_bvh_path")
        if not inp_bvh:
            raise SystemExit(f"sources[{index}] missing inp_bvh_path.")
        bvh_path = Path(inp_bvh)
        if not bvh_path.is_absolute():
            bvh_path = (_REPO_ROOT / bvh_path).resolve()
        else:
            bvh_path = bvh_path.resolve()
        if not bvh_path.exists():
            raise SystemExit(f"Source BVH not found: {bvh_path}")
        clip_id = entry.get("clip_id") or bvh_path.stem
        clip_id = slugify(str(clip_id))
        item = {
            "clip_id": clip_id,
            "inp_bvh_path": str(bvh_path),
            "inp_fbx_path": entry.get("inp_fbx_path"),
        }
        if entry.get("inp_shape_path"):
            shape_path = Path(entry["inp_shape_path"])
            if not shape_path.is_absolute():
                shape_path = (_REPO_ROOT / shape_path).resolve()
            else:
                shape_path = shape_path.resolve()
            if not shape_path.exists():
                raise SystemExit(f"Source shape not found: {shape_path}")
            item["inp_shape_path"] = str(shape_path)
        # Optional per-clip motion overrides (mixed shepherd / Planet Zoo sequences).
        for key in (
            "inp_axis_transform",
            "inp_forward_mode",
            "inp_post_axis_yaw_deg",
            "inp_canonicalize_bind_pose",
        ):
            if key in entry and entry[key] is not None:
                item[key] = entry[key]
        resolved.append(item)
    return resolved


def resolve_target(cfg: dict[str, Any]) -> dict[str, str]:
    target = cfg.get("target") or {}
    tgt_bvh = target.get("tgt_bvh_path")
    if not tgt_bvh:
        raise SystemExit("Config target.tgt_bvh_path is required.")
    bvh_path = Path(tgt_bvh)
    if not bvh_path.is_absolute():
        bvh_path = (_REPO_ROOT / bvh_path).resolve()
    else:
        bvh_path = bvh_path.resolve()
    if not bvh_path.exists():
        raise SystemExit(f"Target BVH not found: {bvh_path}")

    assets_cfg = cfg.get("assets", {}) or {}
    explicit_fbx = target.get("tgt_fbx_path")
    if explicit_fbx:
        fbx_path = Path(explicit_fbx)
        if not fbx_path.is_absolute():
            fbx_path = (_REPO_ROOT / fbx_path).resolve()
        else:
            fbx_path = fbx_path.resolve()
        if not fbx_path.exists():
            raise SystemExit(f"Target FBX not found: {fbx_path}")
    else:
        fbx_path = resolve_fbx_from_bvh(
            bvh_path,
            recursive_search=bool(assets_cfg.get("recursive_fbx_search", False)),
        )
    out = {
        "tgt_bvh_path": str(bvh_path),
        "tgt_fbx_path": str(fbx_path),
    }
    if target.get("tgt_shape_path"):
        shape_path = Path(target["tgt_shape_path"])
        if not shape_path.is_absolute():
            shape_path = (_REPO_ROOT / shape_path).resolve()
        else:
            shape_path = shape_path.resolve()
        if not shape_path.exists():
            raise SystemExit(f"Target shape not found: {shape_path}")
        out["tgt_shape_path"] = str(shape_path)
    return out


def build_case_cfg(
    clip: dict[str, Any],
    target: dict[str, str],
) -> dict[str, Any]:
    case = {
        "case_id": clip["clip_id"],
        "inp_bvh_path": clip["inp_bvh_path"],
        "tgt_bvh_path": target["tgt_bvh_path"],
        "tgt_fbx_path": target["tgt_fbx_path"],
        "arp_bvh_path": None,
    }
    if clip.get("inp_fbx_path"):
        case["inp_fbx_path"] = clip["inp_fbx_path"]
    if clip.get("inp_shape_path"):
        case["inp_shape_path"] = clip["inp_shape_path"]
    if target.get("tgt_shape_path"):
        case["tgt_shape_path"] = target["tgt_shape_path"]
    return case


def build_runtime_config(
    base_cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    work_dir: Path,
    retarget_mode: str,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["output_root"] = str(work_dir)
    cfg["cases"] = cases
    cfg["arp"] = dict(cfg.get("arp", {}) or {})
    cfg["direct"] = dict(cfg.get("direct", {}) or {})
    cfg["r2et"] = dict(cfg.get("r2et", {}) or {})
    cfg["motion"] = dict(cfg.get("motion", {}) or {})
    if retarget_mode == "direct":
        cfg["direct"]["output_dir"] = str(work_dir / "direct_outputs")
        cfg["direct"]["mesh_output_dir"] = str(work_dir / "direct_mesh_outputs")
        # Keep target block for shape_root / tgt_shape_path resolution.
        cfg["target"] = dict(base_cfg.get("target", {}) or {})
        for key in (
            "mesh_orientation_correction",
            "mesh_lock_horizontal_translation",
            "mesh_lock_reference",
            "shape_root",
        ):
            if key not in cfg["direct"] and key in cfg["arp"]:
                cfg["direct"][key] = cfg["arp"][key]
    elif retarget_mode == "r2et":
        cfg["r2et"]["mesh_output_dir"] = str(work_dir / "r2et_mesh_outputs")
        cfg["target"] = dict(base_cfg.get("target", {}) or {})
        cfg["source"] = dict(base_cfg.get("source", {}) or {})
        cfg["model"] = dict(base_cfg.get("model", {}) or {})
        cfg["stage2"] = dict(base_cfg.get("stage2", {}) or {})
        cfg["ret_model_args"] = dict(base_cfg.get("ret_model_args", {}) or {})
        # Reuse direct.shape_root for target shape lookup when r2et.shape_root unset.
        for key in (
            "mesh_orientation_correction",
            "mesh_lock_horizontal_translation",
            "mesh_lock_reference",
            "shape_root",
        ):
            if key not in cfg["r2et"] and key in cfg["direct"]:
                cfg["r2et"][key] = cfg["direct"][key]
            if key not in cfg["r2et"] and key in cfg["arp"]:
                cfg["r2et"][key] = cfg["arp"][key]
    else:
        cfg["arp"]["output_dir"] = str(work_dir / "arp_outputs")
        cfg["arp"]["mesh_output_dir"] = str(work_dir / "arp_mesh_outputs")
        cfg.pop("target", None)
    cfg.pop("sources", None)
    cfg.pop("sequence", None)
    cfg.pop("sequence_id", None)
    return cfg


def clip_mesh_path(work_dir: Path, clip_id: str, retarget_mode: str) -> Path:
    if retarget_mode == "direct":
        return work_dir / "direct_mesh_outputs" / f"{clip_id}_direct_mesh.npz"
    if retarget_mode == "r2et":
        return work_dir / "r2et_mesh_outputs" / f"{clip_id}_r2et_mesh.npz"
    return work_dir / "arp_mesh_outputs" / f"{clip_id}_arp_mesh.npz"


def source_mesh_path(work_dir: Path, clip_id: str) -> Path:
    return work_dir / "source_mesh_outputs" / f"{clip_id}_source_mesh.npz"


def source_cache_is_usable(path: Path, expected_backend: str) -> bool:
    """Reject legacy / wrong-backend Source caches so we don't keep 拉皮 results."""
    if not path.is_file():
        return False
    try:
        payload = np.load(str(path), allow_pickle=True)
    except Exception:
        return False
    if "vertices" not in payload.files:
        return False
    actual = (
        str(payload["source_backend"])
        if "source_backend" in payload.files
        else "lbs"  # pre-tag caches were always joint-ball LBS
    )
    return actual == str(expected_backend)


def resolve_show_source(args, seq_cfg: dict[str, Any], render_cfg: dict[str, Any]) -> bool:
    if bool(getattr(args, "no_show_source", False)):
        return False
    if bool(getattr(args, "show_source", False)):
        return True
    if "show_source" in seq_cfg:
        return bool(seq_cfg.get("show_source"))
    return bool(render_cfg.get("show_source", False))


def retarget_script(retarget_mode: str) -> Path:
    if retarget_mode == "direct":
        return _SCRIPT_DIR / "direct_copy_mesh.py"
    if retarget_mode == "r2et":
        return _SCRIPT_DIR / "r2et_mesh.py"
    return _SCRIPT_DIR / "arp_export_mesh_blender.py"


def run_retarget_export(
    *,
    args,
    base_cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    work_dir: Path,
    retarget_mode: str,
    step_name: str,
    config_name: str,
):
    runtime_cfg = build_runtime_config(base_cfg, cases, work_dir, retarget_mode)
    runtime_cfg_path = work_dir / config_name
    write_yaml(runtime_cfg_path, runtime_cfg)

    if retarget_mode in ("direct", "r2et"):
        # Pure Python paths (CopyQuat / R2ET); no Blender/ARP for mesh export.
        cmd = [
            sys.executable,
            str(retarget_script(retarget_mode)),
            "--config",
            str(runtime_cfg_path),
            "--device",
            str(int(base_cfg.get("device", 0))),
        ]
        if retarget_mode == "r2et":
            if args.stage2_mode is not None:
                cmd.extend(["--stage2_mode", args.stage2_mode])
            if args.gate_scale is not None:
                cmd.extend(["--gate_scale", str(args.gate_scale)])
            if args.stage1_weights is not None:
                cmd.extend(["--stage1_weights", str(args.stage1_weights)])
            if args.stage2_weights is not None:
                cmd.extend(["--stage2_weights", str(args.stage2_weights)])
    else:
        cmd = [
            *blender_prefix(args.blender, args.blender_threads),
            "--background",
            "--python",
            str(retarget_script(retarget_mode)),
            "--",
            "--config",
            str(runtime_cfg_path),
            "--arp_addon_modules",
            *args.arp_addon_modules,
        ]
    run_command(cmd, cwd=_REPO_ROOT, step=step_name)
    return runtime_cfg_path


def _retarget_label(retarget_mode: str, stage2_cfg: dict[str, Any] | None = None) -> str:
    if retarget_mode == "direct":
        return "Direct"
    if retarget_mode == "r2et":
        mode = str((stage2_cfg or {}).get("mode", "blend")).strip().lower() or "blend"
        return f"R2ET-{mode}"
    return "ARP"


def main():
    args = parse_args()
    cfg_path = args.config.resolve()
    base_cfg = load_yaml(cfg_path)
    base_cfg = apply_r2et_cli_overrides(base_cfg, args)

    sequence_id = slugify(args.sequence_id or base_cfg.get("sequence_id") or "arp_sequence")
    seq_cfg = dict(base_cfg.get("sequence", {}) or {})
    retarget_mode = resolve_retarget_mode(args, seq_cfg)
    pause_frames = (
        int(args.pause_frames)
        if args.pause_frames is not None
        else int(seq_cfg.get("pause_frames", 24))
    )
    pause_mode = args.pause_mode or str(seq_cfg.get("pause_mode", "hold_last"))
    skip_retarget = bool(
        seq_cfg.get("skip_arp_if_exists", False)
        or seq_cfg.get("skip_retarget_if_exists", False)
        or args.skip_arp
        or args.skip_retarget
    )

    render_cfg = dict(base_cfg.get("render", {}) or {})
    if args.render_engine is not None:
        render_cfg["render_engine"] = args.render_engine
    camera_cfg = dict(base_cfg.get("camera", {}) or {})
    arp_cfg = dict(base_cfg.get("arp", {}) or {})
    direct_cfg = dict(base_cfg.get("direct", {}) or {})
    r2et_cfg = dict(base_cfg.get("r2et", {}) or {})
    stage2_cfg = dict(base_cfg.get("stage2", {}) or {})
    model_cfg = dict(base_cfg.get("model", {}) or {})
    assets_cfg = dict(base_cfg.get("assets", {}) or {})
    show_source = resolve_show_source(args, seq_cfg, render_cfg)
    render_cfg["show_source"] = show_source
    if show_source and "lane_spacing" not in render_cfg:
        render_cfg["lane_spacing"] = 2.8

    if retarget_mode == "r2et":
        if not (base_cfg.get("ret_model_args") or {}):
            raise SystemExit("retarget_mode=r2et requires ret_model_args in config.")
        if not model_cfg.get("stats_path"):
            raise SystemExit("retarget_mode=r2et requires model.stats_path.")
        mode = str(stage2_cfg.get("mode", "blend")).strip().lower()
        if mode in ("stage1", "skel", "skeleton", "base"):
            if not model_cfg.get("stage1_weights"):
                raise SystemExit(
                    "stage2.mode=stage1 requires model.stage1_weights."
                )
        elif not model_cfg.get("stage2_weights"):
            raise SystemExit(
                f"stage2.mode={mode or 'blend'} requires model.stage2_weights."
            )

    sources = resolve_source_entries(base_cfg)
    target = resolve_target(base_cfg)

    for clip in sources:
        case = build_case_cfg(clip, target)
        enrich_case_assets(case, assets_cfg)

    output_root = Path(base_cfg.get("output_root", "./visualization/videos/arp_sequence"))
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).resolve()
    work_dir = output_root / sequence_id
    work_dir.mkdir(parents=True, exist_ok=True)

    cases = [build_case_cfg(clip, target) for clip in sources]
    runtime_cfg_path = work_dir / f"_{retarget_mode}_batch_config.yaml"
    write_yaml(
        runtime_cfg_path,
        build_runtime_config(base_cfg, cases, work_dir, retarget_mode),
    )

    summary = {
        "sequence_id": sequence_id,
        "retarget_mode": retarget_mode,
        "target": target,
        "sources": [
            {
                "clip_id": c["clip_id"],
                "inp_bvh_path": c["inp_bvh_path"],
                **(
                    {"inp_shape_path": c["inp_shape_path"]}
                    if c.get("inp_shape_path")
                    else {}
                ),
            }
            for c in sources
        ],
        "pause_frames": pause_frames,
        "pause_mode": pause_mode,
        "show_source": show_source,
        "remap_preset_name": arp_cfg.get("remap_preset_name") if retarget_mode == "arp" else None,
        "direct": {
            "scale_root_translation": bool(direct_cfg.get("scale_root_translation", True)),
        }
        if retarget_mode == "direct"
        else None,
        "r2et": {
            "stage2_mode": stage2_cfg.get("mode", "blend"),
            "gate_scale": stage2_cfg.get("gate_scale", 1.0),
            "stage1_weights": model_cfg.get("stage1_weights"),
            "stage2_weights": model_cfg.get("stage2_weights"),
            "stats_path": model_cfg.get("stats_path"),
            "source_train_shape": (base_cfg.get("source", {}) or {}).get("train_shape")
            or r2et_cfg.get("train_shape"),
        }
        if retarget_mode == "r2et"
        else None,
        "work_dir": str(work_dir),
        "runtime_config": str(runtime_cfg_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    missing = [
        c for c in sources if not clip_mesh_path(work_dir, c["clip_id"], retarget_mode).exists()
    ]
    if skip_retarget and not missing:
        print(f"[arp-seq] all {retarget_mode} mesh caches present; skipping retarget step.")
    elif skip_retarget and missing:
        partial_cases = [build_case_cfg(c, target) for c in missing]
        run_retarget_export(
            args=args,
            base_cfg=base_cfg,
            cases=partial_cases,
            work_dir=work_dir,
            retarget_mode=retarget_mode,
            step_name=f"{retarget_mode}-mesh-partial",
            config_name=f"_{retarget_mode}_batch_config_partial.yaml",
        )
    else:
        run_retarget_export(
            args=args,
            base_cfg=base_cfg,
            cases=cases,
            work_dir=work_dir,
            retarget_mode=retarget_mode,
            step_name=f"{retarget_mode}-mesh",
            config_name=f"_{retarget_mode}_batch_config.yaml",
        )

    if show_source:
        from source_skin_mesh import export_source_mesh_caches, resolve_source_skin_backend

        backends = {
            c["clip_id"]: resolve_source_skin_backend(c, base_cfg) for c in sources
        }
        source_missing = [
            c
            for c in sources
            if not source_cache_is_usable(
                source_mesh_path(work_dir, c["clip_id"]),
                backends[c["clip_id"]],
            )
        ]
        to_export = source_missing if skip_retarget else list(sources)
        if skip_retarget and not source_missing:
            print("[arp-seq] all source mesh caches present; skipping source skin step.")
        else:
            backend_counts = {}
            for b in (backends[c["clip_id"]] for c in to_export):
                backend_counts[b] = backend_counts.get(b, 0) + 1
            print(
                f"[arp-seq] exporting source skinned meshes "
                f"({len(to_export)} clip(s); backends={backend_counts})...",
                flush=True,
            )
            # Only spin up CUDA when at least one clip still uses joint-ball LBS.
            device = None
            if any(backends[c["clip_id"]] == "lbs" for c in to_export):
                from datasets.smal33_motion_io import setup_cuda_device

                device = setup_cuda_device(int(base_cfg.get("device", 0)))
            exported = export_source_mesh_caches(
                to_export,
                base_cfg,
                work_dir=work_dir,
                device=device,
                blender=args.blender,
                blender_threads=args.blender_threads,
            )
            for clip_id, out_path in exported.items():
                print(
                    f"[arp-seq] source mesh {clip_id}: {out_path} "
                    f"(backend={backends.get(clip_id)})",
                    flush=True,
                )

    mesh_load_cfg = dict(arp_cfg)
    if retarget_mode == "direct":
        mesh_load_cfg.update(direct_cfg)
    elif retarget_mode == "r2et":
        mesh_load_cfg.update(r2et_cfg)

    source_cfg = dict(base_cfg.get("source", {}) or {})
    # Source lane load knobs (default: no bbox horizontal lock — avoids Idle jitter).
    source_mesh_load_cfg = {
        "mesh_orientation_correction": source_cfg.get(
            "mesh_orientation_correction", "identity"
        ),
        "mesh_lock_horizontal_translation": bool(
            source_cfg.get("mesh_lock_horizontal_translation", False)
        ),
        "mesh_lock_reference": source_cfg.get("mesh_lock_reference", "first"),
    }
    # Keep renderer in sync even if only render.* is consulted later.
    render_cfg["source_mesh_lock_horizontal_translation"] = bool(
        source_mesh_load_cfg["mesh_lock_horizontal_translation"]
    )
    render_cfg["source_mesh_lock_reference"] = source_mesh_load_cfg["mesh_lock_reference"]

    clip_verts: list[np.ndarray] = []
    faces = None
    clip_meta = []
    for clip in sources:
        mesh_path = clip_mesh_path(work_dir, clip["clip_id"], retarget_mode)
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Missing {retarget_mode} mesh cache after export: {mesh_path}"
            )
        verts, clip_faces = load_arp_mesh_for_sequence(mesh_path, mesh_load_cfg)
        if faces is None:
            faces = clip_faces
        elif faces.shape != clip_faces.shape or not np.array_equal(faces, clip_faces):
            raise RuntimeError(
                f"Face topology mismatch for clip '{clip['clip_id']}'. "
                "All clips must share the same target mesh."
            )
        meta = {
            "clip_id": clip["clip_id"],
            "inp_bvh_path": clip["inp_bvh_path"],
            "action_label": action_label_from_clip(clip),
            "frame_count": int(verts.shape[0]),
            "mesh_path": str(mesh_path),
            "retarget_mode": retarget_mode,
        }
        if show_source:
            from source_skin_mesh import resolve_source_shape_path, resolve_source_skin_backend

            src_path = source_mesh_path(work_dir, clip["clip_id"])
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source mesh cache: {src_path}")
            src_verts, _src_faces = load_arp_mesh_for_sequence(
                src_path, source_mesh_load_cfg
            )
            if int(src_verts.shape[0]) != int(verts.shape[0]):
                raise RuntimeError(
                    f"Source/retarget frame mismatch for '{clip['clip_id']}': "
                    f"source_T={src_verts.shape[0]} retarget_T={verts.shape[0]}. "
                    "Re-export both with the same source FBX frame range "
                    "(delete clip mesh caches and rerun without --skip_retarget)."
                )
            src_fbx = clip.get("inp_fbx_path")
            if not src_fbx:
                try:
                    src_fbx = str(
                        resolve_fbx_from_bvh(
                            Path(clip["inp_bvh_path"]),
                            recursive_search=bool(
                                assets_cfg.get("recursive_fbx_search", False)
                            ),
                        )
                    )
                except Exception:
                    src_fbx = None
            meta["source_mesh_path"] = str(src_path)
            meta["source_fbx_path"] = src_fbx
            meta["source_backend"] = resolve_source_skin_backend(clip, base_cfg)
            try:
                meta["source_shape_path"] = str(
                    resolve_source_shape_path(clip, base_cfg)
                )
            except FileNotFoundError:
                meta["source_shape_path"] = None
            meta["source_vertex_count"] = int(src_verts.shape[1])
        clip_verts.append(verts)
        clip_meta.append(meta)
        print(
            f"[arp-seq] loaded {clip['clip_id']}: T={verts.shape[0]} V={verts.shape[1]}"
            + (
                f" source_V={meta.get('source_vertex_count')}"
                if show_source
                else ""
            ),
            flush=True,
        )

    assert faces is not None
    stitched = stitch_clips_with_pause(
        clip_verts,
        pause_frames=pause_frames,
        pause_mode=pause_mode,
    )
    fps = int(render_cfg.get("fps", 24))
    npz_path = work_dir / "arp_sequence.npz"
    np.savez_compressed(
        npz_path,
        arp_vertices=stitched.astype(np.float32),
        target_faces=faces.astype(np.int32),
        fps=np.int32(fps),
        frame_count=np.int32(stitched.shape[0]),
        pause_frames=np.int32(pause_frames),
        camera_zoom=np.float32(float(camera_cfg.get("camera_zoom", 1.0))),
        view_elev=np.float32(float(camera_cfg.get("view_elev", 18.0))),
        view_azim=np.float32(float(camera_cfg.get("view_azim", -75.0))),
        retarget_mode=np.asarray(retarget_mode),
    )
    print(
        f"[arp-seq] stitched frames={stitched.shape[0]} "
        f"(clips={len(clip_verts)}, pause_frames={pause_frames}, mode={pause_mode}) "
        f"-> {npz_path}",
        flush=True,
    )

    label = _retarget_label(retarget_mode, stage2_cfg)
    action_segments = build_action_label_segments(
        clip_meta,
        pause_frames=pause_frames,
        pause_mode=pause_mode,
    )
    manifest = {
        "sequence_id": sequence_id,
        "npz_path": str(npz_path),
        "label": label,
        "retarget_mode": retarget_mode,
        "target_bvh_path": target["tgt_bvh_path"],
        "target_fbx_path": target["tgt_fbx_path"],
        "source_clips": clip_meta,
        "action_segments": action_segments,
        "show_source": show_source,
        "pause_frames": pause_frames,
        "pause_mode": pause_mode,
        "fps": fps,
        "frame_count": int(stitched.shape[0]),
        "camera": camera_cfg,
        "render": render_cfg,
        "source": {
            "train_shape": source_cfg.get("train_shape"),
            "mesh_lock_horizontal_translation": bool(
                source_mesh_load_cfg["mesh_lock_horizontal_translation"]
            ),
            "mesh_lock_reference": source_mesh_load_cfg["mesh_lock_reference"],
            "mesh_orientation_correction": source_mesh_load_cfg[
                "mesh_orientation_correction"
            ],
        }
        if show_source
        else None,
        "arp": {
            "remap_preset_name": arp_cfg.get("remap_preset_name"),
            "mesh_orientation_correction": arp_cfg.get("mesh_orientation_correction"),
            "mesh_export_space": arp_cfg.get("mesh_export_space"),
        }
        if retarget_mode == "arp"
        else None,
        "direct": {
            "scale_root_translation": bool(direct_cfg.get("scale_root_translation", True)),
            "mesh_export_space": direct_cfg.get(
                "mesh_export_space", arp_cfg.get("mesh_export_space")
            ),
            "mesh_orientation_correction": direct_cfg.get(
                "mesh_orientation_correction",
                arp_cfg.get("mesh_orientation_correction"),
            ),
        }
        if retarget_mode == "direct"
        else None,
        "r2et": {
            "stage2_mode": stage2_cfg.get("mode", "blend"),
            "gate_scale": stage2_cfg.get("gate_scale", 1.0),
            "stage1_weights": model_cfg.get("stage1_weights"),
            "stage2_weights": model_cfg.get("stage2_weights"),
            "stats_path": model_cfg.get("stats_path"),
            "mesh_orientation_correction": r2et_cfg.get(
                "mesh_orientation_correction",
                direct_cfg.get(
                    "mesh_orientation_correction",
                    arp_cfg.get("mesh_orientation_correction"),
                ),
            ),
        }
        if retarget_mode == "r2et"
        else None,
    }
    manifest_path = work_dir / "arp_sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[arp-seq] manifest: {manifest_path}", flush=True)

    if args.skip_render:
        print("[arp-seq] skip_render set; done after stitch.")
        return

    run_command(
        [
            *blender_prefix(args.blender, args.blender_threads),
            "--background",
            "--python",
            str(_SCRIPT_DIR / "render_arp_sequence_lbs_blender.py"),
            "--",
            "--manifest",
            str(manifest_path),
            "--render_engine",
            str(render_cfg.get("render_engine", "eevee")),
        ],
        cwd=_REPO_ROOT,
        step="render-arp-seq",
    )

    final_video = work_dir / "arp_sequence.mp4"
    if final_video.exists():
        print(f"[arp-seq] done: {final_video}", flush=True)
    else:
        raise SystemExit(f"Expected output video missing: {final_video}")

    if not args.keep_intermediates:
        for sub in (
            "arp_outputs",
            "arp_mesh_outputs",
            "direct_outputs",
            "direct_mesh_outputs",
            "r2et_mesh_outputs",
            "source_mesh_outputs",
        ):
            path = work_dir / sub
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        for cfg_name in (
            "_arp_batch_config.yaml",
            "_arp_batch_config_partial.yaml",
            "_direct_batch_config.yaml",
            "_direct_batch_config_partial.yaml",
            "_r2et_batch_config.yaml",
            "_r2et_batch_config_partial.yaml",
        ):
            cfg_file = work_dir / cfg_name
            if cfg_file.exists():
                cfg_file.unlink()


if __name__ == "__main__":
    main()
