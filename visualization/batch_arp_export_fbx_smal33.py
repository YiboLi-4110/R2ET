#!/usr/bin/env python3
"""
Batch ARP retarget -> one complete skinned FBX per source clip.

Inputs (config):
  - source_dir / source_dirs: one or more directories of paired .bvh + .fbx actions
  - one target character (BVH/FBX with mesh + armature + materials)
  - one ARP .bmap remap preset
  - output directory (each output FBX reuses the source action basename)

Example:
  python visualization/batch_arp_export_fbx_smal33.py \\
    --config config/visualization_arp_export_fbx_smal33.yaml \\
    --blender blender \\
    --arp_addon_modules auto_rig_pro-master

  python visualization/batch_arp_export_fbx_smal33.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_sequence_common import slugify  # noqa: E402
from compare_assets import enrich_case_assets, resolve_fbx_from_bvh  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "config/visualization_arp_export_fbx_smal33.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARP multi-source retarget exporting full skinned FBX files."
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
        "--source_dir",
        type=Path,
        nargs="+",
        default=None,
        help="Override source_dir(s) from config. Accepts one or more directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override fbx output directory from config.",
    )
    parser.add_argument(
        "--remap_preset_path",
        type=Path,
        default=None,
        help="Override arp.remap_preset_path (.bmap file).",
    )
    parser.add_argument(
        "--remap_preset_name",
        type=str,
        default=None,
        help="Override arp.remap_preset_name.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Override top-level job_id.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip clips whose output FBX already exists.",
    )
    parser.add_argument(
        "--blender_threads",
        type=int,
        default=None,
        help="Optional Blender --threads value.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
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
    print(f"[arp-fbx-batch][run] {step}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def normalize_source_dir_list(raw: Any) -> list[Any]:
    """Accept a single path string or a list/tuple of paths."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def resolve_source_dirs(
    cfg: dict[str, Any],
    override: list[Path] | None,
) -> list[Path]:
    if override:
        raw_list = list(override)
    else:
        # Prefer source_dirs; fall back to source_dir (string or list).
        raw_list = normalize_source_dir_list(cfg.get("source_dirs"))
        if not raw_list:
            raw_list = normalize_source_dir_list(cfg.get("source_dir"))

    if not raw_list:
        raise SystemExit(
            "Config source_dir / source_dirs is required "
            "(one or more directories of paired .bvh/.fbx), or pass --source_dir."
        )

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_list:
        source_dir = resolve_repo_path(raw)
        if not source_dir.is_dir():
            raise SystemExit(f"source_dir is not a directory: {source_dir}")
        if source_dir in seen:
            continue
        seen.add(source_dir)
        resolved.append(source_dir)
    return resolved


def resolve_source_entries_from_dir(
    source_dir: Path,
    assets_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan source_dir for .bvh files that have a paired .fbx (same stem)."""
    recursive = bool(assets_cfg.get("recursive_fbx_search", False))
    bvh_files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".bvh" and not path.name.startswith(".")
    )

    resolved: list[dict[str, Any]] = []
    skipped: list[str] = []
    if not bvh_files:
        skipped.append(f"{source_dir}: no .bvh files")
        return resolved, skipped

    for bvh_path in bvh_files:
        try:
            fbx_path = resolve_fbx_from_bvh(bvh_path, recursive_search=recursive)
        except FileNotFoundError as exc:
            skipped.append(f"{bvh_path}: {exc}")
            continue
        # Keep the original source stem so output FBX can reuse the same name.
        clip_id = bvh_path.stem
        resolved.append(
            {
                "clip_id": clip_id,
                "source_stem": bvh_path.stem,
                "source_dir": str(source_dir),
                "inp_bvh_path": str(bvh_path.resolve()),
                "inp_fbx_path": str(fbx_path.resolve()),
            }
        )
    return resolved, skipped


def resolve_source_entries_from_dirs(
    source_dirs: list[Path],
    assets_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Scan one or more source directories; dedupe by source_stem (first wins)."""
    resolved: list[dict[str, Any]] = []
    skipped: list[str] = []
    seen_stems: dict[str, str] = {}

    for source_dir in source_dirs:
        clips, dir_skipped = resolve_source_entries_from_dir(source_dir, assets_cfg)
        skipped.extend(dir_skipped)
        for clip in clips:
            stem = clip["source_stem"]
            if stem in seen_stems:
                skipped.append(
                    f"{clip['inp_bvh_path']}: duplicate stem '{stem}' "
                    f"(already from {seen_stems[stem]}); keeping first"
                )
                continue
            seen_stems[stem] = clip["source_dir"]
            resolved.append(clip)

    if skipped:
        print(
            f"[arp-fbx-batch][warn] skipped/ignored {len(skipped)} entries:",
            flush=True,
        )
        for line in skipped:
            print(f"  - {line}", flush=True)

    if not resolved:
        dirs = ", ".join(str(path) for path in source_dirs)
        raise SystemExit(
            f"No usable source clips in [{dirs}] "
            "(need matching .bvh + .fbx with the same stem)."
        )
    return resolved


def resolve_target(cfg: dict[str, Any]) -> dict[str, str]:
    target = cfg.get("target") or {}
    tgt_bvh = target.get("tgt_bvh_path")
    if not tgt_bvh:
        raise SystemExit("Config target.tgt_bvh_path is required.")
    bvh_path = resolve_repo_path(tgt_bvh)
    if not bvh_path.exists():
        raise SystemExit(f"Target BVH not found: {bvh_path}")

    assets_cfg = cfg.get("assets", {}) or {}
    explicit_fbx = target.get("tgt_fbx_path")
    if explicit_fbx:
        fbx_path = resolve_repo_path(explicit_fbx)
        if not fbx_path.exists():
            raise SystemExit(f"Target FBX not found: {fbx_path}")
    else:
        fbx_path = resolve_fbx_from_bvh(
            bvh_path,
            recursive_search=bool(assets_cfg.get("recursive_fbx_search", False)),
        )
    return {
        "tgt_bvh_path": str(bvh_path),
        "tgt_fbx_path": str(fbx_path),
        "target_stem": Path(fbx_path).stem,
    }


def resolve_output_dir(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        out = override if override.is_absolute() else (_REPO_ROOT / override)
        return out.resolve()
    raw = cfg.get("output_dir")
    if not raw:
        raise SystemExit("Config output_dir is required (or pass --output_dir).")
    return resolve_repo_path(raw)


def build_out_fbx_path(output_dir: Path, source_stem: str) -> Path:
    """Output FBX uses the same basename as the source action file."""
    return output_dir / f"{source_stem}.fbx"


def main():
    args = parse_args()
    cfg_path = args.config.resolve()
    base_cfg = load_yaml(cfg_path)

    job_id = slugify(args.job_id or base_cfg.get("job_id") or "arp_fbx_export")
    arp_cfg = dict(base_cfg.get("arp", {}) or {})
    fbx_cfg = dict(base_cfg.get("fbx_export", {}) or {})
    assets_cfg = dict(base_cfg.get("assets", {}) or {})

    if args.remap_preset_path is not None:
        arp_cfg["remap_preset_path"] = str(resolve_repo_path(args.remap_preset_path))
    elif arp_cfg.get("remap_preset_path"):
        arp_cfg["remap_preset_path"] = str(resolve_repo_path(arp_cfg["remap_preset_path"]))
    if args.remap_preset_name is not None:
        arp_cfg["remap_preset_name"] = args.remap_preset_name

    if not arp_cfg.get("remap_preset_path") and not arp_cfg.get("remap_preset_name"):
        raise SystemExit(
            "Provide arp.remap_preset_path and/or arp.remap_preset_name "
            "(or --remap_preset_path / --remap_preset_name)."
        )

    if arp_cfg.get("remap_preset_path"):
        preset_path = Path(arp_cfg["remap_preset_path"])
        if not preset_path.exists():
            raise SystemExit(f"Remap preset not found: {preset_path}")

    source_dirs = resolve_source_dirs(base_cfg, args.source_dir)
    sources = resolve_source_entries_from_dirs(source_dirs, assets_cfg)
    target = resolve_target(base_cfg)
    output_dir = resolve_output_dir(base_cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = bool(fbx_cfg.get("skip_existing", False)) or args.skip_existing

    cases = []
    planned = []
    for clip in sources:
        out_fbx = build_out_fbx_path(output_dir, clip["source_stem"])
        entry = {
            "clip_id": clip["clip_id"],
            "source_stem": clip["source_stem"],
            "source_dir": clip["source_dir"],
            "inp_bvh_path": clip["inp_bvh_path"],
            "inp_fbx_path": clip["inp_fbx_path"],
            "out_fbx_path": str(out_fbx),
            "skip": bool(skip_existing and out_fbx.exists()),
        }
        planned.append(entry)

        if entry["skip"]:
            continue

        case = {
            "case_id": clip["clip_id"],
            "inp_bvh_path": clip["inp_bvh_path"],
            "inp_fbx_path": clip["inp_fbx_path"],
            "tgt_bvh_path": target["tgt_bvh_path"],
            "tgt_fbx_path": target["tgt_fbx_path"],
            "out_fbx_path": str(out_fbx),
        }
        # Validate FBX pairing early.
        enrich_case_assets(case, assets_cfg)
        cases.append(case)

    summary = {
        "job_id": job_id,
        "source_dirs": [str(path) for path in source_dirs],
        "output_dir": str(output_dir),
        "target": target,
        "remap_preset_path": arp_cfg.get("remap_preset_path"),
        "remap_preset_name": arp_cfg.get("remap_preset_name"),
        "num_clips": len(planned),
        "clips": planned,
        "to_export": [c["case_id"] for c in cases],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    if not cases:
        print("[arp-fbx-batch] nothing to export (all skipped or empty).")
        return

    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = {
        "assets": assets_cfg,
        "arp": arp_cfg,
        "fbx_export": fbx_cfg,
        "cases": cases,
    }
    runtime_cfg_path = work_dir / f"{job_id}_arp_fbx_config.yaml"
    write_yaml(runtime_cfg_path, runtime_cfg)

    run_command(
        [
            *blender_prefix(args.blender, args.blender_threads),
            "--background",
            "--python",
            str(_SCRIPT_DIR / "arp_export_fbx_blender.py"),
            "--",
            "--config",
            str(runtime_cfg_path),
            "--arp_addon_modules",
            *args.arp_addon_modules,
        ],
        cwd=_REPO_ROOT,
        step="arp-export-fbx",
    )

    missing = []
    produced = []
    for case in cases:
        out_path = Path(case["out_fbx_path"])
        if out_path.exists() and out_path.stat().st_size > 0:
            produced.append(str(out_path))
        else:
            missing.append(case["case_id"])

    manifest = {
        "job_id": job_id,
        "source_dirs": [str(path) for path in source_dirs],
        "output_dir": str(output_dir),
        "target": target,
        "remap_preset_path": arp_cfg.get("remap_preset_path"),
        "remap_preset_name": arp_cfg.get("remap_preset_name"),
        "produced": produced,
        "missing": missing,
        "skipped": [c["clip_id"] for c in planned if c["skip"]],
    }
    manifest_path = output_dir / f"{job_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[arp-fbx-batch] manifest: {manifest_path}", flush=True)
    print(f"[arp-fbx-batch] produced {len(produced)} FBX -> {output_dir}", flush=True)
    if missing:
        raise SystemExit(f"Missing FBX for clips: {', '.join(missing)}")


if __name__ == "__main__":
    main()
