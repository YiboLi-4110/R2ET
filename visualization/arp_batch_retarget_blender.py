#!/usr/bin/env python3
"""
Batch retarget with Auto-Rig Pro (ARP) inside Blender.

This script is designed for unified-parameter baseline generation.
It provides:
  - addon availability check
  - optional addon enable attempts
  - batch import(source BVH + target rest BVH) / ARP operator chain / BVH export

Run:
  blender --background --python visualization/arp_batch_retarget_blender.py -- \
      --config config/visualization_compare_smal33.yaml --check_only

If --background still fails ARP poll checks, try without --background on a display
or via xvfb-run (see README_compare_pipeline.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from arp_blender_common import (
    DEFAULT_REMAP_PRESET,
    assign_arp_rigs,
    assert_exists,
    blender_argv,
    call_arp_operator,
    clean_scene,
    ensure_blender_ui_context,
    get_action_frame_range,
    import_bvh_armature,
    import_fbx_armature,
    install_remap_preset,
    list_arp_ops,
    load_cfg,
    pick_armature_by_hint,
    print_arp_diagnostics,
    reset_armature_object_transform,
    resolve_operator_sequence,
    try_enable_addons,
)
from compare_assets import enrich_case_assets


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Batch ARP retarget helper.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--case_ids", type=str, nargs="+", default=None)
    parser.add_argument("--check_only", action="store_true", default=False)
    parser.add_argument("--save_userpref", action="store_true", default=False)
    parser.add_argument(
        "--arp_addon_modules",
        type=str,
        nargs="+",
        default=["auto_rig_pro-master", "auto_rig_pro"],
        help="Addon module names to try enabling in Blender preferences.",
    )
    parser.add_argument(
        "--operator_sequence",
        type=str,
        nargs="+",
        default=None,
        help="ARP bpy.ops.arp operator sequence without prefix. "
        "Defaults to config arp.operator_sequence.",
    )
    return parser.parse_args(argv)


def run_case(case_cfg, cfg, op_sequence):
    case_id = case_cfg["case_id"]
    arp_cfg = cfg.get("arp", {})
    clean_scene()

    inp_bvh = assert_exists(case_cfg["inp_bvh_path"], "inp_bvh_path")
    tgt_bvh = assert_exists(case_cfg["tgt_bvh_path"], "tgt_bvh_path")
    assert_exists(case_cfg["tgt_fbx_path"], "tgt_fbx_path")

    source_arm = import_bvh_armature(inp_bvh)
    if arp_cfg.get("use_target_bvh", True):
        target_arm = import_bvh_armature(tgt_bvh)
    else:
        target_arm = import_fbx_armature(Path(case_cfg["tgt_fbx_path"]))

    source_hint = arp_cfg.get("source_armature_hint")
    target_hint = arp_cfg.get("target_armature_hint")
    if source_hint:
        source_arm = pick_armature_by_hint(source_hint, exclude_names={target_arm.name})
    if target_hint:
        target_arm = pick_armature_by_hint(target_hint, exclude_names={source_arm.name})

    reset_armature_object_transform(source_arm)
    reset_armature_object_transform(target_arm)
    assign_arp_rigs(source_arm, target_arm)

    frame_start, frame_end = get_action_frame_range(source_arm)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    print(f"[arp][{case_id}] frame range: {frame_start}-{frame_end}")

    for op_name in op_sequence:
        result = call_arp_operator(
            op_name,
            target_arm,
            arp_cfg,
            frame_range=(frame_start, frame_end),
        )
        print(f"[arp][{case_id}] bpy.ops.arp.{op_name} -> {result}")

    out_dir = Path(arp_cfg.get("output_dir", "./visualization/videos/compare/arp_outputs"))
    suffix = arp_cfg.get("output_suffix", "_arp_retarget.bvh")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}{suffix}"

    bpy.ops.object.select_all(action="DESELECT")
    target_arm.select_set(True)
    bpy.context.view_layer.objects.active = target_arm
    bpy.ops.export_anim.bvh(
        filepath=str(out_path),
        check_existing=False,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=False,
    )
    print(f"[arp][{case_id}] exported: {out_path}")
    return out_path


def main():
    ensure_blender_ui_context()
    args = parse_args(blender_argv())
    cfg = load_cfg(args.config)
    arp_cfg = cfg.get("arp", {})
    op_sequence = resolve_operator_sequence(cfg, args.operator_sequence)

    enabled, failed = try_enable_addons(args.arp_addon_modules)
    print_arp_diagnostics(args.arp_addon_modules)
    if enabled:
        print("[arp] addon enable ok:", enabled)
    if failed:
        print("[arp] addon enable failed:", json.dumps(failed, ensure_ascii=False, indent=2))
    if args.save_userpref:
        bpy.ops.wm.save_userpref()
        print("[arp] saved user preferences.")

    if "import_config_preset" in op_sequence:
        install_remap_preset(arp_cfg)

    case_ids = set(args.case_ids) if args.case_ids else None
    cases = cfg.get("cases", [])
    if case_ids is not None:
        cases = [c for c in cases if c.get("case_id") in case_ids]
    if not cases:
        raise SystemExit("No cases selected for ARP batch.")

    assets_cfg = cfg.get("assets", {})
    resolved_cases = []
    for case in cases:
        try:
            resolved = enrich_case_assets(case, assets_cfg)
            print(
                f"[arp][assets] {resolved['case_id']}: "
                f"inp_bvh={resolved['inp_bvh_path']} "
                f"tgt_bvh={resolved['tgt_bvh_path']} "
                f"tgt_fbx={resolved['tgt_fbx_path']}"
            )
            resolved_cases.append(resolved)
        except FileNotFoundError as exc:
            raise SystemExit(f"[arp][assets] {case.get('case_id')}: {exc}") from exc

    if args.check_only:
        preset_name = arp_cfg.get("remap_preset_name", "smal33_to_smal33")
        print("[arp] operator sequence:", " -> ".join(op_sequence))
        print("[arp] remap preset name:", preset_name)
        print("[arp] remap preset source:", DEFAULT_REMAP_PRESET)
        return

    if not list_arp_ops():
        raise SystemExit(
            "ARP operators unavailable. Install and enable Auto-Rig Pro addon first, "
            "then re-run with --check_only to verify."
        )

    for case in resolved_cases:
        run_case(case, cfg, op_sequence)


if __name__ == "__main__":
    main()
