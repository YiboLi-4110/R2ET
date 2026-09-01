"""Shared Auto-Rig Pro helpers for Blender scripts in this package."""

from __future__ import annotations

import importlib
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import bpy
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_REMAP_PRESET = REPO_ROOT / "visualization/arp_presets/smal33_to_smal33.bmap"


def blender_argv():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def ensure_blender_ui_context():
    """Background Blender may lack a window/3D viewport; load factory defaults."""
    wm = bpy.context.window_manager
    if wm.windows:
        return
    bpy.ops.wm.read_factory_settings(use_empty=False)
    if not wm.windows:
        raise RuntimeError(
            "No Blender window available. Run without --background or use xvfb-run."
        )


@contextmanager
def arp_view3d_context():
    """VIEW_3D context without pinning active_object (retarget switches source/target)."""
    ensure_blender_ui_context()
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    area = None
    region = None
    for candidate in screen.areas:
        if candidate.type != "VIEW_3D":
            continue
        area = candidate
        for reg in candidate.regions:
            if reg.type == "WINDOW":
                region = reg
                break
        if region is not None:
            break
    if area is None or region is None:
        raise RuntimeError("No VIEW_3D area found; cannot run ARP operators.")

    with bpy.context.temp_override(
        window=window,
        screen=screen,
        area=area,
        region=region,
        scene=bpy.context.scene,
    ):
        yield


@contextmanager
def arp_operator_context(active_object):
    """ARP setup operators poll on VIEW_3D context and a chosen active armature."""
    ensure_blender_ui_context()
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    area = None
    region = None
    for candidate in screen.areas:
        if candidate.type != "VIEW_3D":
            continue
        area = candidate
        for reg in candidate.regions:
            if reg.type == "WINDOW":
                region = reg
                break
        if region is not None:
            break
    if area is None or region is None:
        raise RuntimeError("No VIEW_3D area found; cannot run ARP operators.")

    bpy.ops.object.select_all(action="DESELECT")
    active_object.select_set(True)
    bpy.context.view_layer.objects.active = active_object

    with bpy.context.temp_override(
        window=window,
        screen=screen,
        area=area,
        region=region,
        scene=bpy.context.scene,
        active_object=active_object,
        object=active_object,
        selected_objects=[active_object],
        selected_editable_objects=[active_object],
    ):
        yield


def try_enable_addons(module_names):
    enabled = []
    failed = {}
    for module in module_names:
        try:
            bpy.ops.preferences.addon_enable(module=module)
            enabled.append(module)
        except Exception as exc:  # pragma: no cover (Blender context specific)
            failed[module] = str(exc)
    return enabled, failed


def list_arp_ops():
    if not hasattr(bpy.ops, "arp"):
        return []
    return [x for x in dir(bpy.ops.arp) if not x.startswith("_")]


def print_arp_diagnostics(module_names):
    addon_keys = sorted(bpy.context.preferences.addons.keys())
    arp_addons = [k for k in addon_keys if ("auto_rig" in k.lower() or "arp" in k.lower())]
    print("[arp] blender:", bpy.app.version_string)
    print("[arp] enabled arp-like addons:", arp_addons)
    print("[arp] requested addon modules:", module_names)
    ops = list_arp_ops()
    print("[arp] bpy.ops.arp available:", bool(ops))
    if ops:
        print("[arp] operators:", ", ".join(ops))


def assert_exists(path_str, what):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def get_armatures():
    return [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]


def pick_armature_by_hint(hint: str | None, exclude_names: set[str] | None = None):
    exclude_names = exclude_names or set()
    arms = [a for a in get_armatures() if a.name not in exclude_names]
    if not arms:
        raise RuntimeError("No matching armature found.")
    if hint:
        hit = [a for a in arms if hint.lower() in a.name.lower()]
        if hit:
            return hit[0]
    return arms[-1]


def import_bvh_armature(path: Path):
    bpy.ops.import_anim.bvh(filepath=str(path))
    obj = bpy.context.object
    if obj is None or obj.type != "ARMATURE":
        raise RuntimeError(f"BVH import did not produce an armature: {path}")
    return obj


def import_fbx_armature(path: Path):
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    arms = get_armatures()
    if not arms:
        raise RuntimeError(f"FBX import did not produce an armature: {path}")
    return arms[-1]


def import_source_fbx_with_anim(path: Path):
    """Import the SOURCE animation from FBX (keeping the action).

    We deliberately import the source rig from FBX rather than BVH: Blender's BVH
    importer applies its own up/forward axis convention and recomputes bone rolls,
    so a BVH source ends up with bone local axes that differ from the FBX target.
    ARP's ABSOLUTE rotation remap then maps rotations about mismatched axes, which
    turns the source's vertical (jump) motion into target horizontal sway. Using
    the same FBX importer for source and target keeps their bone frames consistent.
    """
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    imported = [obj for obj in bpy.data.objects if obj.name not in before]
    arms = [obj for obj in imported if obj.type == "ARMATURE"]
    if not arms:
        raise RuntimeError(f"Source FBX import produced no armature: {path}")
    animated = [a for a in arms if a.animation_data and a.animation_data.action]
    return (animated or arms)[-1]


def reset_armature_object_transform(armature):
    """ARP remap doc: zero object location/rotation before retargeting."""
    armature.location = (0.0, 0.0, 0.0)
    armature.rotation_euler = (0.0, 0.0, 0.0)
    armature.scale = (1.0, 1.0, 1.0)


def assign_arp_rigs(source_arm, target_arm):
    scene = bpy.context.scene
    scene.source_rig = source_arm.name
    scene.target_rig = target_arm.name
    print(
        f"[arp] assigned rigs: source_rig={scene.source_rig} "
        f"target_rig={scene.target_rig}"
    )


def get_action_frame_range(armature):
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        return int(action.frame_range[0]), int(action.frame_range[1])
    scene = bpy.context.scene
    return int(scene.frame_start), int(scene.frame_end)


def resolve_operator_sequence(cfg: dict, cli_sequence: list[str] | None) -> list[str]:
    arp_cfg = cfg.get("arp", {})
    if cli_sequence:
        return list(cli_sequence)
    seq = arp_cfg.get("operator_sequence")
    if seq:
        return list(seq)
    return ["auto_scale", "build_bones_list", "import_config_preset", "retarget"]


def find_enabled_arp_addon_root() -> Path:
    for key in bpy.context.preferences.addons.keys():
        if "auto_rig" not in key.lower():
            continue
        module_name = bpy.context.preferences.addons[key].module
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file:
            return Path(module_file).resolve().parent
    raise RuntimeError(
        "Auto-Rig Pro addon root not found. Enable auto_rig_pro-master first."
    )


def resolve_remap_preset_source(arp_cfg: dict) -> tuple[Path, str]:
    """Return (source_bmap_path, preset_name_for_arp_import)."""
    explicit = arp_cfg.get("remap_preset_path")
    preset_name = arp_cfg.get("remap_preset_name")
    if explicit:
        source = Path(explicit)
        if not source.is_absolute():
            source = (REPO_ROOT / source).resolve()
        else:
            source = source.resolve()
        if not preset_name:
            preset_name = source.stem
    else:
        if not preset_name:
            preset_name = "smal33_to_smal33"
        source = REPO_ROOT / "visualization" / "arp_presets" / f"{preset_name}.bmap"
        if not source.exists():
            source = DEFAULT_REMAP_PRESET
    if not source.exists():
        raise FileNotFoundError(f"Remap preset source not found: {source}")
    return source, str(preset_name)


def install_remap_preset(arp_cfg: dict) -> Path:
    """Copy repo .bmap into ARP remap_presets/ for import_config_preset.

    Resolution order:
      1) arp.remap_preset_path (explicit file; name defaults to file stem)
      2) visualization/arp_presets/<remap_preset_name>.bmap
      3) fallback DEFAULT_REMAP_PRESET (smal33_to_smal33.bmap)
    """
    source, preset_name = resolve_remap_preset_source(arp_cfg)
    # Keep name in cfg so import_config_preset uses the installed filename stem.
    arp_cfg["remap_preset_name"] = preset_name

    preset_dir = find_enabled_arp_addon_root() / "remap_presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    target = preset_dir / f"{preset_name}.bmap"
    shutil.copy2(source, target)
    print(f"[arp] installed remap preset: {source} -> {target}")
    return target


# These ARP ops expect GUI/trajectory-bone context and often fail in headless batch.
DEFAULT_OPTIONAL_ARP_OPERATORS = frozenset({"clear_root_motion", "extract_root_motion"})


def call_arp_operator(op_name: str, active_object, arp_cfg: dict, frame_range: tuple[int, int]):
    if not hasattr(bpy.ops, "arp"):
        raise RuntimeError("bpy.ops.arp is unavailable; ARP addon not enabled.")

    optional_ops = set(arp_cfg.get("optional_operators", DEFAULT_OPTIONAL_ARP_OPERATORS))

    if op_name == "retarget":
        frame_start, frame_end = frame_range
        if not hasattr(bpy.ops.arp, "retarget"):
            raise RuntimeError("bpy.ops.arp.retarget not found.")
        with arp_view3d_context():
            return bpy.ops.arp.retarget(frame_start=frame_start, frame_end=frame_end)

    with arp_operator_context(active_object):
        if op_name == "import_config_preset":
            preset_name = arp_cfg.get("remap_preset_name")
            if not preset_name:
                raise RuntimeError(
                    "arp.remap_preset_name is required for import_config_preset step."
                )
            if not hasattr(bpy.ops.arp, "import_config_preset"):
                raise RuntimeError("bpy.ops.arp.import_config_preset not found.")
            return bpy.ops.arp.import_config_preset(preset_name=preset_name)

        op = getattr(bpy.ops.arp, op_name, None)
        if op is None:
            raise RuntimeError(f"bpy.ops.arp.{op_name} not found.")
        try:
            return op()
        except RuntimeError as exc:
            if op_name in optional_ops:
                print(f"[arp][warn] bpy.ops.arp.{op_name} skipped: {exc}")
                return {"CANCELLED"}
            raise
