"""
Resolve textured FBX paths for four-way compare rendering.

FBX assets live next to BVH files under train_char / batch2_dogs_char
character subdirectories (same layout as fbx2bvh_smal33: foo.fbx -> foo.bvh).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def list_fbx_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".fbx" and not path.name.startswith(".")
    )


def find_fbx_in_directory(directory: Path, recursive: bool = False) -> Path | None:
    direct = list_fbx_files(directory)
    if direct:
        return direct[0]
    if not recursive:
        return None
    for path in sorted(directory.rglob("*.fbx")):
        if path.is_file() and not path.name.startswith("."):
            return path
    return None


def pick_fbx_for_bvh_directory(bvh: Path) -> Path | None:
    """Pick the best FBX in the BVH parent directory when stem pairing fails."""
    candidates = list_fbx_files(bvh.parent)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    paired = bvh.with_suffix(".fbx")
    for candidate in candidates:
        if candidate == paired:
            return candidate

    char_name = bvh.parent.name
    for candidate in candidates:
        if candidate.stem == char_name:
            return candidate

    bvh_stem = bvh.stem

    def common_prefix_len(left: str, right: str) -> int:
        count = 0
        for left_ch, right_ch in zip(left, right):
            if left_ch != right_ch:
                break
            count += 1
        return count

    return max(
        candidates,
        key=lambda candidate: (
            common_prefix_len(bvh_stem, candidate.stem),
            -len(candidate.stem),
            candidate.name,
        ),
    )


def resolve_fbx_from_bvh(
    bvh_path: str | Path,
    *,
    explicit_fbx_path: str | Path | None = None,
    recursive_search: bool = False,
) -> Path:
    """
    Resolve the textured FBX that pairs with a BVH clip.

    Priority:
      1) explicit_fbx_path if provided and exists
      2) same directory, same stem (.bvh -> .fbx)
      3) first .fbx in the BVH parent directory (optional recursive search)
    """
    if explicit_fbx_path:
        resolved = Path(explicit_fbx_path)
        if resolved.exists():
            return resolved.resolve()
        raise FileNotFoundError(f"Explicit FBX not found: {resolved}")

    bvh = Path(bvh_path)
    if not bvh.exists():
        raise FileNotFoundError(f"BVH not found: {bvh}")

    paired = bvh.with_suffix(".fbx")
    if paired.exists():
        return paired.resolve()

    picked = pick_fbx_for_bvh_directory(bvh)
    if picked is not None:
        return picked.resolve()

    if recursive_search:
        found = find_fbx_in_directory(bvh.parent, recursive=True)
        if found is not None:
            return found.resolve()

    raise FileNotFoundError(
        f"No FBX found for BVH '{bvh}'. Expected '{paired}' or another .fbx "
        f"in '{bvh.parent}'."
    )


def resolve_case_fbx_paths(case_cfg: dict[str, Any], assets_cfg: dict[str, Any] | None = None):
    assets_cfg = assets_cfg or {}
    recursive = bool(assets_cfg.get("recursive_fbx_search", False))

    inp_bvh = case_cfg["inp_bvh_path"]
    tgt_bvh = case_cfg["tgt_bvh_path"]
    inp_fbx = resolve_fbx_from_bvh(
        inp_bvh,
        explicit_fbx_path=case_cfg.get("inp_fbx_path"),
        recursive_search=recursive,
    )
    tgt_fbx = resolve_fbx_from_bvh(
        tgt_bvh,
        explicit_fbx_path=case_cfg.get("tgt_fbx_path"),
        recursive_search=recursive,
    )
    return inp_fbx, tgt_fbx


def enrich_case_assets(case_cfg: dict[str, Any], assets_cfg: dict[str, Any] | None = None):
    """Return a shallow copy of case_cfg with resolved inp/tgt FBX paths."""
    out = dict(case_cfg)
    inp_fbx, tgt_fbx = resolve_case_fbx_paths(out, assets_cfg)
    out["inp_fbx_path"] = str(inp_fbx)
    out["tgt_fbx_path"] = str(tgt_fbx)
    return out
