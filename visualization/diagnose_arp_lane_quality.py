#!/usr/bin/env python3
"""
Deep-dive diagnostic for the ARP lane in a fourway_compare pipeline.

This script is pure NumPy (no torch / no Blender), so it runs in any Python env
on the server. It answers three questions that the previous space diagnostic
could not separate:

  Q1. SPACE CHECK
      Is the ARP mesh cache (arp_mesh.npz) actually exported in
      root_bone_local_zup, or did a stale Blender script fall back to world
      space? We measure horizontal center span; world space ~ meters,
      root-local ~ 0.1m.

  Q2. VERTICAL-vs-HORIZONTAL MOTION DECOMPOSITION
      A jump should move the body mostly UP/DOWN (vertical). If the ARP lane's
      body-center energy is mostly HORIZONTAL / side-to-side, that is the "S
      wobble" symptom. We report, per lane, the fraction of center motion that
      is vertical vs lateral, and a wobble/sign-change count.

  Q3. LEFT/RIGHT LIMB SWAP CHECK (is mirror_z swapping legs?)
      A pure reflection (mirror_z / mirror_x) makes symmetric standing poses
      match well but SWAPS which limb moves during asymmetric motion. Using the
      SOURCE lane as ground truth for "which side moves when", we test whether
      the ARP lane's left/right limb motion correlates better WITHOUT swap
      (good) or WITH left<->right swap (bad = reflection artifact).

Two input modes:
  (A) --npz path/to/fourway_compare.npz
      Uses lane vertex arrays: source/ours/arp/copyquat. Best for Q1/Q2 and a
      geometric limb-swap proxy (Q3 via limb-region centroids).

  (B) --arp_mesh path/to/*_arp_mesh.npz
      Directly inspects the raw ARP mesh cache for Q1 (space + raw span).

Usage examples:
  python visualization/diagnose_arp_lane_quality.py \
      --npz temp/fourway_batch_XXXX/work/<case>/fourway_compare.npz \
      --arp_mesh temp/fourway_batch_XXXX/work/arp_mesh_outputs/<case>_arp_mesh.npz \
      --source_lane source --out_json temp/arp_lane_quality_<case>.json

  python visualization/diagnose_arp_lane_quality.py \
      --npz .../fourway_compare.npz            # npz only

See the printed "verdict" block and the docstring of each metric for how to
read the numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate convention
# ---------------------------------------------------------------------------
# fourway_compare.npz vertices are stored Y-up (LBS space, before Blender's
# y_up_to_z_up render conversion). So:
#   horizontal plane = (X, Z), vertical = Y.
# arp_mesh.npz raw vertices are Blender Z-up, so:
#   horizontal plane = (X, Y), vertical = Z.
H_AXES_YUP = (0, 2)
V_AXIS_YUP = 1
H_AXES_ZUP = (0, 1)
V_AXIS_ZUP = 2


def bbox_center_per_frame(verts: np.ndarray) -> np.ndarray:
    mn = verts.min(axis=1)
    mx = verts.max(axis=1)
    return ((mn + mx) * 0.5).astype(np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep ARP lane quality diagnostic.")
    p.add_argument("--npz", type=str, default=None, help="fourway_compare.npz")
    p.add_argument("--arp_mesh", type=str, default=None, help="raw *_arp_mesh.npz")
    p.add_argument(
        "--source_lane",
        type=str,
        default="source",
        choices=["source", "copyquat", "ours"],
        help="Ground-truth lane for the left/right limb-swap test (Q3).",
    )
    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Q1. SPACE CHECK
# ---------------------------------------------------------------------------
def inspect_arp_mesh_cache(path: Path) -> dict[str, Any]:
    payload = np.load(str(path))
    files = list(payload.files)
    space = str(payload["space"]) if "space" in files else "MISSING"
    root_bone = str(payload["root_bone_name"]) if "root_bone_name" in files else "MISSING"
    verts = np.asarray(payload["vertices"], dtype=np.float32)  # Blender Z-up
    centers = bbox_center_per_frame(verts)
    span = centers.max(axis=0) - centers.min(axis=0)
    h_span = float(np.linalg.norm(span[list(H_AXES_ZUP)]))
    v_span = float(abs(span[V_AXIS_ZUP]))

    if space == "MISSING":
        interpretation = (
            "NO 'space' field -> stale arp_export_mesh_blender.py (pre root-local). "
            "Vertices are almost certainly Blender WORLD space."
        )
    else:
        interpretation = f"space field present: {space}"

    likely_world = h_span > 1.0
    return {
        "path": str(path),
        "npz_files": files,
        "space_field": space,
        "root_bone_field": root_bone,
        "frames": int(verts.shape[0]),
        "num_vertices": int(verts.shape[1]),
        "center_span_xyz_zup": span.tolist(),
        "center_span_horizontal_zup": h_span,
        "center_span_vertical_zup": v_span,
        "likely_world_space": bool(likely_world),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# npz lane helpers
# ---------------------------------------------------------------------------
def load_lane_vertices(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    lanes: dict[str, np.ndarray] = {}
    mapping = {
        "source": "source_vertices",
        "ours": "ours_vertices",
        "arp": "arp_vertices",
        "copyquat": "copyquat_vertices",
    }
    for lane, key in mapping.items():
        if key in npz.files:
            lanes[lane] = np.asarray(npz[key], dtype=np.float32)
    return lanes


# ---------------------------------------------------------------------------
# Q2. VERTICAL vs HORIZONTAL body-center motion decomposition
# ---------------------------------------------------------------------------
def vertical_vs_horizontal(verts: np.ndarray) -> dict[str, Any]:
    """
    Decompose per-frame body-center displacement into vertical vs horizontal
    energy. Jump-like motion should be vertical-dominant. Side-to-side S wobble
    shows up as high lateral energy and many sign changes.
    """
    centers = bbox_center_per_frame(verts)  # (T, 3) Y-up
    if len(centers) <= 1:
        return {"frames": int(len(centers)), "insufficient_frames": True}

    step = centers[1:] - centers[:-1]
    v_step = np.abs(step[:, V_AXIS_YUP])
    h_step = np.linalg.norm(step[:, list(H_AXES_YUP)], axis=-1)

    v_energy = float(np.sum(v_step ** 2))
    h_energy = float(np.sum(h_step ** 2))
    total = v_energy + h_energy + 1e-12
    vertical_fraction = v_energy / total

    # Lateral wobble in this lane's OWN frame: project horizontal motion onto
    # its dominant horizontal direction (forward) and the orthogonal (lateral).
    h = centers[:, list(H_AXES_YUP)]
    disp = h[-1] - h[0]
    n = np.linalg.norm(disp)
    forward = disp / n if n > 1e-8 else np.array([1.0, 0.0])
    right = np.array([forward[1], -forward[0]])
    h_rel = h - h[0]
    lateral = h_rel @ right
    signs = np.sign(lateral)
    sign_changes = int(np.sum((signs[1:] * signs[:-1]) < 0))

    return {
        "frames": int(len(centers)),
        "vertical_span": float(centers[:, V_AXIS_YUP].max() - centers[:, V_AXIS_YUP].min()),
        "horizontal_span": float(
            np.linalg.norm(
                centers[:, list(H_AXES_YUP)].max(axis=0)
                - centers[:, list(H_AXES_YUP)].min(axis=0)
            )
        ),
        "vertical_energy_fraction": vertical_fraction,
        "horizontal_energy_fraction": 1.0 - vertical_fraction,
        "lateral_std": float(np.std(lateral)),
        "lateral_sign_changes": sign_changes,
    }


# ---------------------------------------------------------------------------
# Q3. LEFT/RIGHT LIMB SWAP CHECK
# ---------------------------------------------------------------------------
# We cannot rely on bone indices here (arp lane is skinned mesh vertices), so we
# use a geometric proxy: split vertices into left/right halves by their sign on
# the lateral axis in a canonical frame, then track each half's centroid motion.
# If ARP is a correct (non-reflected) match to SOURCE, the LEFT half motion
# should correlate with SOURCE's LEFT half. If mirror_z swapped sides, ARP LEFT
# will instead correlate with SOURCE RIGHT.
def _canonical_lateral_axis(verts: np.ndarray) -> int:
    """Pick the horizontal axis with the largest static spread as 'lateral'."""
    v0 = verts[0]
    spread = v0.max(axis=0) - v0.min(axis=0)
    # Among horizontal axes (Y-up: X and Z), pick the wider one as left-right.
    hx, hz = H_AXES_YUP
    return hx if spread[hx] >= spread[hz] else hz


def _half_centroids(verts: np.ndarray, lateral_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_centroid_seq, right_centroid_seq), each (T,3)."""
    coord0 = verts[0, :, lateral_axis]
    median = float(np.median(coord0))
    left_mask = coord0 < median
    right_mask = ~left_mask
    left = verts[:, left_mask, :].mean(axis=1)
    right = verts[:, right_mask, :].mean(axis=1)
    return left.astype(np.float64), right.astype(np.float64)


def _motion_signal(centroid_seq: np.ndarray) -> np.ndarray:
    """Per-frame speed magnitude of a centroid, mean-removed."""
    step = np.linalg.norm(np.diff(centroid_seq, axis=0), axis=-1)
    step = np.concatenate([step[:1], step])  # pad to length T
    return step - step.mean()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def left_right_swap_check(
    arp_verts: np.ndarray,
    src_verts: np.ndarray,
) -> dict[str, Any]:
    """
    Compare ARP left/right limb motion timing against SOURCE.

    Note: this uses a geometric left/right split, so it is a PROXY. It works
    best when source and target meshes share vertex order & topology (they are
    both SMAL33-skinned target/source meshes; source is a different mesh, so we
    rely on the temporal correlation of left vs right half motion, not on
    per-vertex identity).
    """
    lat_arp = _canonical_lateral_axis(arp_verts)
    lat_src = _canonical_lateral_axis(src_verts)

    arp_l, arp_r = _half_centroids(arp_verts, lat_arp)
    src_l, src_r = _half_centroids(src_verts, lat_src)

    sig_arp_l = _motion_signal(arp_l)
    sig_arp_r = _motion_signal(arp_r)
    sig_src_l = _motion_signal(src_l)
    sig_src_r = _motion_signal(src_r)

    # No-swap: arp left <-> src left, arp right <-> src right
    noswap = 0.5 * (_corr(sig_arp_l, sig_src_l) + _corr(sig_arp_r, sig_src_r))
    # Swap: arp left <-> src right, arp right <-> src left
    swap = 0.5 * (_corr(sig_arp_l, sig_src_r) + _corr(sig_arp_r, sig_src_l))

    if abs(noswap - swap) < 0.05:
        verdict = "inconclusive (left/right motion too symmetric to tell)"
    elif swap > noswap:
        verdict = "LIKELY SWAPPED: reflection (mirror_*) appears to swap left/right limbs"
    else:
        verdict = "OK: left/right limb timing matches source without swap"

    return {
        "lateral_axis_arp": int(lat_arp),
        "lateral_axis_source": int(lat_src),
        "corr_no_swap": float(noswap),
        "corr_swapped": float(swap),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Shape / orientation candidates (kept for convenience, per-vertex RMSE)
# ---------------------------------------------------------------------------
def center_vertices(verts: np.ndarray) -> np.ndarray:
    return verts - bbox_center_per_frame(verts)[:, None, :]


def apply_candidate(verts: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(verts, dtype=np.float64).copy()
    if name == "identity":
        return out
    if name == "yaw_180":
        out[..., 0] *= -1.0
        out[..., 2] *= -1.0
    elif name == "mirror_x":
        out[..., 0] *= -1.0
    elif name == "mirror_z":
        out[..., 2] *= -1.0
    return out


def orientation_rmse(arp: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if arp.shape != ref.shape:
        return {"comparable": False, "reason": f"arp={arp.shape} ref={ref.shape}"}
    a = center_vertices(arp).astype(np.float64)
    r = center_vertices(ref).astype(np.float64)
    scores = {}
    for name in ("identity", "yaw_180", "mirror_x", "mirror_z"):
        t = apply_candidate(a, name)
        rmse = np.sqrt(np.mean((t - r) ** 2, axis=(1, 2)))
        scores[name] = {"rmse_mean": float(rmse.mean()), "rmse_p95": float(np.percentile(rmse, 95))}
    best = min(scores, key=lambda k: scores[k]["rmse_mean"])
    return {"comparable": True, "best_candidate": best, "scores": scores}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if not args.npz and not args.arp_mesh:
        raise SystemExit("Provide at least one of --npz or --arp_mesh.")

    report: dict[str, Any] = {"inputs": {"npz": args.npz, "arp_mesh": args.arp_mesh}}
    verdicts: list[str] = []

    # Q1 (raw cache) --------------------------------------------------------
    if args.arp_mesh:
        mesh_path = Path(args.arp_mesh)
        if not mesh_path.exists():
            raise FileNotFoundError(f"arp_mesh not found: {mesh_path}")
        q1 = inspect_arp_mesh_cache(mesh_path)
        report["Q1_arp_mesh_space_check"] = q1
        if q1["space_field"] == "MISSING" or q1["likely_world_space"]:
            verdicts.append(
                "Q1: ARP mesh cache looks like WORLD space (stale exporter). "
                "Sync arp_export_mesh_blender.py + arp_blender_common.py and rerun."
            )
        else:
            verdicts.append(f"Q1: ARP mesh space OK = {q1['space_field']}.")

    # npz-based analyses ----------------------------------------------------
    if args.npz:
        npz_path = Path(args.npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"npz not found: {npz_path}")
        payload = np.load(str(npz_path))
        lanes = load_lane_vertices(payload)
        report["available_lanes"] = sorted(lanes.keys())
        report["lane_shapes"] = {k: list(v.shape) for k, v in lanes.items()}

        if "arp" not in lanes:
            raise KeyError("npz has no arp_vertices.")

        # Q2 -------------------------------------------------------------
        report["Q2_vertical_vs_horizontal"] = {
            lane: vertical_vs_horizontal(v) for lane, v in lanes.items()
        }
        arp_vf = report["Q2_vertical_vs_horizontal"]["arp"].get("vertical_energy_fraction")
        ref_lane = "source" if "source" in lanes else args.source_lane
        ref_vf = report["Q2_vertical_vs_horizontal"].get(ref_lane, {}).get(
            "vertical_energy_fraction"
        )
        if arp_vf is not None and ref_vf is not None:
            if arp_vf + 0.15 < ref_vf:
                verdicts.append(
                    f"Q2: ARP body motion is too HORIZONTAL "
                    f"(vertical_fraction arp={arp_vf:.2f} vs {ref_lane}={ref_vf:.2f}) "
                    "-> matches the 'S wobble instead of jump' symptom."
                )
            else:
                verdicts.append(
                    f"Q2: ARP vertical/horizontal balance ok "
                    f"(arp={arp_vf:.2f}, {ref_lane}={ref_vf:.2f})."
                )

        # Q3 -------------------------------------------------------------
        if args.source_lane in lanes:
            q3 = left_right_swap_check(lanes["arp"], lanes[args.source_lane])
            report["Q3_left_right_swap_check"] = q3
            verdicts.append("Q3: " + q3["verdict"])
        else:
            report["Q3_left_right_swap_check"] = {
                "skipped": True,
                "reason": f"source_lane '{args.source_lane}' not in npz",
            }

        # Bonus: orientation RMSE vs reference lanes
        for ref in ("copyquat", "ours", "source"):
            if ref in lanes and ref != "arp":
                report[f"orientation_rmse_vs_{ref}"] = orientation_rmse(
                    lanes["arp"], lanes[ref]
                )

    report["verdicts"] = verdicts

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    print("\n================ VERDICT ================")
    for v in verdicts:
        print(" - " + v)
    print("=========================================")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[diag] wrote: {out}")


if __name__ == "__main__":
    main()
