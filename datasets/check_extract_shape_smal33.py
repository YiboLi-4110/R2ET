import argparse
import json
import sys
from pathlib import Path

import numpy as np

JOINT_NAME_SMAL_33 = [
    "Root",
    "Spine1", "Spine2", "Spine3", "Spine4", "Spine5", "Spine6",
    "LeftScapula", "LeftUpperArm", "LeftForeLeg", "LeftFrontPaw",
    "RightScapula", "RightUpperArm", "RightForeLeg", "RightFrontPaw",
    "Neck", "Head", "Jaw",
    "LeftThigh", "LeftShin", "LeftHock", "LeftHindPaw",
    "RightThigh", "RightShin", "RightHock", "RightHindPaw",
    "Tail1", "Tail2", "Tail3", "Tail4", "Tail5", "Tail6", "Tail7",
]
BODY_JOINT_IDS = set([0, 1, 2, 3, 4, 5, 6, 15, 16, 17])
LEG_JOINT_IDS = set(
    [7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25]
)


def finite_stats(arr):
    return {
        "finite": bool(np.isfinite(arr).all()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


def basic_stats(arr):
    arr = np.asarray(arr)
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "mean": float(np.mean(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def per_axis_stats(points):
    flat = np.asarray(points).reshape(-1, 3)
    return {
        axis: {
            "min": float(flat[:, i].min()),
            "max": float(flat[:, i].max()),
            "range": float(flat[:, i].max() - flat[:, i].min()),
        }
        for i, axis in enumerate(["x", "y", "z"])
    }


def load_npz_dict(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_expected_offsets_from_npz(joint_names, skeleton):
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    expected = np.zeros_like(skeleton, dtype=np.float64)
    expected[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for idx, joint_name in enumerate(joint_names):
        if idx == 0:
            continue
        parent_idx = None
        parent_name = None
        if joint_name == "Spine1":
            parent_name = "Root"
        elif joint_name.startswith("Spine"):
            parent_name = f"Spine{int(joint_name.replace('Spine', '')) - 1}"
        elif joint_name == "LeftScapula":
            parent_name = "Spine6"
        elif joint_name == "LeftUpperArm":
            parent_name = "LeftScapula"
        elif joint_name == "LeftForeLeg":
            parent_name = "LeftUpperArm"
        elif joint_name == "LeftFrontPaw":
            parent_name = "LeftForeLeg"
        elif joint_name == "RightScapula":
            parent_name = "Spine6"
        elif joint_name == "RightUpperArm":
            parent_name = "RightScapula"
        elif joint_name == "RightForeLeg":
            parent_name = "RightUpperArm"
        elif joint_name == "RightFrontPaw":
            parent_name = "RightForeLeg"
        elif joint_name == "Neck":
            parent_name = "Spine6"
        elif joint_name == "Head":
            parent_name = "Neck"
        elif joint_name == "Jaw":
            parent_name = "Head"
        elif joint_name == "LeftThigh":
            parent_name = "Root"
        elif joint_name == "LeftShin":
            parent_name = "LeftThigh"
        elif joint_name == "LeftHock":
            parent_name = "LeftShin"
        elif joint_name == "LeftHindPaw":
            parent_name = "LeftHock"
        elif joint_name == "RightThigh":
            parent_name = "Root"
        elif joint_name == "RightShin":
            parent_name = "RightThigh"
        elif joint_name == "RightHock":
            parent_name = "RightShin"
        elif joint_name == "RightHindPaw":
            parent_name = "RightHock"
        elif joint_name == "Tail1":
            parent_name = "Root"
        elif joint_name.startswith("Tail"):
            parent_name = f"Tail{int(joint_name.replace('Tail', '')) - 1}"

        if parent_name is None or parent_name not in name_to_idx:
            raise ValueError(f"Could not infer parent for joint {joint_name}")
        parent_idx = name_to_idx[parent_name]
        expected[idx] = skeleton[idx]  # placeholder to preserve shape
        _ = parent_idx
    return expected


def check_one(npz_path):
    data = load_npz_dict(npz_path)
    result = {
        "npz_path": str(npz_path),
        "status": "ok",
        "errors": [],
        "warnings": [],
        "keys": sorted(data.keys()),
    }

    required = [
        "skinning_weights",
        "joint_names",
        "root_orient",
        "rest_vertices",
        "rest_faces",
        "skeleton",
        "subject",
        "vertex_part",
        "rest_body_vertices",
        "rest_arm_vertices",
        "body_width",
        "full_width",
        "joint_shape",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        result["errors"].append(f"missing keys: {missing}")
        result["status"] = "error"
        return result

    skinning_weights = np.asarray(data["skinning_weights"])
    joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
    root_orient = np.asarray(data["root_orient"])
    rest_vertices = np.asarray(data["rest_vertices"])
    rest_faces = np.asarray(data["rest_faces"])
    skeleton = np.asarray(data["skeleton"])
    vertex_part = np.asarray(data["vertex_part"])
    rest_body_vertices = np.asarray(data["rest_body_vertices"])
    rest_arm_vertices = np.asarray(data["rest_arm_vertices"])
    body_width = np.asarray(data["body_width"])
    full_width = np.asarray(data["full_width"])
    joint_shape = np.asarray(data["joint_shape"])

    result["shapes"] = {
        "skinning_weights": list(skinning_weights.shape),
        "root_orient": list(root_orient.shape),
        "rest_vertices": list(rest_vertices.shape),
        "rest_faces": list(rest_faces.shape),
        "skeleton": list(skeleton.shape),
        "vertex_part": list(vertex_part.shape),
        "rest_body_vertices": list(rest_body_vertices.shape),
        "rest_arm_vertices": list(rest_arm_vertices.shape),
        "body_width": list(body_width.shape),
        "full_width": list(full_width.shape),
        "joint_shape": list(joint_shape.shape),
    }
    result["finite"] = {
        "skinning_weights": finite_stats(skinning_weights),
        "root_orient": finite_stats(root_orient),
        "rest_vertices": finite_stats(rest_vertices),
        "skeleton": finite_stats(skeleton),
        "body_width": finite_stats(body_width),
        "full_width": finite_stats(full_width),
        "joint_shape": finite_stats(joint_shape),
    }

    num_joints = len(JOINT_NAME_SMAL_33)
    if skinning_weights.ndim != 2 or skinning_weights.shape[1] != num_joints:
        result["errors"].append(f"skinning_weights shape should be (V, {num_joints})")
    if len(joint_names) != num_joints:
        result["errors"].append(f"joint_names length should be {num_joints}")
    if joint_names != JOINT_NAME_SMAL_33:
        result["errors"].append("joint_names do not match JOINT_NAME_SMAL_33 order")
    if root_orient.ndim != 2 or root_orient.shape[1] != 3:
        result["errors"].append("root_orient should have shape (T, 3)")
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        result["errors"].append("rest_vertices should have shape (V, 3)")
    if rest_faces.ndim != 2 or rest_faces.shape[1] != 3:
        result["errors"].append("rest_faces should have shape (F, 3)")
    if skeleton.shape != (num_joints, 3):
        result["errors"].append(f"skeleton shape should be ({num_joints}, 3)")
    if vertex_part.ndim != 1:
        result["errors"].append("vertex_part should have shape (V,)")
    if vertex_part.shape[0] != rest_vertices.shape[0]:
        result["errors"].append("vertex_part length should equal rest_vertices count")
    if joint_shape.shape != (num_joints, 3):
        result["errors"].append(f"joint_shape shape should be ({num_joints}, 3)")
    if body_width.shape != (3,):
        result["errors"].append("body_width should have shape (3,)")
    if full_width.shape != (3,):
        result["errors"].append("full_width should have shape (3,)")
    if rest_body_vertices.ndim != 2 or (rest_body_vertices.size and rest_body_vertices.shape[1] != 3):
        result["errors"].append("rest_body_vertices should have shape (N, 3)")
    if rest_arm_vertices.ndim != 2 or (rest_arm_vertices.size and rest_arm_vertices.shape[1] != 3):
        result["errors"].append("rest_arm_vertices should have shape (N, 3)")

    if result["errors"]:
        result["status"] = "error"
        return result

    result["rest_vertices_axis"] = per_axis_stats(rest_vertices)
    result["skeleton_axis"] = per_axis_stats(skeleton)

    skin_sum = skinning_weights.sum(axis=1)
    result["skinning_weight_row_sum"] = basic_stats(skin_sum)
    if not np.allclose(skin_sum, 1.0, atol=1e-3):
        result["warnings"].append("skinning_weights rows are not all close to 1")

    negative_weights = int((skinning_weights < -1e-6).sum())
    if negative_weights > 0:
        result["errors"].append(f"skinning_weights contain {negative_weights} negative entries")

    result["vertex_part"] = {
        "min": int(vertex_part.min()) if vertex_part.size else None,
        "max": int(vertex_part.max()) if vertex_part.size else None,
        "unique_joint_count": int(len(np.unique(vertex_part))),
        "counts": {
            JOINT_NAME_SMAL_33[i]: int((vertex_part == i).sum())
            for i in range(num_joints)
            if int((vertex_part == i).sum()) > 0
        },
    }
    if vertex_part.size and (vertex_part.min() < 0 or vertex_part.max() >= num_joints):
        result["errors"].append("vertex_part contains out-of-range joint ids")

    body_mask = np.isin(vertex_part, list(BODY_JOINT_IDS))
    legs_mask = np.isin(vertex_part, list(LEG_JOINT_IDS))
    if body_mask.any():
        computed_body_width = rest_vertices[body_mask].max(axis=0) - rest_vertices[body_mask].min(axis=0)
        result["body_width_error"] = basic_stats(np.abs(computed_body_width - body_width))
        if not np.allclose(computed_body_width, body_width, atol=1e-5):
            result["warnings"].append("body_width does not match rest_vertices selected by body joints")
    else:
        result["warnings"].append("vertex_part assigns no vertices to the body joint group")

    computed_full_width = rest_vertices.max(axis=0) - rest_vertices.min(axis=0)
    result["full_width_error"] = basic_stats(np.abs(computed_full_width - full_width))
    if not np.allclose(computed_full_width, full_width, atol=1e-5):
        result["warnings"].append("full_width does not match rest_vertices bounding box")

    if legs_mask.any():
        if rest_arm_vertices.shape[0] != int(legs_mask.sum()):
            result["warnings"].append(
                "rest_arm_vertices count does not match vertices assigned to leg joints"
            )
    else:
        result["warnings"].append("vertex_part assigns no vertices to the leg joint group")

    joint_shape_recomputed = np.zeros_like(joint_shape, dtype=np.float64)
    for joint_id in range(num_joints):
        joint_mask = vertex_part == joint_id
        if np.any(joint_mask):
            verts = rest_vertices[joint_mask]
            joint_shape_recomputed[joint_id] = verts.max(axis=0) - verts.min(axis=0)
    joint_shape_err = np.abs(joint_shape_recomputed - joint_shape)
    result["joint_shape_error"] = {
        "max": float(joint_shape_err.max()),
        "mean": float(joint_shape_err.mean()),
        "p99": float(np.percentile(joint_shape_err, 99)),
    }
    if joint_shape_err.max() > 1e-5:
        result["warnings"].append("joint_shape does not match widths recomputed from vertex_part")

    expected_offsets = build_expected_offsets_from_npz(joint_names, skeleton)
    result["skeleton_root_offset_norm"] = float(np.linalg.norm(skeleton[0]))
    if result["skeleton_root_offset_norm"] > 1e-6:
        result["warnings"].append(
            "root joint offset is not near zero; verify whether skeleton[0] is intended to be local-to-root"
        )

    if result["errors"]:
        result["status"] = "error"
    elif result["warnings"]:
        result["status"] = "warning"
    return result


def summarize(results):
    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "warning": sum(1 for r in results if r["status"] == "warning"),
        "error": sum(1 for r in results if r["status"] == "error"),
    }

    if results:
        skin_sum_mean = [
            r.get("skinning_weight_row_sum", {}).get("mean")
            for r in results
            if r.get("skinning_weight_row_sum", {}).get("mean") is not None
        ]
        if skin_sum_mean:
            summary["skinning_weight_row_sum_mean"] = basic_stats(skin_sum_mean)

        joint_shape_max = [
            r.get("joint_shape_error", {}).get("max")
            for r in results
            if r.get("joint_shape_error", {}).get("max") is not None
        ]
        if joint_shape_max:
            summary["joint_shape_error_max"] = float(max(joint_shape_max))

        root_offset_norm = [
            r.get("skeleton_root_offset_norm")
            for r in results
            if r.get("skeleton_root_offset_norm") is not None
        ]
        if root_offset_norm:
            summary["skeleton_root_offset_norm"] = basic_stats(root_offset_norm)

    return summary


def print_human_report(summary, results, max_items):
    print("=== SMAL 33 shape extraction check summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()

    for result in results[:max_items]:
        print(f"[{result['status'].upper()}] {result['npz_path']}")
        print(f"  keys: {result['keys']}")
        if "shapes" in result:
            print(f"  shapes: {result['shapes']}")
        if result["errors"]:
            print(f"  errors: {result['errors']}")
        if result["warnings"]:
            print(f"  warnings: {result['warnings']}")
        if "skinning_weight_row_sum" in result:
            print(
                "  skin row-sum min/mean/max: "
                f"{result['skinning_weight_row_sum']['min']:.6g} / "
                f"{result['skinning_weight_row_sum']['mean']:.6g} / "
                f"{result['skinning_weight_row_sum']['max']:.6g}"
            )
        if "joint_shape_error" in result:
            print(
                "  joint_shape err max/mean/p99: "
                f"{result['joint_shape_error']['max']:.6g} / "
                f"{result['joint_shape_error']['mean']:.6g} / "
                f"{result['joint_shape_error']['p99']:.6g}"
            )
        if "skeleton_root_offset_norm" in result:
            print(f"  skeleton root offset norm: {result['skeleton_root_offset_norm']:.6g}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Check SMAL 33 extract_shape outputs."
    )
    parser.add_argument(
        "--npz-root",
        required=True,
        type=Path,
        help="Directory containing extracted .npz shape files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only check the first N .npz files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON payload instead of the short human report.",
    )
    parser.add_argument(
        "--max-report-items",
        type=int,
        default=20,
        help="Maximum per-file entries shown in the human report.",
    )
    args = parser.parse_args()

    npz_root = args.npz_root.resolve()
    npz_paths = sorted(npz_root.glob("*.npz"))
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]

    results = [check_one(path) for path in npz_paths]
    summary = summarize(results)
    payload = {"summary": summary, "results": results}

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human_report(summary, results, args.max_report_items)

    if summary["error"] > 0:
        raise SystemExit(2)
    if summary["warning"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
