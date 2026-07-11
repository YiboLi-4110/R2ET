import sys
import numpy as np
from pathlib import Path

PARENTS = np.array([
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 16,
    0, 18, 19, 20, 0, 22, 23, 24, 0, 26, 27, 28, 29, 30, 31,
], dtype=np.int64)

PAWS = [10, 14, 21, 25]

def norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def offsets_to_global(offsets):
    out = np.zeros_like(offsets)
    for i, p in enumerate(PARENTS):
        if p == -1:
            out[i] = offsets[i]
        else:
            out[i] = out[p] + offsets[i]
    return out

def floor_axis(j):
    paws = j[PAWS]
    spans = {}
    for i, name in enumerate("xyz"):
        spans[name] = float(np.percentile(paws[:, i], 95) - np.percentile(paws[:, i], 5))
    return min(spans, key=spans.get), spans

def describe(name, j):
    hip = (j[18] + j[22]) / 2.0
    shoulder = (j[8] + j[12]) / 2.0

    body = norm(shoulder - hip)
    across = norm((j[18] - j[22]) + (j[8] - j[12]))
    proc_forward = norm(np.cross(across, np.array([0.0, 1.0, 0.0])))

    fa, spans = floor_axis(j)
    bbox = j.max(axis=0) - j.min(axis=0)

    print(f"\n=== {name} ===")
    print("bbox xyz:", bbox)
    print("floor_axis_from_paws:", fa, spans)
    print("body hip->shoulder:", body)
    print("across left-right:", across)
    print("process_forward cross(across,Y):", proc_forward)
    print("dot(body, process_forward):", float(np.dot(body, proc_forward)))
    print("abs body vertical component |y|:", abs(float(body[1])))

def main(seq_path):
    seq_path = Path(seq_path)
    stem = seq_path.name[:-len("_seq.npy")]
    quat_path = seq_path.with_name(stem + "_quat.npy")
    skel_path = seq_path.with_name(stem + "_skel.npy")

    seq = np.load(seq_path)
    skel = np.load(skel_path)

    seq0 = seq[0, :33 * 3].reshape(33, 3)
    bind_global = offsets_to_global(skel[0].reshape(33, 3))

    describe("seq0 canonical local pose", seq0)
    describe("skel[0] bind identity global", bind_global)

if __name__ == "__main__":
    main(sys.argv[1])