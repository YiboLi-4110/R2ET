"""
SMAL33 pet-data feeder for skeleton-aware / shape-aware training.

Changes vs datasets/train_feeder_r2et.py:
  - 33-joint SMAL quadruped topology
  - min_frames filtering (default 32)
  - load precomputed smal33_* stats when available
  - quadruped height proxy
  - fixed global-motion padding shape (T, 4)
  - random crop only when sequence length > max_length
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join

NUM_JOINTS = 33
GLOBAL_DIM = 4
SEQ_TAIL_DIM = 8

SMAL33_PARENTS = np.array(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        6,
        11,
        12,
        13,
        6,
        15,
        16,
        0,
        18,
        19,
        20,
        0,
        22,
        23,
        24,
        0,
        26,
        27,
        28,
        29,
        30,
        31,
    ],
    dtype=np.int64,
)

STATS_FILES = {
    "local_mean": "smal33_local_motion_mean.npy",
    "local_std": "smal33_local_motion_std.npy",
    "global_mean": "smal33_global_motion_mean.npy",
    "global_std": "smal33_global_motion_std.npy",
    "quat_mean": "smal33_quat_mean.npy",
    "quat_std": "smal33_quat_std.npy",
    "shape_mean": "smal33_shape_mean_xyz.npy",
    "shape_std": "smal33_shape_std_xyz.npy",
}


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        stats_path,
        shape_path,
        max_length,
        min_frames=32,
        recompute_stats=False,
    ):
        self.data_path = data_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.min_frames = min_frames
        self.recompute_stats = recompute_stats
        self.parents = SMAL33_PARENTS.copy()

        self.left_front_leg_lst = np.array([7, 8, 9, 10])
        self.right_front_leg_lst = np.array([11, 12, 13, 14])
        self.left_hind_leg_lst = np.array([18, 19, 20, 21])
        self.right_hind_leg_lst = np.array([22, 23, 24, 25])
        self.body_bone_lst = np.array([0, 1, 2, 3, 4, 5, 6, 15, 16, 17])

        self.shape_dic = {}
        shape_lst = []
        file_names = sorted(
            f for f in listdir(shape_path) if not f.startswith(".") and f.endswith(".npz")
        )
        if not file_names:
            raise FileNotFoundError(f"No .npz shape files found under {shape_path}")

        for shape_name in file_names:
            fbx_file = np.load(join(shape_path, shape_name))
            full_width = fbx_file["full_width"].astype(np.single)
            joint_shape = fbx_file["joint_shape"].astype(np.single)
            if joint_shape.shape[0] != NUM_JOINTS:
                raise ValueError(
                    f"Expected {NUM_JOINTS} joints in {shape_name}, got {joint_shape.shape[0]}"
                )

            shape_vecotr = np.divide(joint_shape, full_width[None, :])
            self.shape_dic[shape_name.split(".")[0]] = shape_vecotr
            shape_lst.append(shape_vecotr[:, :])

        shape_array = np.concatenate(shape_lst, axis=0)
        self.shape_mean = shape_array.mean(axis=0)
        self.shape_std = shape_array.std(axis=0)

        self.load_data()

    def _stats_available(self):
        return all(
            exists(join(self.stats_path, fname)) for fname in STATS_FILES.values()
        )

    def _load_stats_from_disk(self):
        return {
            key: np.load(join(self.stats_path, fname))
            for key, fname in STATS_FILES.items()
        }

    def _save_stats(self, stats):
        if not exists(self.stats_path):
            makedirs(self.stats_path)
        np.save(
            join(self.stats_path, STATS_FILES["local_mean"]), stats["local_mean"]
        )
        np.save(join(self.stats_path, STATS_FILES["local_std"]), stats["local_std"])
        np.save(
            join(self.stats_path, STATS_FILES["global_mean"]), stats["global_mean"]
        )
        np.save(
            join(self.stats_path, STATS_FILES["global_std"]), stats["global_std"]
        )
        np.save(join(self.stats_path, STATS_FILES["quat_mean"]), stats["quat_mean"])
        np.save(join(self.stats_path, STATS_FILES["quat_std"]), stats["quat_std"])
        np.save(
            join(self.stats_path, STATS_FILES["shape_mean"]),
            stats["shape_mean"],
        )
        np.save(
            join(self.stats_path, STATS_FILES["shape_std"]),
            stats["shape_std"],
        )

    def load_data(self):
        all_local = []
        all_global = []
        all_skel = []
        all_names = []
        t_skel = []
        all_quats = []
        seq_names = []
        skipped_short = 0
        skipped_missing = 0

        folders = sorted(
            f
            for f in listdir(self.data_path)
            if not f.startswith(".") and not f.endswith("py") and not f.endswith(".npz")
        )
        for folder_name in folders:
            files = sorted(
                f
                for f in listdir(join(self.data_path, folder_name))
                if not f.startswith(".") and f.endswith("_seq.npy")
            )
            for cfile in files:
                file_name = cfile[: -len("_seq.npy")]
                skel_path = join(self.data_path, folder_name, file_name + "_skel.npy")
                quat_path = join(self.data_path, folder_name, file_name + "_quat.npy")
                if not exists(skel_path) or not exists(quat_path):
                    skipped_missing += 1
                    continue

                sequence = np.load(join(self.data_path, folder_name, cfile))
                if sequence.shape[0] < self.min_frames:
                    skipped_short += 1
                    continue

                positions = np.load(skel_path)
                offset = sequence[:, -SEQ_TAIL_DIM:-4]
                sequence = np.reshape(sequence[:, :-SEQ_TAIL_DIM], [sequence.shape[0], -1, 3])
                if sequence.shape[1] != NUM_JOINTS:
                    raise ValueError(
                        f"Expected {NUM_JOINTS} joints in {cfile}, got {sequence.shape[1]}"
                    )

                positions = positions.copy()
                positions[:, 0, :] = sequence[:, 0, :]

                all_local.append(sequence)
                all_global.append(offset)
                all_skel.append(positions)
                all_names.append(folder_name)
                seq_names.append(file_name)

                quat = np.load(quat_path)
                if quat.shape[1] != NUM_JOINTS:
                    raise ValueError(
                        f"Expected {NUM_JOINTS} quat joints in {quat_path}, got {quat.shape[1]}"
                    )
                all_quats.append(quat)

        if not all_local:
            raise RuntimeError(
                f"No sequences loaded from {self.data_path} with min_frames={self.min_frames}."
            )

        train_local = all_local
        train_global = all_global
        train_skel = all_skel

        for tt in train_skel:
            t_skel.append(tt[0:1])

        ntotal_sequences = len(train_local)
        print(f"Number of sequences: {ntotal_sequences}")
        print(f"Skipped short (<{self.min_frames}): {skipped_short}")
        print(f"Skipped missing sidecars: {skipped_missing}")

        use_cached_stats = self._stats_available() and not self.recompute_stats
        if use_cached_stats:
            print(f"Loading precomputed stats from {self.stats_path}")
            cached = self._load_stats_from_disk()
            local_mean = cached["local_mean"]
            local_std = cached["local_std"]
            global_mean = cached["global_mean"]
            global_std = cached["global_std"]
            quat_mean = cached["quat_mean"]
            quat_std = cached["quat_std"]
            self.shape_mean = cached["shape_mean"]
            self.shape_std = cached["shape_std"]
        else:
            print("Computing stats from loaded motion data")
            allframes_n_skel = np.concatenate(train_local + t_skel)
            local_mean = allframes_n_skel.mean(axis=0)[None, :]
            global_mean = np.concatenate(train_global).mean(axis=0)[None, :]
            local_std = allframes_n_skel.std(axis=0)[None, :]
            global_std = np.concatenate(train_global).std(axis=0)[None, :]

            allframes_quat = np.concatenate(all_quats)
            quat_mean = allframes_quat.mean(axis=0)[None, :]
            quat_std = allframes_quat.std(axis=0)[None, :]

            self._save_stats(
                {
                    "local_mean": local_mean,
                    "local_std": local_std,
                    "global_mean": global_mean,
                    "global_std": global_std,
                    "quat_mean": quat_mean,
                    "quat_std": quat_std,
                    "shape_mean": self.shape_mean,
                    "shape_std": self.shape_std,
                }
            )

        self.num_joint = NUM_JOINTS
        local_std = local_std.copy()
        local_std[local_std == 0] = 1
        quat_std = quat_std.copy()
        quat_std[quat_std == 0] = 1

        for i in range(len(train_local)):
            train_local[i] = (train_local[i] - local_mean) / local_std
            train_global[i] = train_global[i]
            train_skel[i] = (train_skel[i] - local_mean) / local_std
            all_quats[i] = (all_quats[i] - quat_mean) / quat_std

        self.train_local = train_local
        self.train_global = train_global
        self.train_skel = train_skel
        self.local_mean = local_mean
        self.local_std = local_std
        self.quat_mean = quat_mean
        self.quat_std = quat_std
        self.global_mean = global_mean
        self.global_std = global_std
        self.all_names = all_names
        self.seq_names = seq_names
        self.all_quats = all_quats

    @staticmethod
    def _crop_start(seq_len, max_len):
        if seq_len <= max_len:
            return 0
        return np.random.randint(low=0, high=seq_len - max_len)

    @staticmethod
    def _pad_time(array, target_len, tail_shape, identity_quat=False):
        if array.shape[0] >= target_len:
            return array
        pad_count = target_len - array.shape[0]
        if identity_quat:
            tail = np.zeros((pad_count,) + tail_shape, dtype=array.dtype)
            tail[..., 0] = 1.0
        else:
            tail = np.zeros((pad_count,) + tail_shape, dtype=array.dtype)
        return np.concatenate((array, tail), axis=0)

    def __len__(self):
        return len(self.train_skel)

    def __iter__(self):
        return self

    def __getitem__(self, indexA):
        local_i = self.train_local[indexA]
        global_i = self.train_global[indexA]
        skel_i = self.train_skel[indexA]
        quat_i = self.all_quats[indexA]

        n_joints = local_i.shape[1]
        max_len = self.max_length
        seq_len = local_i.shape[0]
        stidx = self._crop_start(seq_len, max_len)

        mask = torch.zeros((max_len,), dtype=torch.float32)
        aeReg = torch.zeros((1,), dtype=torch.float32)
        heightA = torch.zeros((1,), dtype=torch.float32)
        heightB = torch.zeros((1,), dtype=torch.float32)

        clocalA = local_i[stidx : stidx + max_len]
        valid_len = min(max_len, clocalA.shape[0])
        mask[:valid_len] = 1.0
        clocalA = self._pad_time(clocalA, max_len, (n_joints, 3))

        cglobalA = global_i[stidx : stidx + max_len]
        cglobalA = self._pad_time(cglobalA, max_len, (GLOBAL_DIM,))

        cskelA = skel_i[stidx : stidx + max_len]
        cskelA = self._pad_time(cskelA, max_len, (n_joints, 3))

        cquatA = quat_i[stidx : stidx + max_len]
        cquatA = self._pad_time(cquatA, max_len, (n_joints, 4), identity_quat=True)

        indexB = np.random.randint(len(self.train_skel))
        cskelB = self.train_skel[indexB][0:max_len]
        cskelB = self._pad_time(cskelB, max_len, (n_joints, 3))

        joints_a = (cskelA[0][None] * self.local_std) + self.local_mean
        height_a = self.get_height_from_skel(joints_a[0]) / 100.0

        joints_b = (cskelB[0][None] * self.local_std) + self.local_mean
        height_b = self.get_height_from_skel(joints_b[0]) / 100.0

        aeReg_on = np.random.binomial(1, p=0.5)
        if aeReg_on:
            cskelB = cskelA.copy()
            aeReg[0] = 1
            heightA[0] = height_a
            heightB[0] = height_a
            indexB = indexA
        else:
            aeReg[0] = 0
            heightA[0] = height_a
            heightB[0] = height_b

        localA = clocalA.reshape((max_len, -1))
        globalA = cglobalA.reshape((max_len, -1))
        seqA = np.concatenate((localA, globalA), axis=-1).astype(np.float32)
        skelA = cskelA.reshape((max_len, -1)).astype(np.float32)
        quatA = cquatA.astype(np.float32)

        localB = clocalA.reshape((max_len, -1))
        globalB = cglobalA.reshape((max_len, -1))
        seqB = np.concatenate((localB, globalB), axis=-1).astype(np.float32)
        skelB = cskelB.reshape((max_len, -1)).astype(np.float32)

        shapeA = self.shape_dic[self.all_names[indexA]].reshape(-1)
        shapeB = self.shape_dic[self.all_names[indexB]].reshape(-1)

        return (
            indexA,
            indexB,
            seqA,
            skelA,
            seqB,
            skelB,
            aeReg,
            mask,
            heightA,
            heightB,
            shapeA,
            shapeB,
            quatA,
        )

    @staticmethod
    def get_height_from_skel(skel):
        """
        Quadruped scale proxy aligned with the Mixamo design:
        spine chain + one hind-leg lower chain.
        """
        diffs = np.sqrt((skel ** 2).sum(axis=-1))
        spine_len = diffs[1:7].sum()
        hind_len = diffs[19:22].sum()
        return spine_len + hind_len


if __name__ == "__main__":
    pass
