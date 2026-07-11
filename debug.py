"""
import numpy as np
d = np.load("./work_dir/train_shapeaware_smal33/inspect/self_root_centered_eurasian_lynx_female_to_eurasian_lynx_female_eurasian_lynx_female@enrichmentboxshake/predictions.npz")
err = np.linalg.norm(d["input_world"] - d["stage2_world"], axis=-1)
print("mean", err.mean(), "p95", np.percentile(err, 95), "max", err.max())
# 前肢关节 id 可单独看 mean err
"""

import numpy as np

def body_forward_xz(joints):  # joints: (33,3)
    hip = (joints[18] + joints[22]) / 2
    sh  = (joints[8] + joints[12]) / 2
    v = sh - hip
    v[1] = 0
    v /= np.linalg.norm(v) + 1e-8
    return v

d = np.load("./work_dir/train_shapeaware_smal33/inspect/self_root_centered_shepherd@Digging_to_shepherd@Digging_shepherd@Digging_loop_smal_dog-foot_on_ground/predictions.npz")
f0 = d["input_world"][0]          # 只看绿色源
fwd = body_forward_xz(f0)
print("forward_xz", fwd)

# 帧间平滑度（突变）
vel = np.linalg.norm(np.diff(d["input_world"], axis=0), axis=-1).mean(axis=1)
print("input frame jerk mean", np.abs(np.diff(vel)).mean())
print("stage2 frame jerk mean", np.abs(np.diff(
    np.linalg.norm(np.diff(d["stage2_world"], axis=0), axis=-1).mean(axis=1)
)).mean())