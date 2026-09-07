# python -c "
# import torch
# from sdf import SDF
# faces = torch.zeros((1,3), dtype=torch.int32, device='cuda')
# verts = torch.zeros((1,4,3), dtype=torch.float32, device='cuda', requires_grad=True)
# phi = SDF()(faces, verts, grid_size=8)
# loss = phi.sum()
# loss.backward()
# print('SDF backward OK')
# "


# python datasets/check_smal33_preprocess.py \
#     --npy-root ./datasets/Planet_Zoo_FBX-smal2/train_q \
#     --bvh-root ./datasets/Planet_Zoo_FBX-smal2/train_char \
#     > datasets/output_smal33.log

# python datasets/check_smal33_preprocess.py \
#     --npy-root ./datasets/shepherd/smal@shepherd/train_q \
#     --bvh-root ./datasets/shepherd/smal@shepherd/train_char \
#     > datasets/output.log
#     # > datasets/output_smal@shepherd.log
 
# python datasets/check_smal33_preprocess.py \
#     --npy-root ./datasets/shepherd/batch2_dogs/batch2_dogs_q \
#     --bvh-root ./datasets/shepherd/batch2_dogs/batch2_dogs_char \
#     --axis_transform shepherd_y_z_x \
#     > datasets/output_batch2_dogs.log

# python -c "
# import numpy as np
# d = np.load("./work_dir/train_shapeaware_smal33/inspect/self_root_centered_shepherd@Attack_to_shepherd@Attack_shepherd@Attack_Bite_RM_smal_dog-foot_on_ground/predictions.npz")
# err = np.linalg.norm(d["input_world"] - d["stage2_world"], axis=-1)
# print("mean", err.mean(), "p95", np.percentile(err, 95), "max", err.max())
# # 前肢关节 id 可单独看 mean err
# "

# python debug_axis_search.py \
#   datasets/shepherd/smal@shepherd/train_q/shepherd@Attack/shepherd@Attack_Bite_RM_smal_dog-foot_on_ground_seq.npy \
#   datasets/Planet_Zoo_FBX-smal2/train_q/african_leopard_female/african_leopard_female@matingritual_seq.npy \
#   --top_k 5 \

# python datasets/check_smal33_preprocess.py \
#   --npy-root ./datasets/shepherd/smal@shepherd/train_q_fixed \
#   --bvh-root ./datasets/shepherd/smal@shepherd/train_char \
#   --axis_transform shepherd_y_z_x \
#   > datasets/output.log

# python - <<'EOF'
# import numpy as np
# from pathlib import Path
# 
# stats = Path("./datasets/Planet_Zoo_FBX-smal2/stats")
# files = [
#     "smal33_local_motion_std.npy",
#     "smal33_quat_std.npy",
#     "smal33_shape_std_xyz.npy",
# ]
# 
# for f in files:
#     x = np.load(stats / f).astype(float)
#     flat = x.reshape(-1)
#     print("\n", f, x.shape)
#     print("min/p001/p01/p05/p50/mean/p95/max:")
#     print(np.percentile(flat, [0, 0.1, 1, 5, 50, 100]))
#     for th in [1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]:
#         print(f"count < {th:g}: {(flat < th).sum()} / {flat.size}")
# EOF

# python - <<'EOF'
# import json
# import shutil
# import numpy as np
# from pathlib import Path
# 
# src = Path("./datasets/shepherd/stats")
# dst = Path("./datasets/shepherd/stats_clamp_l005_q005")
# dst.mkdir(parents=True, exist_ok=True)
# 
# for p in src.glob("*.npy"):
#     arr = np.load(p)
#     if p.name in {
#         "smal33_local_motion_std.npy",
#         "mixamo_local_motion_std.npy",
#     }:
#         arr = np.maximum(arr, 0.20)
#     elif p.name in {
#         "smal33_quat_std.npy",
#         "mixamo_quat_std.npy",
#     }:
#         arr = np.maximum(arr, 0.20)
#     np.save(dst / p.name, arr)
# 
# summary = src / "smal33_stats_summary.json"
# if summary.exists():
#     data = json.load(open(summary))
#     data["regularized_stats"] = {
#         "source": str(src),
#         "local_std_floor": 0.05,
#         "quat_std_floor": 0.05,
#     }
#     json.dump(data, open(dst / summary.name, "w"), indent=2)
# EOF

BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender

# $BLENDER --background --python visualization/arp_batch_retarget_blender.py -- \
#   --config config/visualization_compare_smal33.yaml \
#   --check_only

# $BLENDER --background --python visualization/arp_batch_retarget_blender.py -- \
#   --config config/visualization_compare_smal33.yaml \
#   --arp_addon_modules auto_rig_pro-master
# 
# python visualization/export_fourway_compare_smal33.py \
#   --config config/visualization_compare_smal33.yaml
# 
# $BLENDER --background --python visualization/render_fourway_lbs_blender.py -- \
#   --manifest_index visualization/videos/compare/fourway_manifest_index.json \
#   --render_engine eevee \
#   --lane_spacing 1.3
# 
# $BLENDER --background --python visualization/render_fourway_blender.py -- \
#   --manifest_index visualization/videos/compare/fourway_manifest_index.json \
#   --render_engine eevee \
#   --lane_spacing 1.0
# 
# $BLENDER --background --python visualization/debug_fourway_binding_blender.py -- \
#   --manifest_index visualization/videos/compare/fourway_manifest_index.json \
#   --case_id attack_to_bomei3 \
#   --out_json visualization/videos/compare/attack_to_bomei3/binding_debug.json

# $BLENDER --background --python visualization/debug_lbs_texture_ready_blender.py -- \
#   --manifest_index visualization/videos/compare/fourway_manifest_index.json \
#   --case_id attack_to_bomei3 \
#   --out_json visualization/videos/compare/attack_to_bomei3/lbs_texture_ready.json

# $BLENDER --background \
#   --python /home/lutianyi/easyb/R2ET/visualization/diagnose_arp_mesh_spaces_blender.py -- \
#   --config /home/lutianyi/easyb/R2ET/temp/fourway_batch_7wty46p1/work/_configs/Attack_Bite_RM_to_边牧_4.yaml \
#   --case_id Attack_Bite_RM_to_边牧_4 \
#   --fourway_npz /home/lutianyi/easyb/R2ET/temp/fourway_batch_7wty46p1/work/Attack_Bite_RM_to_边牧_4/fourway_compare.npz \
#   --reference_lane copyquat \
#   --lock_anchor first \
#   --out_json /home/lutianyi/easyb/R2ET/temp/diag_arp_mesh_spaces_blender.json \
#   --out_spaces_npz /home/lutianyi/easyb/R2ET/temp/arp_mesh_spaces_Attack_Bite_RM_to_边牧_4.npz

# $BLENDER --background \
#   --python /home/lutianyi/easyb/R2ET/visualization/diagnose_arp_mesh_spaces_blender.py -- \
#   --config /home/lutianyi/easyb/R2ET/temp/fourway_batch_vukeujji/work/_configs/Attack_Bite_RM_to_边牧_4.yaml \
#   --case_id Attack_Bite_RM_to_边牧_4 \
#   --fourway_npz /home/lutianyi/easyb/R2ET/temp/fourway_batch_vukeujji/work/Attack_Bite_RM_to_边牧_4/fourway_compare.npz \
#   --reference_lane copyquat \
#   --lock_anchor first \
#   --operator_sequence auto_scale build_bones_list import_config_preset retarget clear_root_motion \
#   --out_json /home/lutianyi/easyb/R2ET/temp/diag_arp_mesh_spaces_clear_root_motion.json \
#   --out_spaces_npz /home/lutianyi/easyb/R2ET/temp/arp_mesh_spaces_clear_root_motion.npz

# $BLENDER --background --python visualization/diagnose_r2et_blend_binding.py -- \
#   --manifest temp/r2et_dog_blend_7o64k58m/博美_3/dog_actions_manifest.json \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --action_id Attack_J \
#   --blender_fbx_check \
#   --output temp/diagnose_blend_binding_bomei3_blender.json

# 在 R2ET conda 环境里
python visualization/diagnose_r2et_blend_binding.py \
  --manifest temp/r2et_dog_blend_7o64k58m/work/博美_3/dog_actions_manifest.json \
  --config config/visualization_blend_per_dog_smal33.yaml \
  --action_id Attack_J \
  --output temp/diagnose_blend_binding_bomei3.json