# python inspect_stage1_outputs_smal33.py \
#   --config config/inference_bvh_smal33_cfg.yaml \
#   --device 0 \
#   --save_path work_dir/train_skeleton_aware_smal33/inspect \
#   --self_recon \
#   --compare_spacing 0.5 \
#   --frame_margin 0.03 \
#   --show_target_rest \

# python inspect_stage1_outputs_smal33.py \
#   --config config/inference_bvh_smal33_cfg.yaml \
#   --device 2 \
#   --save_path work_dir/train_skeleton_aware_smal33/inspect \
#   --compare_spacing 0.5 \
#   --frame_margin 0.05


# python eval_skeleton_aware_smal33.py \
#   --config ./config/eval_skeleton_aware_smal33.yaml \
#   --device 0 \

# python eval_shape_aware_smal33.py \
#   --config ./config/eval_shape_aware_smal33.yaml \
#   --num_cross_pairs 100 \
#   --num_self_samples 100 \
#   # --checkpoint_glob "./work_dir/train_shapeaware_smal33/r2et_shape_aware_smal33_ret-30.pt"

# python inspect_shape_aware_smal33.py \
#   --config ./config/inspect_shape_aware_smal33_cfg.yaml \
#   --compare_spacing 0.2 \
#   --mesh_spacing 0.30 \
#   --frame_margin 0.05 \
#   --view_azim 45 \
#   --device 2 \
#   --self_recon \
#   --show_target_rest

# python eval_skeleton_aware_smal33.py \
#     --config config/eval_shepherd_skeleton_aware_smal33.yaml

# python inspect_skeleton_aware_smal33.py \
#   --config config/inspect_shepherd_skeleton_aware_smal33_cfg.yaml \
#   --device 2 \
#   --self_recon \
#   --compare_spacing 0.5 \
#   --frame_margin 0.03 \
#   --show_target_rest \

# python inspect_skeleton_aware_smal33.py \
#   --config config/inspect_shepherd_skeleton_aware_smal33_cfg.yaml \
#   --device 2 \
#   --compare_spacing 0.5 \
#   --frame_margin 0.05 \

# python datasets/compare_bind_pose_smal33.py \
#   --shepherd_q ./datasets/shepherd/smal@shepherd/train_q \
#   --batch2_q ./datasets/shepherd/batch2_dogs/batch2_dogs_q \
#   --shepherd_bvh_root ./datasets/shepherd/smal@shepherd/train_char \
#   --batch2_bvh_root ./datasets/shepherd/batch2_dogs/batch2_dogs_char \
#   --axis_transform shepherd_y_z_x \
#   --shepherd_forward_mode body \
#   --batch2_forward_mode body \
#   --compare_bvh 30 \
#   --output ./datasets/bind_pose_compare.json

# python inspect_shape_aware_smal33.py \
#   --config ./config/inspect_shepherd_shape_aware_smal33_cfg.yaml \
#   --compare_spacing 0.2 \
#   --mesh_spacing 0.30 \
#   --frame_margin 0.05 \
#   --view_azim 45 \
#   --device 0 \
#   # --self_recon \
#   --show_target_rest

# python visualization/batch_fourway_compare_from_csv.py \
#   --csv temp/clipping_bad_dogs.csv \
#   --config config/visualization_compare_smal33.yaml \
#   --dry_run

# python visualization/batch_fourway_compare_from_csv.py \
#   --csv temp/clipping_bad_dogs.csv \
#   --config config/visualization_compare_smal33.yaml \
#   --limit 1 \
#   --video_output_dir visualization/videos/compare/demos \
#   # --keep_intermediates

# export BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender
# python visualization/batch_fourway_compare_from_csv.py \
#   --csv temp/clipping_bad_dogs.csv \
#   --config config/visualization_compare_smal33.yaml \
#   --video_output_dir visualization/videos/compare/demos_17 \
#   --gpus 2,3 \
#   --num_workers 2 \
#   --total_cpu_threads 12
#   # --cpu_threads_per_worker 4 \

# python visualization/diagnose_arp_lbs.py \
#   --config config/visualization_compare_smal33.yaml \
#   --manifest temp/fourway_batch_o7gwlv4r/work/Attack_Bite_RM_to_边牧_4/fourway_manifest.json \
#   --output temp/arp_lbs_diag_Attack_Bite_RM_to_边牧_4.json

# python visualization/diagnose_compare_lane_motion.py \
#   --npz "temp/fourway_batch_h9cviuwk/work/Attack_Bite_RM_to_边牧_4/fourway_compare.npz" \
#   --reference_lane copyquat \
#   --out_json "temp/diag_compare_lane_motion.json"

# python visualization/diagnose_arp_pipeline_stages.py \
#   --fourway_npz "temp/fourway_batch_7wty46p1/work/Attack_Bite_RM_to_边牧_4/fourway_compare.npz" \
#   --arp_mesh_npz "temp/fourway_batch_7wty46p1/work/arp_mesh_outputs/Attack_Bite_RM_to_边牧_4_arp_mesh.npz" \
#   --reference_lane copyquat \
#   --lock_anchor first \
#   --out_json "temp/diag_arp_pipeline_stages.json"

# python visualization/diagnose_arp_lane_quality.py \
#   --npz   temp/fourway_batch_s7gl0th7/work/Attack_Bite_RM_to_边牧_4/fourway_compare.npz \
#   --arp_mesh temp/fourway_batch_s7gl0th7/work/arp_mesh_outputs/Attack_Bite_RM_to_边牧_4_arp_mesh.npz \
#   --source_lane source \
#   --out_json temp/arp_lane_quality_边牧_4.json

# export BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender
# python visualization/batch_r2et_dog_blend.py \
#   --dog_ids 博美_3 \
#   --gpus 0 \
#   --limit_actions 3 \
#   --keep_intermediates

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --dog_ids 博美_3 \
#   --limit_actions 1 \
#   --gpus 0 \
#   --keep_intermediates \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_diag

# WORK=temp/r2et_dog_blend_wdm24993/work
# MANIFEST=$WORK/博美_3/dog_actions_manifest.json
# BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender
# 
# # 从 manifest 读路径
# python - <<'PY'
# import json
# from pathlib import Path
# import os
# m=json.loads(Path(os.environ["MANIFEST"]).read_text())
# a=m["actions"][0]
# print("FBX=", m["target_fbx_path"])
# print("RAW_BVH=", m["target_bvh_path"])
# print("OURS_MODEL=", a.get("ours_bvh_path"))
# print("OURS_NATIVE=", a.get("ours_native_bvh_path"))
# print("ACTION=", a["action_id"])
# PY

# python visualization/diagnose_r2et_blend_binding.py \
#   --manifest "$MANIFEST" \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --action_id Attack_Bite_RM \
#   --output temp/diagnose_bomei3_file.json


# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --dog_ids 比熊_5 \
#   --limit_actions 2 \
#   --gpus 0 \
#   --keep_intermediates \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_try

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --gpus 2,3 \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_cat \
#   # --min_frames 48

# python visualization/batch_arp_sequence_smal33.py \
#   --config config/visualization_arp_sequence_smal33.yaml \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --retarget_mode direct \
#   --show_source
  # --arp_addon_modules auto_rig_pro-master \
  # --show_source
  # --keep_intermediates
  # --stage2_mode stage1 \
  # --show_source \
  # --no_show_source
#   # --arp_addon_modules auto_rig_pro-master \
#   # --keep_intermediates

# python inspect_skeleton_aware_smal33.py \
#   --config config/inspect_shepherd_skeleton_aware_smal33_cfg.yaml \
#   --device 0 \
#   --compare_spacing 0.5 \
#   --frame_margin 0.03 \
#   # --show_target_rest \
#   # --self_recon \

# python visualization/batch_arp_export_fbx_smal33.py \
#   --config config/visualization_arp_export_fbx_smal33.yaml \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --arp_addon_modules auto_rig_pro-master

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --gpus 0 \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_dog_smal@shepherd \
#   --min_frames 24

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --retarget_mode direct \
#   --gpus 1,2 \
#   --cpu_threads_per_worker 4 \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_new_dog \
# #   --min_frames 16

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --retarget_mode arp \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --arp_addon_modules auto_rig_pro-master \
#   --gpus 0 \
#   --blend_output_dir visualization/blend_new_dog_arp \


# Export cat_actions SMAL .blend clips -> armature-only FBX (no mesh).
# Requires --allow_armature_only because train_char_gt blends have armature+anim only.
# blender -b -P ./datasets/shepherd/blend_motion_to_fbx.py -- \
#   --blend_root ./datasets/shepherd/cat_actions/train_char_gt \
#   --output_root ./datasets/shepherd/cat_actions/train_char \
#   --allow_armature_only \
#   --overwrite_existing \
#   --json_out ./datasets/shepherd/cat_actions/train_char/blend_motion_export_summary.json

# python inspect_shape_aware_smal33.py \
#   --config ./config/inspect_shepherd_shape_aware_smal33_cfg.yaml \
#   --compare_spacing 0.2 \
#   --mesh_spacing 0.30 \
#   --frame_margin 0.05 \
#   --view_azim 45 \
#   --device 0 \

python visualization/batch_arp_sequence_smal33.py \
  --config config/visualization_arp_sequence_smal33.yaml \
  --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
  --retarget_mode direct \
  --show_source \
  # --arp_addon_modules auto_rig_pro-master \
  # --keep_intermediates
  # --stage2_mode stage1 \
  # --no_show_source

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_cat_smal33.yaml \
#   --retarget_mode arp \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --arp_addon_modules auto_rig_pro-master \
#   --gpus 1,2 \
#   --blend_output_dir visualization/blend_retarget_cats_new \

# python visualization/batch_r2et_dog_blend.py \
#   --config config/visualization_blend_per_dog_smal33.yaml \
#   --retarget_mode direct \
#   --gpus 1,2 \
#   --cpu_threads_per_worker 4 \
#   --blender /home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender \
#   --blend_output_dir visualization/blend_retarget_dogs_new \