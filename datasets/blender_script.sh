BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./Planet_Zoo_FBX-cleaned/train_char \
#   --max_cpu_cores 12 \
#   --workers 3 \
#   --threads_per_worker 1 \
#   # --overwrite_existing \

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./Planet_Zoo_FBX-smal2/fbx-bw2 \
#   --save_path ./Planet_Zoo_FBX-smal2/train_shape

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/smal@shepherd/train_char \
#   --max_cpu_cores 8 \
#   --workers 2 \
#   --threads_per_worker 1 \
#   --overwrite_existing \
#   --mocap_y_up \

# $BLENDER -b -P ./check_fbx_frame_stats.py -- \
#   --data_path ./shepherd/smal@shepherd/train_char \
#   --json_out ./shepherd/smal@shepherd_frame_stats.json \

# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#     --blend_root ./shepherd/batch2_dogs/batch2_dogs_char \
#     --output_root ./shepherd/batch2_dogs/batch2_dogs_fbx \

# 错：
# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/smal@shepherd/train_char \
#   --save_path ./shepherd/smal@shepherd/train_shape_old \
#   --one_npz_per_subdirectory

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#     --data_path ./shepherd/batch2_dogs/batch2_dogs_char \
#     --overwrite_existing \
#     --max_cpu_cores 9 \
#     --workers 3 \
#     --threads_per_worker 1 \
#     --force_rest_pose \
#     --mocap_y_up \

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/batch2_dogs/batch2_dogs_char \
#   --save_path ./shepherd/batch2_dogs/batch2_dogs_shape \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \
#   --all_fbx_per_subdirectory

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/smal@shepherd/train_char \
#   --save_path ./shepherd/smal@shepherd/train_shape \
#   --one_npz_per_subdirectory \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \
 
# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#     --blend_root ./shepherd/batch2_dogs/batch2_dogs_gt \
#     --output_root ./shepherd/batch2_dogs/batch2_dogs_char \

# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#   --blend_root ./shepherd/batch2_dogs/batch2_dogs_gt \
#   --output_root ./shepherd/batch2_dogs/batch2_dogs_char \
#   --overwrite_existing \
#   --path_mode COPY \
#   --embed_textures

# Static multi-subject rest-pose export (batch2_dogs etc.):
# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#   --blend_root ./shepherd/dog_actions/train_blend \
#   --output_root ./shepherd/dog_actions/train_char \
#   --overwrite_existing \
#   --path_mode COPY \
#   --embed_textures

# Animated shepherd motion clips: train_blend -> train_char (one FBX per .blend)
# Smoke test first:
# $BLENDER -b -P ./shepherd/blend_motion_to_fbx.py -- \
#   --blend_root ./shepherd/dog_actions/train_blend \
#   --output_root ./shepherd/dog_actions/train_char \
#   --list_only \
#   --limit_blends 3

# # Full export:
# $BLENDER -b -P ./shepherd/blend_motion_to_fbx.py -- \
#   --blend_root ./shepherd/dog_actions/train_blend \
#   --output_root ./shepherd/dog_actions/train_char \
#   --overwrite_existing \
#   --path_mode COPY \
#   --embed_textures

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/dog_actions/train_char \
#   --max_cpu_cores 12 \
#   --workers 3 \
#   --threads_per_worker 1 \
#   --overwrite_existing \
#   --mocap_y_up \

# ARP cat_actions / batch2_dogs: bake +90 yaw into BVH so facing matches dog_actions.
# Then preprocess_q_smal33 can omit --post_axis_yaw_deg (same as dog_actions).
# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/cat_actions/train_char \
#   --max_cpu_cores 8 \
#   --workers 2 \
#   --threads_per_worker 2 \
#   --overwrite_existing \
#   --mocap_y_up \
#   --bvh_yaw_deg 90 \

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/batch2_dogs/batch2_dogs_char \
#   --max_cpu_cores 8 \
#   --workers 2 \
#   --threads_per_worker 2 \
#   --overwrite_existing \
#   --mocap_y_up \
#   --bvh_yaw_deg 90 \
#   --force_rest_pose \

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/cat_actions/train_char \
#   --max_cpu_cores 8 \
#   --workers 2 \
#   --threads_per_worker 2 \
#   --overwrite_existing \
#   --mocap_y_up \
#   --bvh_yaw_deg 90

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/cat_actions/train_char \
#   --save_path ./shepherd/cat_actions/train_shape \
#   --one_npz_per_subdirectory \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \

# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#   --blend_root ./shepherd/batch2_dogs_1/mia25-batch2_gt \
#   --output_root ./shepherd/batch2_dogs_1/batch2_dogs_char \
#   --overwrite_existing \
#   --path_mode COPY \
#   --embed_textures

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#     --data_path ./shepherd/batch2_dogs_1/batch2_dogs_char \
#     --overwrite_existing \
#     --max_cpu_cores 9 \
#     --workers 3 \
#     --threads_per_worker 1 \
#     --force_rest_pose \
#     --mocap_y_up \

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/batch2_dogs_1/batch2_dogs_char \
#   --save_path ./shepherd/batch2_dogs_1/batch2_dogs_shape \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \
#   --all_fbx_per_subdirectory

# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#     --blend_root ./shepherd/batch3_dogs/batch3_dogs_gt \
#     --output_root ./shepherd/batch3_dogs/batch3_dogs_char \
# 
# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#     --data_path ./shepherd/batch3_dogs/batch3_dogs_char \
#     --overwrite_existing \
#     --max_cpu_cores 12 \
#     --workers 3 \
#     --threads_per_worker 1 \
#     --force_rest_pose \
#     --mocap_y_up 
# 
# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/batch3_dogs/batch3_dogs_char \
#   --save_path ./shepherd/batch3_dogs/batch3_dogs_shape \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \
#   --all_fbx_per_subdirectory

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#     --data_path ./shepherd/cat_actions/train_char \
#     --overwrite_existing \
#     --max_cpu_cores 12 \
#     --workers 3 \
#     --threads_per_worker 1 \

# $BLENDER -b -P ./extract_shape_smal33.py -- \
#   --fbx_root ./shepherd/cat_actions/train_char \
#   --save_path ./shepherd/cat_actions/train_shape \
#   --one_npz_per_subdirectory \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing \

# $BLENDER -b -P ./fbx2bvh_smal33.py -- \
#   --data_path ./shepherd/cat_actions_sucaibao/train_char \
#   --max_cpu_cores 9 \
#   --workers 3 \
#   --threads_per_worker 1 \
#   --overwrite_existing \
#   --mocap_y_up \

# $BLENDER -b -P ./extract_shape_sucaibao_cat.py -- \
#   --fbx_root ./shepherd/cat_actions_sucaibao/train_char \
#   --save_path ./shepherd/cat_actions_sucaibao/train_shape \
#   --one_npz_per_subdirectory \
#   --axis_transform shepherd_y_z_x \
#   --mocap_y_up \
#   --overwrite_existing

# $BLENDER -b -P ./shepherd/blend_subjects_to_fbx.py -- \
#   --blend_root ./shepherd/batch4_dogs/batch4_dogs_gt \
#   --output_root ./shepherd/batch4_dogs/batch4_dogs_char \
#   --overwrite_existing \
#   --path_mode COPY \
#   --embed_textures

$BLENDER -b -P ./fbx2bvh_smal33.py -- \
    --data_path ./shepherd/batch4_dogs/batch4_dogs_char \
    --overwrite_existing \
    --max_cpu_cores 12 \
    --workers 3 \
    --threads_per_worker 1 \
    --force_rest_pose \
    --mocap_y_up 

$BLENDER -b -P ./extract_shape_smal33.py -- \
  --fbx_root ./shepherd/batch4_dogs/batch4_dogs_char \
  --save_path ./shepherd/batch4_dogs/batch4_dogs_shape \
  --axis_transform shepherd_y_z_x \
  --mocap_y_up \
  --overwrite_existing \
  --all_fbx_per_subdirectory