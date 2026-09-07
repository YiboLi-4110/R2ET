# python compute_stats_smal33.py \
#   --data_path ./shepherd/smal@shepherd/train_q \
#   --shape_path ./shepherd/smal@shepherd/train_shape \
#   --stats_path ./shepherd/smal@shepherd/stats \
#   --min_frames 32

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/smal@shepherd/train_char \
#   --save_path ./shepherd/smal@shepherd/train_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

# After fbx2bvh with --bvh_yaw_deg 90 for ARP cat / batch2, preprocess like dog
# (no --post_axis_yaw_deg):
# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/batch2_dogs/batch2_dogs_char \
#   --save_path ./shepherd/batch2_dogs/batch2_dogs_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

# python ./check_smal33_preprocess.py \
#   --npy-root ./shepherd/smal@shepherd/train_q \
#   --bvh-root ./shepherd/smal@shepherd/train_char \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --roundtrip \
#   > output.log

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/smal@shepherd/train_char \
#   --save_path ./shepherd/smal@shepherd/train_q_old \

# python ./check_smal33_preprocess.py \
#   --npy-root ./shepherd/batch2_dogs/batch2_dogs_q \
#   --bvh-root ./shepherd/batch2_dogs/batch2_dogs_char \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --roundtrip \
#   > batch2_check.json

# python ./check_smal33_preprocess.py \
#   --npy-root ./shepherd/smal@shepherd/train_q \
#   --bvh-root ./shepherd/smal@shepherd/train_char \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --roundtrip \
#   > shepherd_check.json

# # Align batch2 bind-pose skeleton geometry to shepherd (+Z forward in skel space).
# python ./fix_batch2_bind_pose_smal33.py \
#   --q_path ./shepherd/batch2_dogs/batch2_dogs_q \
#   --shape_path ./shepherd/batch2_dogs/batch2_dogs_shape \
#   --forward_mode body \
#   --report ./shepherd/batch2_dogs/bind_pose_fix_report.json

# python list_shepherd_bvh_min_frames.py \
#   --data_path ./shepherd/smal@shepherd/train_char \
#   --min_frames 90 \
#   --output ./shepherd/smal@shepherd/train_char_bvh_ge90.txt

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/dog_actions/train_char \
#   --save_path ./shepherd/dog_actions/train_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

# ---------------------------------------------------------------------------
# Preferred pipeline for ARP cat_actions / batch2_dogs:
#   1) fbx2bvh with --bvh_yaw_deg 90  (see blender_script.sh)
#   2) preprocess WITHOUT --post_axis_yaw_deg (same as dog_actions)
# Legacy alternative: leave BVH as-is and pass --post_axis_yaw_deg 90 here.
# ---------------------------------------------------------------------------

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/cat_actions/train_char \
#   --save_path ./shepherd/cat_actions/train_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/batch2_dogs/batch2_dogs_char \
#   --save_path ./shepherd/batch2_dogs/batch2_dogs_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

# python ./preprocess_q_smal33.py \
#   --data_path ./shepherd/batch3_dogs/batch3_dogs_char \
#   --save_path ./shepherd/batch3_dogs/batch3_dogs_q \
#   --axis_transform shepherd_y_z_x \
#   --forward_mode body \
#   --overwrite_existing

python ./preprocess_q_smal33.py \
  --data_path ./shepherd/batch4_dogs/batch4_dogs_char \
  --save_path ./shepherd/batch4_dogs/batch4_dogs_q \
  --axis_transform shepherd_y_z_x \
  --forward_mode body \
  --overwrite_existing