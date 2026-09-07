BLENDER=/home/lutianyi/Softwares/blender-4.5.3-linux-x64/blender

$BLENDER -b -P ./fbx2bvh_smal33.py -- \
  --data_path ./shepherd/smal@shepherd/train_char \
  --max_cpu_cores 9 \
  --workers 3 \
  --threads_per_worker 1 \
  --overwrite_existing \
  --mocap_y_up \

python ./preprocess_q_smal33.py \
  --data_path ./shepherd/smal@shepherd/train_char \
  --save_path ./shepherd/smal@shepherd/train_q \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --forward_mode body \

$BLENDER -b -P ./extract_shape_smal33.py -- \
  --fbx_root ./shepherd/smal@shepherd/train_char \
  --save_path ./shepherd/smal@shepherd/train_shape \
  --one_npz_per_subdirectory \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --mocap_y_up \

$BLENDER -b -P ./fbx2bvh_smal33.py -- \
    --data_path ./shepherd/batch3_dogs/batch3_dogs_char \
    --overwrite_existing \
    --max_cpu_cores 9 \
    --workers 3 \
    --threads_per_worker 1 \
    --force_rest_pose \
    --mocap_y_up \

python ./preprocess_q_smal33.py \
  --data_path ./shepherd/batch3_dogs/batch3_dogs_char \
  --save_path ./shepherd/batch3_dogs/batch3_dogs_q \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --forward_mode body \

$BLENDER -b -P ./extract_shape_smal33.py -- \
  --fbx_root ./shepherd/batch3_dogs/batch3_dogs_char \
  --save_path ./shepherd/batch3_dogs/batch3_dogs_shape \
  --overwrite_existing \
  --all_fbx_per_subdirectory \
  --axis_transform shepherd_y_z_x \
  --mocap_y_up \