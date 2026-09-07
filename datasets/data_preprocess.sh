$BLENDER -b -P ./fbx2bvh_smal33.py -- \
  --data_path ./shepherd/smal@shepherd_dog/train_char \
  --max_cpu_cores 9 \
  --workers 3 \
  --threads_per_worker 1 \
  --overwrite_existing \
  --mocap_y_up \

python ./preprocess_q_smal33.py \
  --data_path ./shepherd/smal@shepherd_dog/train_char \
  --save_path ./shepherd/smal@shepherd_dog/train_q \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --forward_mode body \

$BLENDER -b -P ./extract_shape_smal33.py -- \
  --fbx_root ./shepherd/smal@shepherd_dog/train_char \
  --save_path ./shepherd/smal@shepherd_dog/train_shape \
  --one_npz_per_subdirectory \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --mocap_y_up \

$BLENDER -b -P ./fbx2bvh_smal33.py -- \
  --data_path ./shepherd/smal@shepherd_cat/train_char \
  --max_cpu_cores 9 \
  --workers 3 \
  --threads_per_worker 1 \
  --overwrite_existing \
  --mocap_y_up \
  --bvh_yaw_deg 90

python ./preprocess_q_smal33.py \
  --data_path ./shepherd/smal@shepherd_cat/train_char \
  --save_path ./shepherd/smal@shepherd_cat/train_q \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --forward_mode body \

$BLENDER -b -P ./extract_shape_smal33.py -- \
  --fbx_root ./shepherd/smal@shepherd_cat/train_char \
  --save_path ./shepherd/smal@shepherd_cat/train_shape \
  --one_npz_per_subdirectory \
  --overwrite_existing \
  --axis_transform shepherd_y_z_x \
  --mocap_y_up \