# --blender可以指定自己的blender安装路径
python visualization/batch_arp_sequence_smal33.py \
  --config config/visualization_arp_sequence_smal33_dog.yaml \
  --retarget_mode direct \
  --show_source

# --blender可以指定自己的blender安装路径
python visualization/batch_arp_sequence_smal33.py \
  --config config/visualization_arp_sequence_smal33_cat.yaml \
  --retarget_mode arp \
  --arp_addon_modules auto_rig_pro-master \
  --show_source