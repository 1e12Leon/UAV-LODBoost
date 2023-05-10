#!/bin/bash
#-------------------------------------------------------#
#   一键式半监督自训练：
#   1. 准备未标注图像，放在imgs_path目录下（自行准备）
#   2. 运行getPseudoLabel.py
#   3. 运行merge_data.py合并目标域数据和源域数据
#   4. 运行voc_annotation.py
#   5. 运行train.py
#-------------------------------------------------------#
torchrun getPseudoLabel.py \
  --imgs_path 'Semi-Supervised/JPEGImages' \
  --output_xmlpath 'Semi-Supervised/Annotations'
wait

python3 merge_data.py \
  --origin_image_path 'VOCdevkit/VOC2007/JPEGImages'\
  --origin_xml_path 'VOCdevkit/VOC2007/Annotations'\
  --new_image_path 'Semi-Supervised/JPEGImages'\
  --new_xml_path 'Semi-Supervised/Annotations'
wait

python3 voc_annotation.py
wait

torchrun train.py
wait


