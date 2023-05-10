import argparse
import os
import shutil

#-------------------------------------------------------#
#   本代码用于合并生成数据集（增强、伪标签）和原始数据集
#-------------------------------------------------------#

parser = argparse.ArgumentParser(description='Arguments of Semi-supervised and Self-training')
# 添加命令行参数
parser.add_argument('--origin_image_path', type=str, default='VOCdevkit/VOC2007/JPEGImages')
parser.add_argument('--origin_xml_path', type=str, default='VOCdevkit/VOC2007/Annotations')
parser.add_argument('--new_image_path', type=str, default='Semi-Supervised/JPEGImages')
parser.add_argument('--new_xml_path', type=str, default='Semi-Supervised/Annotations')
# 从命令行中结构化解析参数
args = parser.parse_args()

origin_image_path = args.origin_image_path
origin_xml_path = args.origin_xml_path
new_image_path = args.new_image_path
new_xml_path = args.new_xml_path
#-------------------------------------------------------#
#   移动图片
#-------------------------------------------------------#
imgs = os.listdir(new_image_path)
for img in imgs:
    shutil.move(os.path.join(new_image_path, img), os.path.join(origin_image_path, img))
#-------------------------------------------------------#
#   移动标注
#-------------------------------------------------------#
xmls = os.listdir(new_xml_path)
for xml in xmls:
    shutil.move(os.path.join(new_xml_path, xml), os.path.join(origin_xml_path, xml))

print('数据集合并完成！')