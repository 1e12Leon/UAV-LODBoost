
# -*- encoding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import shutil

#根据自己的情况修改相应的路径
ann_filepath = 'VOCdevkit/VisDrone2019-DET-train/Annotations/'
img_filepath = 'VOCdevkit/VisDrone2019-DET-train/JPEGImages/'
img_savepath = 'VOCdevkit/VisDrone2019-DET-train/JPEGImages4/'
ann_savepath = 'VOCdevkit/VisDrone2019-DET-train/Annotations4/'
if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)

if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)

# 这是VOC数据集中所有类别
# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#             'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#              'dog', 'horse', 'motorbike', 'pottedplant',
#           'sheep', 'sofa', 'train', 'person','tvmonitor']

classes = ['pedestrian', 'bicycle', 'car', 'van']    #这里是需要提取的类别
# ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
def save_annotation(file):
    tree = ET.parse(ann_filepath + '/' + file)
    root = tree.getroot()
    result = root.findall("object")
    bool_num = 0
    for obj in result:
        if obj.find("name").text not in classes:
            root.remove(obj)
        else:
            bool_num = 1
    if bool_num:
        tree.write(ann_savepath + file)
        return True
    else:
        return False

def save_images(file):
    name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
    shutil.copy(name_img, img_savepath)
    #文本文件名自己定义，主要用于生成相应的训练或测试的txt文件
    """with open('test/test.txt', 'a') as file_txt:
        file_txt.write(os.path.splitext(file)[0])
        file_txt.write("\n")"""
    return True

if __name__ == '__main__':
    for f in os.listdir(ann_filepath):
        if save_annotation(f):
            save_images(f)