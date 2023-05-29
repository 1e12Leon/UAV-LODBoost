import argparse

import cv2
import numpy as np
import shutil
from PIL import Image
import os
from tqdm import tqdm
from yolo import YOLO
import xml.etree.ElementTree as ET


def getBboxNum(ann_path):
    # -------------------------------------------------------------------------#
    #   统计ground truth
    # -------------------------------------------------------------------------#
    # 解析XML格式的标注文件
    tree = ET.parse(ann_path)
    root = tree.getroot()

    # 统计目标框的数目
    bbox_num = len(root.findall("./object/bndbox"))
    return bbox_num


if __name__ == '__main__':
    yolo = YOLO()
    parser = argparse.ArgumentParser(description='Arguments of Sample Selection')
    # 添加命令行参数
    parser.add_argument('--dir_img_path', type=str, default='Semi-Supervised/JPEGImages')
    parser.add_argument('--dir_xml_path', type=str, default='Semi-Supervised/Annotations')
    parser.add_argument('--out_img_path', type=str, default='VOCdevkit/VOC2007/JPEGImages')
    parser.add_argument('--out_xml_path', type=str, default='VOCdevkit/VOC2007/Annotations')
    parser.add_argument('--hard_thresh', type=int, default=1)
    # 从命令行中结构化解析参数
    args = parser.parse_args()
    #-------------------------------------------------------------------------#
    #   dir_img_path     指定了增强后图片的文件夹路径
    #   dir_xml_path     指定了增强后标注的文件夹路径
    #-------------------------------------------------------------------------#
    dir_img_path = args.dir_img_path
    dir_xml_path = args.dir_xml_path
    out_img_path = args.out_img_path
    out_xml_path = args.out_xml_path
    #-------------------------------------------------------------------------#
    #   如何定义困难样本:
    #       模型检测到的目标数目与真实目标数目之间的差距超过一阈值，判定为困难样本
    #   hard_thresh     困难样本判定阈值
    #-------------------------------------------------------------------------#
    hard_thresh = args.hard_thresh

    hard_samples = []
    img_names = os.listdir(dir_img_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_img_path, img_name)
            image = Image.open(image_path)
            dets_num = len(yolo.detect_image_dets(image))

            xml_name = img_name[:-4] + '.xml'
            xml_path = os.path.join(dir_xml_path, xml_name)
            ground_truth_num = getBboxNum(xml_path)

            if ground_truth_num - dets_num >= hard_thresh:
                # -------------------------------------------------------------------------#
                #   移动困难样本到数据集
                # -------------------------------------------------------------------------#
                hard_samples.append(img_name)
                shutil.move(os.path.join(dir_img_path, img_name), os.path.join(out_img_path, img_name))
                shutil.move(os.path.join(dir_xml_path, xml_name), os.path.join(out_xml_path, xml_name))

print("困难样本总数:", len(hard_samples))