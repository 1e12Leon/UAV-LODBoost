import cv2
import numpy as np
from PIL import Image
import os

from tqdm import tqdm

from yolo import YOLO
import xml.etree.ElementTree as ET


def pretty_xml(element, indent, newline, level=0):
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def Coordinate2XML(imagepath, outputxmlpath, dets, folder):
    """
    dets:[top(ymin), left(xmin), bottom(ymax), right(xmax), score, predicted_class]
    """
    xml_file_path = 'Semi-Supervised/anno.xml'
    annotation_dir = '/Annotations/'
    img_file_name = os.path.split(imagepath)[1]
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for i, det in enumerate(dets):
        # txt中的第一条information在写入xml时，负责将img_name、size、object写入；其余的information只写object
        if i == 0:
            lable = det[5]
            xmin = det[1]
            ymin = det[0]
            xmax = det[3]
            ymax = det[2]
            root.find('filename').text = img_file_name
            root.find('folder').text = folder
            root.find('path').text = imagepath.replace('\\', '/')
            # size
            sz = root.find('size')
            img = cv2.imread(imagepath)  # 读取图片信息
            # cv2.imshow("img",img)
            # cv2.waitKey(0)
            sz.find('height').text = str(img.shape[0])
            sz.find('width').text = str(img.shape[1])
            sz.find('depth').text = str(img.shape[2])

            # 第一个object
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = lable
            pose = ET.SubElement(obj, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = "0"
            bndbox = ET.SubElement(obj, 'bndbox')
            bndbox_xmin = ET.SubElement(bndbox, 'xmin', )
            bndbox_xmin.text = str(xmin)
            bndbox_ymin = ET.SubElement(bndbox, 'ymin')
            bndbox_ymin.text = str(ymin)
            bndbox_xmax = ET.SubElement(bndbox, 'xmax')
            bndbox_xmax.text = str(xmax)
            bndbox_ymax = ET.SubElement(bndbox, 'ymax')
            bndbox_ymax.text = str(ymax)

        else:
            lable = det[5]
            xmin = det[1]
            ymin = det[0]
            xmax = det[3]
            ymax = det[2]

            # 建立新的object元素
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = lable
            pose = ET.SubElement(obj, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = "0"

            # 建立bndbox信息
            bndbox = ET.SubElement(obj, 'bndbox')
            bndbox_xmin = ET.SubElement(bndbox, 'xmin', )
            bndbox_xmin.text = str(xmin)
            bndbox_ymin = ET.SubElement(bndbox, 'ymin')
            bndbox_ymin.text = str(ymin)
            bndbox_xmax = ET.SubElement(bndbox, 'xmax')
            bndbox_xmax.text = str(xmax)
            bndbox_ymax = ET.SubElement(bndbox, 'ymax')
            bndbox_ymax.text = str(ymax)

    xml_file = img_file_name.replace('jpg', 'xml')
    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write(os.path.join(outputxmlpath, xml_file), encoding='utf-8')


if __name__ == '__main__':
    # threshold = 0.2
    yolo = YOLO()
    imgs_path = 'Semi-Supervised/JPEGImages'
    outputxmlpath = 'Semi-Supervised/Annotations'
    folder = 'JPEGImages'
    img_list = os.listdir(imgs_path)
    for img in tqdm(img_list):
        img = os.path.join(imgs_path, img)
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            dets = yolo.detect_image_pseudo_label(image)
            Coordinate2XML(img, outputxmlpath, dets, folder)