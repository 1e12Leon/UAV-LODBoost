'''
Author: 刘鸿燕 13752614153@163.com
Date: 2022-05-09 10:17:40
LastEditors: 刘鸿燕 13752614153@163.com
LastEditTime: 2022-05-09 11:17:20
FilePath: \VisDrone2019\data_process\visDrone2019_txt2xml.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import datetime
from PIL import Image
from pathlib import Path

FILE = Path(__file__).resolve()
# print("FILE",FILE)
ROOT = FILE.parent.parents[0]  # root directory
print("ROOT", ROOT)


def check_dir(path):
    if os.path.isdir(path):
        print("{}文件路径存在！".format(path))
        pass
    else:
        os.makedirs(path)
        print("{}文件路径创建成功！".format(path))


# 把下面的root_dir路径改成你自己的路径即可
root_dir = ROOT / 'Bytetrack_yolov7/VOCdevkit/VisDrone2019-DET-train'
annotations_dir = root_dir / "annotations/"
image_dir = root_dir / "JPEGImages/"
xml_dir = root_dir / "Annotations/"  # 在工作目录下创建Annotations_XML文件夹保存xml文件
check_dir(xml_dir)
# print("annotation_dir",annotations_dir)
# print("image_dir",image_dir)
# print("xml_dir",xml_dir)

# root_dir = r"D:\object_detection_data\datacovert\VisDrone2019-DET-val/"
# annotations_dir = root_dir+"annotations/"
# image_dir = root_dir + "images/"
# xml_dir = root_dir+"Annotations_XML/"   #在工作目录下创建Annotations_XML文件夹保存xml文件

# 下面的类别也换成你自己数据类别，也可适用于其他的数据集转换
class_name = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
              'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

for filename in os.listdir(annotations_dir):
    fin = open(annotations_dir / filename, 'r')
    image_name = filename.split('.')[0]
    image_path = Path(image_dir).joinpath(image_name + ".jpg")  # 若图像数据是“png”转换成“.png”即可
    img = Image.open(image_path)  # 若图像数据是“png”转换成“.png”即可
    xml_name = Path(xml_dir).joinpath(image_name + '.xml')
    with open(xml_name, 'w') as fout:
        # 写入的xml基本信息
        fout.write('<annotation>' + '\n')
        fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
        fout.write('\t' + '<filename>' + image_name + '.jpg' + '</filename>' + '\n')

        fout.write('\t' + '<source>' + '\n')
        fout.write('\t\t' + '<database>' + 'VisDrone2019-DET' + '</database>' + '\n')
        fout.write('\t\t' + '<annotation>' + 'VisDrone2019-DET' + '</annotation>' + '\n')
        fout.write('\t\t' + '<image>' + 'flickr' + '</image>' + '\n')
        fout.write('\t\t' + '<flickrid>' + 'Unspecified' + '</flickrid>' + '\n')
        fout.write('\t' + '</source>' + '\n')

        fout.write('\t' + '<owner>' + '\n')
        fout.write('\t\t' + '<flickrid>' + 'LJ' + '</flickrid>' + '\n')
        fout.write('\t\t' + '<name>' + 'LJ' + '</name>' + '\n')
        fout.write('\t' + '</owner>' + '\n')

        fout.write('\t' + '<size>' + '\n')
        fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
        fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
        fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
        fout.write('\t' + '</size>' + '\n')

        fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

        for line in fin.readlines():
            line = line.split(',')
            fout.write('\t' + '<object>' + '\n')
            fout.write('\t\t' + '<name>' + class_name[int(line[5])] + '</name>' + '\n')
            fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
            fout.write('\t\t' + '<truncated>' + line[6] + '</truncated>' + '\n')
            fout.write('\t\t' + '<difficult>' + str(int(line[7])) + '</difficult>' + '\n')
            fout.write('\t\t' + '<bndbox>' + '\n')
            fout.write('\t\t\t' + '<xmin>' + line[0] + '</xmin>' + '\n')
            fout.write('\t\t\t' + '<ymin>' + line[1] + '</ymin>' + '\n')
            # pay attention to this point!(0-based)
            fout.write('\t\t\t' + '<xmax>' + str(int(line[0]) + int(line[2]) - 1) + '</xmax>' + '\n')
            fout.write('\t\t\t' + '<ymax>' + str(int(line[1]) + int(line[3]) - 1) + '</ymax>' + '\n')
            fout.write('\t\t' + '</bndbox>' + '\n')
            fout.write('\t' + '</object>' + '\n')

        fin.close()
        fout.write('</annotation>')
