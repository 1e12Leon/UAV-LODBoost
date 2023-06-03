import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

import cv2

from my_utils.random_data import get_random_data, get_random_data_with_MixUp
from my_utils.utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始数据集所在的路径
#   Out_VOCdevkit_path      输出数据集所在的路径
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = "VOCdevkit_Origin"
Out_VOCdevkit_path      = "VOCdevkit"
#-----------------------------------------------------------------------------------#
#   Out_Num                 需要利用小目标图像增强生成多少组图片
#-----------------------------------------------------------------------------------#
Out_Num                 = 20
#-----------------------------------------------------------------------------------#
#   Cur_Num                 当前小目标图像增强生成图片的组数
#-----------------------------------------------------------------------------------#
Cur_Num = 0

#-----------------------------------------------------------------------------------#
#   copy_times              小目标图片复制的个数
#-----------------------------------------------------------------------------------#
copy_times = 2

#-----------------------------------------------------------------------------------#
#   MAX_NUM                 各小目标图片复制总个数不超过MAX_NUM
#-----------------------------------------------------------------------------------#
MAX_NUM = 8

# -----------------------------------------------------------------------------------#
#   下面定义了xml里面的组成模块，无需改动。
# -----------------------------------------------------------------------------------#


headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''
#-----------------------------------------------------------------------------------#
#   thresh   小目标尺寸阈值
#   prob
#-----------------------------------------------------------------------------------#
thresh=64*64

# 判断是否为小物体
def is_small_object(h, w):
    if h * w <= thresh:
        return True
    else:
        return False

# 判断两个真实框是否重叠
# annot = [xmin, ymin, xmax, ymax, label]
def compute_overlap(annot_a, annot_b):
    if annot_a is None:
        return False
    left_max = max(annot_a[0], annot_b[0])
    top_max = max(annot_a[1], annot_b[1])
    right_min = min(annot_a[2], annot_b[2])
    bottom_min = min(annot_a[3], annot_b[3])
    inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
    if inter != 0:
        return True
    else:
        return False


# epochs 次数
epochs = 10

# annot表示小目标
# annots表示目标图上的目标框
def create_copy_annot(h, w, annot, annots):
    # annot = annot.astype(np.int)
    annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
    for _ in range(epochs):
        random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
        xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
        xmax, ymax = xmin + annot_w, ymin + annot_h
        if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
            continue
        new_annot = [xmin, ymin, xmax, ymax, annot[4]]

        if donot_overlap(new_annot, annots) is False:
            continue

        return new_annot
    return None

# 判断新生成的框和原图中的真实框是否有重叠 有重叠则返回False
def donot_overlap(new_annot, annots):
    for annot in annots:
        if compute_overlap(new_annot, annot):
            return False
    return True

# image_tar:目标图片 annot:小目标被粘贴的位置 copy_annot:小目标原位置 image_small:小目标图片
def add_patch_in_img(annot, copy_annot, image_tar, image_small):
    annot = np.array(annot).astype(dtype=int).tolist()
    image_tar[annot[1]:annot[3], annot[0]:annot[2], :] = image_small[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
    return image_tar

# 获得小目标的预测框
def get_small_object_list(annotation_line, is_special = False, cls = -1):
    '''

    Args:
        annotation_line: 小目标所在文件
        is_special: 是否获取特定种类的小目标
        cls: 小目标种类索引

    Returns:

    '''
    small_object_list = list()
    line = annotation_line.split()
    #------------------------------#
    #   读取图像并转换成RGB图像
    #------------------------------#
    image = Image.open(line[0])
    image = image.convert('RGB')
    image_data = np.array(image, np.float32)

    # ------------------------------#
    #   获得小目标预测框
    # ------------------------------#
    annots = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    for idx in range(annots.shape[0]):
        annot = annots[idx]
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

        if is_special == True:
            if is_small_object(annot_h, annot_w) and annot[4] == cls:
                small_object_list.append(idx)
        else:
            if is_small_object(annot_h, annot_w):
                small_object_list.append(idx)

    return small_object_list, annots, image_data

def get_random_data_with_SOA(annotation_line, image_small, small_object_list, annots):
    '''
    Args:
        annotation_line: 目标图片
        image_small: 小目标所在图片
        small_object_list: 小目标索引数组
        annots: 小目标annot信息 [[xmin, ymin, xmax, ymax, label], ...]
    Returns:

    '''
    line = annotation_line.split()
    #------------------------------#
    #   读取图像并转换成RGB图像
    #------------------------------#
    image_tar = Image.open(line[0])
    image_tar = image_tar.convert('RGB')
    image_data = np.array(image_tar, np.float32)
    #------------------------------#
    #   获得图像的高宽
    #------------------------------#
    iw, ih = image_tar.size

    # ------------------------------#
    #   获得预测框
    # ------------------------------#
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    #   获得小目标预测框的数量 数量为0则直接返回图片
    # ------------------------------#
    l = len(small_object_list)
    if l == 0:
        return None, None

    # ------------------------------#
    #   copy小目标的种类数量 copy个数不超过小样本预测框个数
    # ------------------------------#
    if l == 1:
        copy_object_num = 1
    else:
        copy_object_num = np.random.randint(1, l)

    print("小目标的种类数量:", copy_object_num)
    random_list = sample(range(l), copy_object_num)
    annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
    select_annots = annots[annot_idx_of_small_object, :]

    new_annots = box.tolist()
    # print("new_annots", np.array(new_annots))

    iter = 0
    for idx in range(copy_object_num):
        annot = select_annots[idx]

        for i in range(copy_times):
            new_annot = create_copy_annot(ih, iw, annot, np.array(new_annots))
            if new_annot is not None:
                # print("new_annot:", new_annot)
                image_data = add_patch_in_img(new_annot, annot, image_data, image_small)
                new_annots.append(new_annot)

                iter += 1
                if iter == MAX_NUM:
                    return image_data, new_annots

    return image_data, new_annots



if __name__ == "__main__":
    Origin_JPEGImages_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")

    Out_JPEGImages_path = os.path.join(Out_VOCdevkit_path, "VOC2007/JPEGImages")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "VOC2007/Annotations")

    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)

    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    xml_names = os.listdir(Origin_Annotations_path)

    def write_xml(anno_path, jpg_pth, head, input_shape, boxes, unique_labels, tail):
        f = open(anno_path, "w")
        f.write(head%(jpg_pth, input_shape[0], input_shape[1], 3))
        for i, box in enumerate(boxes):
            f.write(objstr%(str(unique_labels[int(box[4])]), box[0], box[1], box[2], box[3]))
        f.write(tail)

    while Cur_Num < Out_Num:
        #------------------------------#
        #   获取两个图像与标签
        #------------------------------#
        sample_xmls = sample(xml_names, 2)
        unique_labels = get_classes(sample_xmls, Origin_Annotations_path)

        if 'gtz' in unique_labels:
            cls = unique_labels.index('gtz')
        else:
            cls = ''

        jpg_name_1 = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[0])[0] + '.jpg')
        jpg_name_2 = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls[1])[0] + '.jpg')
        xml_name_1 = os.path.join(Origin_Annotations_path, sample_xmls[0])
        xml_name_2 = os.path.join(Origin_Annotations_path, sample_xmls[1])

        line_1 = convert_annotation(jpg_name_1, xml_name_1, unique_labels)
        line_2 = convert_annotation(jpg_name_2, xml_name_2, unique_labels)

        input_shape = Image.open(line_1.split()[0]).size
        #------------------------------#
        #  小样本目标检测
        #------------------------------#

        # 复制特定种类的小目标
        small_object_list, annots, image_small = get_small_object_list(line_2, is_special=False, cls=cls)

        image_data, box_data = get_random_data_with_SOA(line_1, image_small, small_object_list, annots)

        if image_data is not None:
            print("line_1", line_1.split()[0])
            print("line_2", line_2.split()[0])
            img = Image.fromarray(image_data.astype(np.uint8))
            img.save(os.path.join(Out_JPEGImages_path, str(Cur_Num) + '.jpg'))
            write_xml(os.path.join(Out_Annotations_path, str(Cur_Num) + '.xml'), os.path.join(Out_JPEGImages_path, str(Cur_Num) + '.jpg'), \
                        headstr, input_shape, box_data, unique_labels, tailstr)
            Cur_Num += 1