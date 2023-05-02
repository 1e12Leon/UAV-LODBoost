import os
from tqdm import tqdm
import cv2
import numpy as np
import random

from Object_Detection_Augmentation.utils import image_resize, plot_bboxes,read_xml,add_xml


def cutmix(image, boxes, r_image, r_boxes):
    """

        Parameter:
            image (ndarray): First input image and target image for the cutout application
            boxes (ndarray): numpy array of bounding boxes beloning to first input image
            r_image (ndarray): Second input image that will be used as cutout area
            r_boxes (ndarray): numpy array of bounding boxes belonging to first input image

        Returns:
            A new image consisting of merged input images and merged input
    """



    # 复制图1
    target_image = image.copy()

    # 获得2个图片的总体尺寸
    img_size = image.size
    r_img_size = r_image.size

    # 获得图片的长高
    img_h, img_w = image.shape[:2]
    r_img_h, r_img_w = r_image.shape[:2]


    # 根据目标图片判断第二个图片是否需要放大或者缩小
    if image.shape is not r_image.shape or img_size is not r_img_size:#如果图像大小不一样
        r_image, ratio = image_resize(r_image, img_w, img_h)#根据图1大小重新塑造图二的大小
        r_img_size=r_image.size

        if img_size >= r_img_size:#如果图1的尺寸大于图二的尺寸(重塑后宽度w相同)
            r_img_h, r_img_w = r_image.shape[:2]
            imsize = min(r_img_h, r_img_w)
        else:
            imsize = min(img_w, img_h)

        # 将标注框重塑
        r_boxes[:, 1:] = r_boxes[:, 1:] * ratio


    else:
        imsize = min(img_w, img_h)

    # 创造要cutmix的部分

    x1, y1 = [int(random.uniform(imsize * 0.1, imsize * 0.45)) for _ in range(2)]
    x2, y2 = [int(random.uniform(imsize * 0.55, imsize * 1)) for _ in range(2)]
    dropout_box = [x1, y1, x2, y2]


    #删除不在随机区图二的标注框根据iou
    index = [i for i, box in enumerate(r_boxes) if bbox_ioa(box[1:], dropout_box) <= 0.0001]
    r_boxes = np.delete(r_boxes, index, axis=0)#删除矩阵参数（矩阵，所要删除的行或列，axis=0行删除=1列删除）


    mixup_boxes = r_boxes.copy()



    # 确定哪些框落在 dropout 区域之外，如果存在重叠，则将它们裁剪到裁剪外部极限
    # 如果边界框与裁剪区域完全重叠，则将其删除

    # 将改的图像边界框裁剪到生成的遮罩区域内部
    mixup_boxes[:, [1, 3]] = mixup_boxes[:, [1, 3]].clip(min=x1, max=x2)
    mixup_boxes[:, [2, 4]] = mixup_boxes[:, [2, 4]].clip(min=y1, max=y2)


    # Translate normalized boxes to absolute coords
    # boxes = get_absolute_coords(boxes, img_w, img_h)

    a=[]
    # 对于目标图像中的所有框，检查它们是否与dropout区域重叠
    #我们需要调整框，以便它们将剪辑到dropout区域的外部边界
    for i, box in enumerate(boxes[:, 1:]):
        # 首先检查边界框是否甚至在 dropout 区域内，如果没有跳过整个迭代
        iou = bbox_ioa(box, dropout_box)
        if iou > 0.0001:
            # 检查框是否完全重叠。如果是这样，请将其从框数组中删除
            if is_box_inside(box, dropout_box):
                a.append(i)
            else:
                box[:] = clip_outer_to_inner(box, dropout_box)[:]

    boxes = np.delete(boxes, a, axis=0)

    # 将所选r_image区域中的矩形区域替换为目标图像中的相同区域

    target_image[y1:y2, x1:x2] = r_image[y1:y2, x1:x2]


    # 将所有边界框合并为一个数组
    mixup_boxes = np.concatenate((boxes, mixup_boxes), axis=0)

    return target_image, mixup_boxes


def bbox_ioa(box1, box2):
    '''

    :param box1:x1y1x2y2
    :param box2:x1y1x2y2
    :return:IUO交并比
    '''
    # 返回给定 box1、box2 的 box2 区域上的交集。box1 是 4，box2 是 nx4。盒子是 x1y1x2y2

    # 获取边界框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # 交叉区域面积
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) *(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)



    # box2区域面积
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # 返回交并比
    return inter_area / box2_area


def is_box_inside(ibox, bbox):
    """
    检查内框是否位于基于 x1、y1、x2、y2 坐标的边界框坐标内。
    :param ibox: 内盒检查它是否在假定的边界框内
    :param bbox: (array) 边界框
    :return: (boolean)
    """

    if bbox[0] <= ibox[0] and bbox[1] <= ibox[1]:
        if bbox[2] >= ibox[2] and bbox[3] >= ibox[3]:
            return True
    else:
        return False


def get_box_area(box):
    return (box[3] - box[1]) * (box[2] - box[0])


def clip_outer_to_inner(outer_box, inner_box):
    """
    将剪裁外框的坐标，与内框重叠
    :param outer_box:
    :param inner_box:
    :return:
    """
    # 如果 dropout 区域没有完全重叠，请确定重叠是否发生在

    bx1, by1, bx2, by2 = outer_box[0], outer_box[1], outer_box[2], outer_box[3]
    dx1, dy1, dx2, dy2 = inner_box[0], inner_box[1], inner_box[2], inner_box[3]

    candidate_boxes = []

    # 检查左边距的重叠
    if bx1 < dx1 < bx2:
        candidate_box = [bx1, by1, dx1, by2]
        candidate_boxes.append(candidate_box)
        # box[2] = dx1

    # 检查上边距的重叠
    if by1 < dy1 < by2:
        candidate_box = [bx1, by1, bx2, dy1]
        candidate_boxes.append(candidate_box)
        # box[3] = dy1

    # 检查右边的重叠
    if bx1 < dx2 < bx2:
        candidate_box = [dx2, by1, bx2, by2]
        candidate_boxes.append(candidate_box)
        # box[0] = dx2

    # 检查下限的重叠
    if by1 < dy2 < by2:
        candidate_box = [bx1, dy2, bx2, by2]
        candidate_boxes.append(candidate_box)
        # box[1] = dy2

    if len(candidate_boxes) == 1:
        new_box = np.array(candidate_boxes[0])
        outer_box[:] = new_box[:]

    elif len(candidate_boxes) > 1:
        max_idx = 0
        max_area = 0

        for i, candidate in enumerate(candidate_boxes):
            box_area = get_box_area(candidate)
            if box_area > max_area:
                max_area = box_area
                max_idx = i
        outer_box[:] = np.array(candidate_boxes[max_idx])[:]

    return outer_box



def CutmixAugmentation(image_path,xml_path,image_save_path,xml_save_path):
    print('------------------------------')
    print('cutmix增强')

    folder = os.path.basename(image_save_path)

    image_names = os.listdir(image_path)
    img_num=len(image_names)
    for image_name in tqdm(image_names):
        i = random.randint(0, img_num - 1)
        #--------------导入图片与导入xml------------
        "=======xml为[1，X1,X2,Y1,Y2]========"
        img1 = cv2.imread(os.path.join(image_path,image_name))
        img2 = cv2.imread(os.path.join(image_path, image_names[i]))
        l1 = read_xml(xml_path,image_name)
        l2 = read_xml(xml_path,image_names[i])


        # -----------进行cutmix操作---------------
        mix_img, labels = cutmix(img2, l2, img1, l1)
        img_w,img_h=mix_img.shape[1],mix_img.shape[0]
        add_xml(xml_path,image_name,xml_save_path,labels,folder,image_save_path,img_w,img_h)
        final_img = plot_bboxes(mix_img, labels)
        # cv2.imshow("output", final_img)  # Show image
        cv2.imwrite(os.path.join(image_save_path,'cutmix_'+image_name),final_img)
