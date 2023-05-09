import os
import random
import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """

    :param image: 输入需要变换的图片
    :param width: 对比图片的宽
    :param height: 对比图片的高
    :param inter:
    :return: 返回变换后的输入图片和缩放比例
    """
    (h, w) = image.shape[:2]#需要缩放的图片尺寸

    # 如果对比图片大小为0返回
    if width is None and height is None:
        return image

    # 检查宽度是否为“无”，如果计算高度
    if width is None:
        #计算高度的比率并构造维度
        r = height / float(h)
        dim = (int(w * r), height)

    # 检查高度是否为“无”
    else:
        # 计算宽度的比率并构造维度
        r = width / float(w)
        dim = (width, int(h * r))

    # 重新塑造图片尺寸
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回重塑图片的尺寸与比率
    return resized, r


def plot_bboxes(img, targets):
    names = ['gtz', 'others', 'group','connection']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    for i, det in enumerate(targets):  # detections per image
        cls = int(det[0])
        xyxy = det[1:]  # x1, y1 are center coordinates for boudning box
        label = names[cls-1]
        plot_one_box(xyxy, img, label=label, color=colors[cls-1], line_thickness=3)

    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    w, h, _ = img.shape

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    bbox_w = int(x[2] * h)
    bbox_h = int(x[3] * w)

    x1, y1 = int(x[0] * h), int(x[1] * w)

    c1 = int(x1 - (bbox_w / 2)), int(y1 - (bbox_h / 2))
    c2 = c1[0] + bbox_w, c1[1] + bbox_h

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def read_xml(xml_path,image_name):
    xmlfile1 = xml_path + r'/' + image_name[:-4] + '.xml'
    tree1 = ET.parse(xmlfile1)
    doc = xml.dom.minidom.Document()
    root = doc.createElement("annotation")
    doc.appendChild(root)

    objects = []

    for obj in tree1.findall("object"):
        bbox = obj.find("bndbox")

        object =             [int(bbox.find("xmin").text),
                              int(bbox.find("ymin").text),
                              int(bbox.find("xmax").text),
                              int(bbox.find("ymax").text)]
        if obj.find("name").text=='gtz':
            object.insert(0,1)
        elif obj.find("name").text=='others':
            object.insert(0,2)
        elif obj.find("name").text=='group':
            object.insert(0,3)
        elif obj.find("name").text=='connection':
            object.insert(0,4)


        objects.append(object)


    objects=np.array(objects)

    return objects
def load_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
    return labels


def get_absolute_coords(boxes, img_w, img_h):
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_w
    boxes[:, [2, 4]] = boxes[:, [2, 4]] * img_h
    return boxes


def get_relative_coords(boxes, img_w, img_h):
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / img_w
    boxes[:, [2, 4]] = boxes[:, [2, 4]] / img_h
    return boxes


def add_xml(xml_path,image_name,xml_save_path,add_object_xml,folders,image_save_name,img_w,img_h):
    xmlfile1 = xml_path + r'/' + image_name[:-4] + '.xml'
    tree1 = ET.parse(xmlfile1)
    doc = xml.dom.minidom.Document()
    root = doc.createElement("annotation")
    doc.appendChild(root)

    for folds in tree1.findall("folder"):
        folder = doc.createElement("folder")
        folder.appendChild(doc.createTextNode(folders))
        root.appendChild(folder)
    for filenames in tree1.findall("filename"):
        filename = doc.createElement("filename")
        filename.appendChild(doc.createTextNode('cutmix_'+image_name))
        root.appendChild(filename)
    for paths in tree1.findall("path"):
        path = doc.createElement("path")
        path.appendChild(doc.createTextNode(os.path.join(image_save_name,'cutmix_'+image_name)))
        root.appendChild(path)
    for sources in tree1.findall("source"):
        source = doc.createElement("source")
        database = doc.createElement("database")
        database.appendChild(doc.createTextNode(str("Unknow")))
        source.appendChild(database)
        root.appendChild(source)
    for sizes in tree1.findall("size"):
        size = doc.createElement("size")
        width = doc.createElement("width")
        height = doc.createElement("height")
        depth = doc.createElement("depth")
        width.appendChild(doc.createTextNode(str(img_w)))
        height.appendChild(doc.createTextNode(str(img_h)))
        depth.appendChild(doc.createTextNode(str(3)))
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        root.appendChild(size)


    nodeframe = doc.createElement("frame")
    nodeframe.appendChild(doc.createTextNode(image_name[:-4] + '_3'))

    objects = []

    for obj in add_object_xml:
        obj_struct = {}
        obj_struct["name"] = "gtz"
        obj_struct["pose"] = 'Unspecified'
        obj_struct["truncated"] = "0"
        obj_struct["difficult"] ="0"
        obj_struct["bbox"] = [obj[1],
                              obj[2],
                              obj[3],
                              obj[4]]
        objects.append(obj_struct)

    for obj in objects:
        nodeobject = doc.createElement("object")
        nodename = doc.createElement("name")
        nodepose = doc.createElement("pose")
        nodetruncated = doc.createElement("truncated")
        nodeDifficult = doc.createElement("Difficult")
        nodebndbox = doc.createElement("bndbox")
        nodexmin = doc.createElement("xmin")
        nodeymin = doc.createElement("ymin")
        nodexmax = doc.createElement("xmax")
        nodeymax = doc.createElement("ymax")
        nodename.appendChild(doc.createTextNode(obj["name"]))
        nodepose.appendChild(doc.createTextNode(obj["pose"]))
        nodetruncated.appendChild(doc.createTextNode(obj["truncated"]))
        nodeDifficult.appendChild(doc.createTextNode(obj["difficult"]))
        nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
        nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
        nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
        nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

        nodebndbox.appendChild(nodexmin)
        nodebndbox.appendChild(nodeymin)
        nodebndbox.appendChild(nodexmax)
        nodebndbox.appendChild(nodeymax)

        nodeobject.appendChild(nodename)
        nodeobject.appendChild(nodepose)
        nodeobject.appendChild(nodetruncated)
        nodeobject.appendChild(nodeDifficult)
        nodeobject.appendChild(nodebndbox)

        root.appendChild(nodeobject)

    fp = open(xml_save_path + '/' + "cutmix_" + image_name[:-4] + ".xml", "w")
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()
