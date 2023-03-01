# coding:utf-8

import cv2
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom

img_path ="D:\yolo v7 data processing\mixupkuocong\gui tiao zai picture"          # 原始图片文件夹路径
save_path = "D:\yolo v7 data processing\mixupkuocong\mix_image"     # mixup的图片文件夹路径
xml_path = "D:\yolo v7 data processing\mixupkuocong\guitiaozai_xml"           # 原始图片对应的标注文件xml文件夹的路径
save_xml = "D:\yolo v7 data processing\mixupkuocong\mix_lmx"    # mixup的图片对应的标注文件xml的文件夹路径
img_names = os.listdir(img_path)
img_num = len(img_names)
print('img_num:', img_num)
def pepper_and_salt(img, percentage):
    num=int(percentage*img.shape[0]*img.shape[1])  # 椒盐噪声点数量
    random.randint(0, img.shape[0])
    img2 = img.copy()
    for i in range(num):
        X=random.randint(0, img2.shape[0]-1)  # 从0到图像长度之间的一个随机整数,因为是闭区间所以-1
        Y=random.randint(0, img2.shape[1]-1)
        if random.randint(0, 1) == 0:  # 黑白色概率55开
            img2[X,Y] = (255,255, 255)  # 白色
        else:
            img2[X, Y] = (0, 0, 0)  # 黑色
    return img2
for imgname in img_names:
    imgpath = os.path.join(img_path,imgname)
    img = cv2.imread(imgpath)
    # img =pepper_and_salt(img,0.04)
    img_h, img_w = img.shape[0], img.shape[1]
    print(img_h,img_w)

    i = random.randint(0, img_num - 1)
    print('i:', i)
    add_path = os.path.join(img_path,img_names[i])
    addimg = cv2.imread(add_path)

    add_h, add_w = addimg.shape[0], addimg.shape[1]
    if add_h != img_h or add_w != img_w:
        print('resize!')
        addimg = cv2.resize(addimg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    scale_h, scale_w = img_h / add_h, img_w / add_w

    lam = np.random.beta(1.5, 1.5)
    print(lam)
    print(imgname)
    mixed_img = lam * img + (1 - lam) * addimg
    save_img = save_path + '/'+imgname[:-4] + '.jpg'
    cv2.imwrite(save_img, mixed_img)
    print(save_img)

    print(imgname, img_names[i])
    if imgname != img_names[i]:
        xmlfile1 = xml_path + r'/'+imgname[:-4] + '.xml'
        xmlfile2 = xml_path + r'/'+ img_names[i][:-4] + '.xml'
        print(xmlfile1,xmlfile2)

        tree1 = ET.parse(xmlfile1)
        tree2 = ET.parse(xmlfile2)

        doc = xml.dom.minidom.Document()
        root = doc.createElement("annotation")
        doc.appendChild(root)


        for folds in tree1.findall("folder"):
            folder = doc.createElement("folder")
            folder.appendChild(doc.createTextNode(str(folds.text)))
            root.appendChild(folder)
        for filenames in tree1.findall("filename"):
            filename = doc.createElement("filename")
            filename.appendChild(doc.createTextNode(str(filenames.text)))
            root.appendChild(filename)
        for paths in tree1.findall("path"):
            path = doc.createElement("path")
            path.appendChild(doc.createTextNode(str(paths.text)))
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
        nodeframe.appendChild(doc.createTextNode(imgname[:-4] + '_3'))

        objects = []

        for obj in tree1.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                  int(bbox.find("ymin").text),
                                  int(bbox.find("xmax").text),
                                  int(bbox.find("ymax").text)]
            objects.append(obj_struct)

        for obj in tree2.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text          # 有的版本的labelImg改参数为小写difficult
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(int(bbox.find("xmin").text) * scale_w),
                                  int(int(bbox.find("ymin").text) * scale_h),
                                  int(int(bbox.find("xmax").text) * scale_w),
                                  int(int(bbox.find("ymax").text) * scale_h)]
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
            nodepose.appendChild(doc.createTextNode(obj["truncated"]))
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

        fp = open(save_xml +'/'+ imgname[:-4] + ".xml", "w")
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()

    else:
        xmlfile1 = xml_path + '/'+imgname[:-4] + '.xml'
        print(xmlfile1)
        tree1 = ET.parse(xmlfile1)

        doc = xml.dom.minidom.Document()
        root = doc.createElement("annotation")


        doc.appendChild(root)

        for folds in tree1.findall("folder"):
            folder=doc.createElement("folder")
            folder.appendChild(doc.createTextNode(str(folds.text)))
            root.appendChild(folder)
        for filenames in tree1.findall("filename"):
            filename=doc.createElement("filename")
            filename.appendChild(doc.createTextNode(str(filenames.text)))
            root.appendChild(filename)
        for paths in tree1.findall("path"):
            path = doc.createElement("path")
            path.appendChild(doc.createTextNode(str(paths.text)))
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
        nodeframe.appendChild(doc.createTextNode(imgname[:-4] + '_3'))
        objects = []

        for obj in tree1.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = obj.find("truncated").text
            obj_struct["difficult"] = obj.find("difficult").text
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                  int(bbox.find("ymin").text),
                                  int(bbox.find("xmax").text),
                                  int(bbox.find("ymax").text)]
            objects.append(obj_struct)

        for obj in objects:
            nodeobject = doc.createElement("object")
            nodename = doc.createElement("name")
            nodepose = doc.createElement("pose")
            nodetruncated = doc.createElement("truncated")
            nodeDifficult = doc.createElement("difficult")
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

        fp = open(save_xml +'/'+ imgname[:-4] + "_3.xml", "w")
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
