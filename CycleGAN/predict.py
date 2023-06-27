#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from cyclegan import CYCLEGAN
import xml.etree.ElementTree as ET
import os
import xml.dom.minidom
from tqdm import tqdm

def cycleGANAugmentaion(image_path, xml_path, image_save_path, xml_save_path, mode="dir_predict"):
    print("-----------------------------")
    print("cycleGAN增强开始")
    cyclegan = CYCLEGAN()
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = cyclegan.detect_image(image)

                r_image.show()

    elif mode == "dir_predict":
        img_names = os.listdir(image_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                pic_path = os.path.join(image_path, img_name)
                # print(pic_path)
                image = Image.open(pic_path)
                # print(image)
                img_h, img_w = image.size[0], image.size[1]
                r_image = cyclegan.detect_image(image)
                # ——————————————保存xml文件---------------------------#
                xmlfile1 = xml_path + r'/' + img_name[:-4] + '.xml'
                tree1 = ET.parse(xmlfile1)
                doc = xml.dom.minidom.Document()
                root = doc.createElement("annotation")
                doc.appendChild(root)

                for folds in tree1.findall("folder"):
                    folder = doc.createElement("folder")
                    folder.appendChild(doc.createTextNode(str(folds.text)))
                    root.appendChild(folder)
                for filenames in tree1.findall("filename"):
                    filename = doc.createElement("filename")
                    filename.appendChild(doc.createTextNode(str(filenames.text[:-4])+'_cyclegan.jpg'))
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
                nodeframe.appendChild(doc.createTextNode(img_name[:-4] + '_3'))

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

                fp = open(xml_save_path + '/' + img_name[:-4] + "_cyclegan.xml", "w")
                doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
                fp.close()

                if not os.path.exists(image_save_path):
                    os.makedirs(image_save_path)
                img_name = img_name[:-4] + "_cyclegan" + img_name[-4:]
                r_image.save(os.path.join(image_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'.")
    print("cycleGAN增强结束")
    print("-----------------------------")


"""if __name__ == "__main__":
    
"""