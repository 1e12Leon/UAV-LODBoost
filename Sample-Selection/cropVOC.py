from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np

ImgPath = r'D:\Deep_Learning_folds\VideoProcess\VOC2007gtz\JPEGImages/'
AnnoPath = r'D:\Deep_Learning_folds\VideoProcess\VOC2007gtz\Annotations/'
ProcessedPath = r'D:\Deep_Learning_folds\VideoProcess\VOC2007gtz\cropImages/'

imagelist = os.listdir(ImgPath)

for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + image
    print(imgfile)
    if not os.path.exists(AnnoPath + image_pre + '.xml'):
        continue
    xmlfile = AnnoPath + image_pre + '.xml'
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    i = 1
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        savepath = ProcessedPath + objectname
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        bndbox = objects.getElementsByTagName('bndbox')
        cropboxes = []
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            w = x2 - x1
            h = y2 - y1
            obj = np.array([x1, y1, x2, y2])
            shift = np.array([[1, 1, 1, 1]])
            XYmatrix = np.tile(obj, (1, 1))
            cropboxes = XYmatrix * shift
            img = Image.open(imgfile)
            for cropbox in cropboxes:
                cropedimg = img.crop(cropbox)
                cropedimg.save(savepath + '/' + image_pre + '.jpg')
                # cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.jpg')
                i += 1