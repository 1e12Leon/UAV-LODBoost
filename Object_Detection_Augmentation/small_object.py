import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xml.dom.minidom


class SmallObjectAugmentation(object):
    def __init__(self, thresh=64*64, prob=None, copy_times=None, epochs=None, all_objects=False, one_object=False,img_1=None,annot_1=None,small_object_list=None,small_bbox=None,small_object=None):
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        prob: the prob to do small object augmentation
        epochs: the epochs to do
        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        self.img_1=  img_1
        self.annot_1=annot_1
        self.small_bbox=small_bbox
        self.small_object=small_object
        self.small_object_list=small_object_list
        if self.all_objects or self.one_object:
            self.copy_times = 1


    def compute_overlap(self, annot_a, annot_b):
        if annot_a is None: return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, new_annot, annots):
        for annot in annots:
            if self.compute_overlap(new_annot, annot): return False
        return True

    def create_copy_annot(self, h, w, annot, annots):
        '''

        :param h: 图片高度
        :param w: 图片宽度
        :param annot: 小目标【xmin,ymin,xmax,ymax】
        :param annots:
        :return:
        '''
        annot = np.array(annot)
        annot = annot.astype(np.int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        for epoch in range(self.epochs):
            try:
                random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                     np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            except Exception as e:
                pass
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h

            if xmin <= 0 or xmax > w or ymin <= 0 or ymax > h:
                continue
            if xmin >= xmax  or ymin >= ymax:
                continue
            # print(xmin, ymin, xmax, ymax, annot[4])
            # print(annot)
            new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(np.int)

            if self.donot_overlap(new_annot, annots) is False:
                continue

            return new_annot
        return None

    def add_patch_in_img(self,copy_annot,annot, img,image):
        """

        :param annot: 要添加的新的框
        :param copy_annot: 复制的框
        :param image: 图片1
        :param img: 图片2（背景）
        :return:
        """
        copy_annot = copy_annot.astype(np.int)
        try:
            image[ copy_annot[1]: copy_annot[3],  copy_annot[0]: copy_annot[2], :] = img
        except Exception as e:
            pass
        return image

    def __call__(self, sample):
        if self.all_objects and self.one_object: return sample
        if np.random.rand() > self.prob: return sample

        img_1, annots_1 = sample['img_1'], sample['annot_1']#定义图像与框
        h, w= img_1.shape[0], img_1.shape[1]#获得图像大小
        annots_1 = annots_1.tolist()

        for i in range(self.copy_times):
            new_annot = self.create_copy_annot(h, w, self.small_bbox, annots_1, )  # 返回的是复制粘贴的放

            if new_annot is not None:
                self.img_1 = self.add_patch_in_img(copy_annot=new_annot, annot=self.small_bbox, img=self.small_object, image=self.img_1)
                annots_1.append(new_annot)

        return {'img_1': self.img_1, 'annot_1': np.array(annots_1)}


def issmallobject(name,lower_thresh, hight_thresh, h, w,objects):
    for i in range(0, len(objects)):
        if lower_thresh<= h * w <= hight_thresh and name==objects[i]:
            return True
    return False


def isnosmallobject( h, w):
    if 64*64<= h * w :
        return True
    else :
        return False


def read_xml(xml_path,image_name,rate_h,rate_w,category):
    xmlfile1 = xml_path + r'/' + image_name[:-4] + '.xml'
    tree1 = ET.parse(xmlfile1)
    doc = xml.dom.minidom.Document()
    root = doc.createElement("annotation")
    doc.appendChild(root)

    objects = []

    for obj in tree1.findall("object"):
        bbox = obj.find("bndbox")
        object =             [int(float(bbox.find("xmin").text)*rate_w),
                              int(float(bbox.find("ymin").text)*rate_h),
                              int(float(bbox.find("xmax").text)*rate_w),
                              int(float(bbox.find("ymax").text)*rate_h)]
        for i in range(0,len(category)):
            if obj.find("name").text==category[i]:
                object.append(i)
        objects.append(object)

    objects=np.array(objects)

    return objects


def add_xml(xml_path,image_name,xml_save_path,add_object_xml,SMB_image_name,image_save_path,img_h, img_w,category):
    xmlfile1 = xml_path + r'/' + image_name[:-4] + '.xml'
    tree1 = ET.parse(xmlfile1)
    doc = xml.dom.minidom.Document()
    root = doc.createElement("annotation")
    doc.appendChild(root)

    for folds in tree1.findall("folder"):
        folder = doc.createElement("folder")
        folder.appendChild(doc.createTextNode(os.path.split(image_save_path)[-1]))
        root.appendChild(folder)
    for filenames in tree1.findall("filename"):
        filename = doc.createElement("filename")
        filename.appendChild(doc.createTextNode(SMB_image_name))
        root.appendChild(filename)
    for paths in tree1.findall("path"):
        path = doc.createElement("path")
        path.appendChild(doc.createTextNode(os.path.join(image_save_path, SMB_image_name)))
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
        obj_struct["name"] =category[int(obj[4])]
        obj_struct["pose"] = 'Unspecified'
        obj_struct["truncated"] = "0"
        obj_struct["difficult"] ="0"
        obj_struct["bbox"] = [obj[0],
                              obj[1],
                              obj[2],
                              obj[3]]
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

    fp = open(xml_save_path + '/' + "smb_" + image_name[:-4] + ".xml", "w")
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()


def collect_small_object(xml_path,SOA_THRESH,Low_SOA_THRESH,category,objects):
    """
    目的：提取图片中所有小目标
    :param xml_path: xml地址
    :param rate_w: 缩放比例w
    :param rate_h: 缩放比例h
    :return: 返回包含小目标的列表【xmin，ymin，xmax，ymax，1，文件名称】
    """
    xmls = os.listdir(xml_path)
    small_objects = []

    for xml_name in xmls:
        tree1 = ET.parse(os.path.join(xml_path,xml_name))
        doc = xml.dom.minidom.Document()
        root = doc.createElement("annotation")
        doc.appendChild(root)

        for obj in tree1.findall("object"):
            bbox = obj.find("bndbox")
            bbox_h=int(float(bbox.find("ymax").text) )-int(float(bbox.find("ymin").text) )
            bbox_w=int(float(bbox.find("xmax").text) )-int(float(bbox.find("xmin").text))
            obj_name=obj.find("name").text
            if issmallobject(name=obj_name ,lower_thresh=Low_SOA_THRESH, hight_thresh=SOA_THRESH, h=bbox_h,w=bbox_w,objects=objects):
                    small_object = [int(float(bbox.find("xmin").text)),
                              int(float(bbox.find("ymin").text)),
                              int(float(bbox.find("xmax").text) ),
                              int(float(bbox.find("ymax").text) )]
                    for i in range(0,len(category)):
                        if obj.find("name").text == category[i]:
                                small_object.append(i)
                    small_object.append(xml_name[:-4])
                    small_objects.append(small_object)

    return  small_objects


def Samll_object_Augmentation(image_path,xml_path,image_save_path,xml_save_path,SOA_THRESH,
                              SOA_PROB, SOA_COPY_TIMES, SOA_EPOCHS,Low_SOA_THRESH,category,objects):
    image_names = os.listdir(image_path)
    print('----------------------------')
    print('小目标数据增强')

    # c_w, c_h = 1294,458
    c_w, c_h = 1920,1080
    small_object_list = collect_small_object(xml_path, SOA_THRESH ,Low_SOA_THRESH,category,objects)
    print(len(small_object_list))
    for image_name in tqdm(image_names):
        #图片1背景
        img_1 = cv2.imread(os.path.join(image_path,image_name))
        h_1, w_1 = img_1.shape[0],img_1.shape[1]
        rate_h_1 = c_h/h_1
        rate_w_1 = c_w/w_1
        img_1=cv2.resize(img_1,(c_w,c_h))

        SMB_image_name = 'smb_' +image_name
        annot_1 = read_xml(xml_path, image_name,rate_h_1,rate_w_1,category)  # 读取所有的xml并且从新按比例划分

        #图片2提取小目标
        i = random.randint(0, len(small_object_list) - 1)
        img_2= cv2.imread(os.path.join(image_path,small_object_list[i][5]+'.jpg'))

        h_2, w_2= img_2.shape[0],img_2.shape[1]

        xml_w=int(small_object_list[i][2])-int(small_object_list[i][0])
        xml_h=int(small_object_list[i][3])-int(small_object_list[i][1])
        if isnosmallobject(xml_h,xml_w):
            j=random.randint(64, 128)
            rate_w_3=j/xml_w
            rate_h_3 = j / xml_h
            small_bbox = [int(small_object_list[i][0] * rate_w_3), int(small_object_list[i][1] * rate_h_3),
                          int(small_object_list[i][2] * rate_w_3), int(small_object_list[i][3] * rate_h_3),
                          small_object_list[i][4]]
            img_2 = cv2.resize(img_2, (int(rate_w_3*h_2), int(rate_h_3*w_2)))
            small_object = img_2[small_bbox[1]:small_bbox[3], small_bbox[0]:small_bbox[2], :]
        else:
            rate_h_2 = c_h / h_2
            rate_w_2 = c_w / w_2
            small_bbox = [int(small_object_list[i][0] * rate_w_2), int(small_object_list[i][1] * rate_h_2),
                          int(small_object_list[i][2] * rate_w_2), int(small_object_list[i][3] * rate_h_2),
                          small_object_list[i][4]]
            img_2 = cv2.resize(img_2, (c_w, c_h))
            small_object = img_2[small_bbox[1]:small_bbox[3], small_bbox[0]:small_bbox[2], :]


        augmenter = SmallObjectAugmentation(SOA_THRESH, SOA_PROB, SOA_COPY_TIMES,
                                            SOA_EPOCHS,img_1=img_1,annot_1=annot_1,small_object_list=small_object_list,
                                            small_bbox=small_bbox,small_object=small_object)

        Sample={"img_1":img_1,"annot_1":annot_1}
        new_Sample= augmenter.__call__(Sample)


        add_object_xml=new_Sample['annot_1']
        img_h, img_w = new_Sample['img_1'].shape[0], new_Sample['img_1'].shape[1]

        add_xml(xml_path,image_name,xml_save_path,add_object_xml,SMB_image_name,image_save_path,img_h, img_w,category)


        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        cv2.imwrite(os.path.join(image_save_path,SMB_image_name ), new_Sample['img_1'])


