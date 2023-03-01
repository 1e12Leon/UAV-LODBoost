
from xml.dom import minidom
import json
import glob
import os
from shutil import copyfile
# 将self.orderDict中的信息写入本地xml文件，参数filename是xml文件名

def Convert_coco_to_voc(annotation_path,JPEGImage_path,out_path):
    '''
    将coco数据转换为voc数据
    :param annotation_path:coco标记文件的地址
    :param JPEGImage_path: coco图片数据集的位置
    :param out_path: 生成的voc数据集保存的位置
    :return:
    '''
    #先创建voc输出文件夹标准格式
    outdir = out_path
    # 创建各级文件夹
    train_xml_out = os.path.join(outdir, 'VOC2007/Annotations')
    if not os.path.exists(train_xml_out ):
        os.makedirs(train_xml_out)
    train_img_out = os.path.join(outdir, 'VOC2007/JPEGImages')
    if not os.path.exists(train_img_out):
        os.makedirs(train_img_out)


    data = json.load(open(annotation_path, 'r'))
    # imgs = data['images']
    # imgs = []
    annotations = data['annotations']
    print(annotations[0])
    categorie=data["categories"]
    print(categorie)
    categorie = ["pedestrian", "car", "truck", "van", "motor", "bicycle", "bus", "trailer"]
    # cate_list=list()
    # print('JSON文件中_图片总数:' + str(len(imgs)))
    """for i  in range(len(categorie)):
        for ca in categorie:
            if ca["id"]==i:
                cate_list.append(ca["supercategory"])
    print("JSON文件中_标注类别:" + str(cate_list))"""

    #遍历图片json列表
    for img in annotations:
        filename = img['image_name']
        print(filename)
        img_w = img['image_width:']
        img_h = img['image_height']

        # img_id = img['id']
        # ana_txt_name = filename.split('.')[0] + '.txt'
        roi_list=list()
        for box in img['bbox']:
            b = [box['left'], box['top'], box['width'], box['height']]
            cls_index = box['class']
            box = convert_boxshape((img_w, img_h), b)

            roi_list.append([categorie[cls_index],box[0], box[1], box[2], box[3]])
        print(box)
        get_xml(filename, [img_w,img_h,3], roi_list,train_xml_out)
        copyfile(os.path.join(JPEGImage_path, filename),os.path.join(train_img_out, filename,))
def convert_boxshape(size, box):
    '''
    coco数据集标注是xmin,ymin,width,height 而voc和yolo是xmin ymin xmax ynax 需要进行转换
    :param size: 图片宽 图片高
    :param box:
    :return:
    '''
    dw = size[0]   # 图像实际宽度
    dh = size[1]  # 图像实际高度
    #x=box[0]
    #y=box[1]
    x = box[0]  #标注区域X坐标
    y = box[1]  #标注区域Y坐标
    w = box[2]   # 标注区域宽度
    h = box[3]   #标注区域高度

    xmin=int(x)
    ymin=int(y)
    xmax=int(x+w)
    ymax=int(y+h)
    return (xmin, ymin, xmax, ymax)


def get_xml(img_name,size,roi,outpath):
    '''
    传入每张图片的名称 大小 和标记点列表生成和图片名相同的xml标记文件
    :param img_name: 图片名称
    :param size:[width,height,depth]
    :param roi:[["label1",xmin,ymin,xmax,ymax]["label2",xmin,ymin,xmax,ymax].....]
    :return:
    '''
    impl = minidom.getDOMImplementation()
    doc = impl.createDocument(None, None, None)
    #创建根节点
    orderlist = doc.createElement("annotation")
    doc.appendChild(orderlist)
    #c创建二级节点
    filename=doc.createElement("filename")
    filename.appendChild(doc.createTextNode(img_name))
    orderlist.appendChild(filename)
    #size节点
    sizes=doc.createElement("size")
    width=doc.createElement("width")
    width.appendChild(doc.createTextNode(str(size[0])))
    height=doc.createElement("height")
    height.appendChild(doc.createTextNode(str(size[1])))
    depth=doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(size[2])))
    sizes.appendChild(width)
    sizes.appendChild(height)
    sizes.appendChild(depth)
    orderlist.appendChild(sizes)
    #object 节点
    for ri in roi:
        object=doc.createElement("object")
        name=doc.createElement("name")
        name.appendChild(doc.createTextNode(ri[0]))
        object.appendChild(name)
        bndbox=doc.createElement("bndbox")

        xmin=doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(ri[1])))
        bndbox.appendChild(xmin)
        ymin=doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(ri[2])))
        bndbox.appendChild(ymin)
        xmax=doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(ri[3])))
        bndbox.appendChild(xmax)
        ymax=doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(ri[4])))
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        orderlist.appendChild(object)
         # 将dom对象写入本地xml文件

    # 打开test.xml文件 准备写入
    f = open(os.path.join(outpath,img_name[:-4]+'.xml'), 'w')
    # 写入文件
    doc.writexml(f, addindent='  ', newl='\n')
    # 关闭
    f.close()

if __name__ == "__main__":
    #传入coco的annotations.json地址，image文件存放的地址 和输出的voc数据存放地址
    Convert_coco_to_voc("VOCdevkit/AU-AIR/annotations.json","/home/zychen/chenning/AU-AIR/auair2019data/images","VOCdevkit/AU-AIR",)
