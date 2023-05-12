# UAV-LODBoost:A User-Friendly Toolkit for UAV Light-weighting Object Detection

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

这个仓库将包含但不限于以下功能：

1. 一套**目标检测数据增强与扩充**代码。
2. 一种**样本筛选**方法。
3. 一套**目标检测与跟踪**模型。
4. 一套**模型压缩**框架。
5. 一个**模型部署**示例。

👉[🎈使用文档](https://hhu-leons-organization.gitbook.io/uav_lodboost_docs/)
👈

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [使用说明](#使用说明)
	- [数据处理](#数据处理)
    - [YOLO训练步骤](#YOLO训练步骤)
    - [YOLO预测步骤](#YOLO预测步骤)
    - [YOLO评估步骤](#YOLO评估步骤)
- [示例](#示例)
- [相关仓库](#相关仓库)
- [维护者](#维护者)
- [如何贡献](#如何贡献)

## 背景

这个仓库是一个通用的工程级别目标检测框架包含：
1. 一套**目标检测数据增强与扩充**代码。它将是一个一直在持续维护的代码库，将不断添加有效的特殊数据增强方法。
2. 一种**样本筛选**方法。它将通过深度特征来有效聚类，用于剔除数据增强后的无效样本。
3. 一套**目标检测与跟踪**模型。它使用较为热门的yolov7以及ByteTrack，能够胜任工程中常见的检测与跟踪任务。
4. 一套**模型压缩**框架。它将有效集成剪枝、蒸馏、量化的方法压缩目标检测模型，提升推理速度的同时尽可能减少精度损失。
5. 一个**模型部署**示例。它将演示如何部署模型至嵌入式开发板，可有效应用于无人机等小型设备。

## 安装


## 使用说明
![图片3](https://user-images.githubusercontent.com/44053847/229462603-d1bc5589-5665-4332-b7e8-378ca13ee258.png)


### 数据处理

1. 数据集增强与扩充
   准备voc格式数据集，coco、yolo等格式数据使用data_process文件夹下的代码做格式转换。
   修改Augment.py中的相关路径，选择增强方式（默认全选）。
   修改增强相关参数。
   运行Augment.py。
   
2. 样本筛选
   建立用于存放有效样本和无效样本的文件夹。
   修改Sample-Selection/Sample_Selection.py中的对应路径。
   运行Sample-Selection/Sample_Selection.py。
   
### YOLO训练步骤

1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，** coco、yolo等格式数据使用data_process文件夹下的代码做格式转换   
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。
   
### YOLO预测步骤
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolov7_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #------------------------------------------------------#
    #   所使用到的yolov7的版本，本仓库一共提供两个：
    #   l : 对应yolov7
    #   x : 对应yolov7_x
    #------------------------------------------------------#
    "phi"               : 'l',
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入  
```
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### YOLO评估步骤

1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## 示例

想了解我们的框架是如何被应用的，请参考

## 相关仓库

- [Art of Readme](https://github.com/noffle/art-of-readme)
- [Classification by bubbliiiing](https://github.com/bubbliiiing/classification-pytorch/)
- [YOLOv7 by bubbliiiing](https://github.com/bubbliiiing/yolov7-pytorch)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

## 维护者

[@HHU_Leon](https://github.com/2436917927)。

## 如何贡献


### 贡献者


