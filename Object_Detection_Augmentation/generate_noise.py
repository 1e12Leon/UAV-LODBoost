import cv2
import numpy as np
import os
import shutil

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始数据集所在的路径
#   Out_VOCdevkit_path      输出数据集所在的路径
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = "VOCdevkit_Origin"
Out_VOCdevkit_path      = "VOCdevkit"
noise_type = ["s&p", "gaussian", "poisson", "speckle"]

# 椒盐噪声
def SaltAndPepper(image, percentage=0.3):
    """
    :param image:输入图片
    :param percentage:椒盐噪声比例
    :return:处理后的椒盐噪声图片

    """
    SP_NoiseImg = image.copy()
    h = image.shape[0]
    w = image.shape[1]
    SP_NoiseNum = int(percentage * h * w)
    for _ in range(SP_NoiseNum):
        randR = np.random.randint(0, h)
        randG = np.random.randint(0, w)
        if np.random.randint(0, 2) == 0:
            SP_NoiseImg[randR, randG] = 0
        else:
            SP_NoiseImg[randR, randG] = 255
    return SP_NoiseImg

# 高斯噪声
def addGaussionNoise(image, mean=0, var=0.01):
    """
    :param image:cv2所读取的图片
    :param mean:高斯噪声分布均值
    :param var:高斯噪声分布标准差
    :return:处理后的高斯噪声图片

    """
    # 将图片灰度标准化
    img = np.array(image/255, dtype=float)
    # 产生高斯噪声
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    # 将噪声和图片叠加
    G_NoiseImg = img + noise
    # 定义低阈值
    if G_NoiseImg.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    # 将低于low_clip的置为low_clip,将超过1的置为1
    G_NoiseImg = np.clip(G_NoiseImg, low_clip, 1.0)
    # 将图片灰度范围恢复为0-255
    G_NoiseImg = np.uint8(G_NoiseImg*255)

    return G_NoiseImg

# 泊松噪声
def addPoissonNoise(image, lam=0.03):
    """
     :param image:cv2所读取的图片
     :param lam:泊松噪声出现的期望
     :return:处理后的泊松噪声图片

    """
    noise_type = np.random.poisson(lam=lam, size=(image.shape[0], image.shape[1], 1)).astype(dtype='uint8')
    noise_image = image + noise_type
    return noise_image

# 散斑噪声
def addSpeckleNoise(image):
    """
     :param image:cv2所读取的图片
     :return:处理后的散斑噪声图片

    """
    Speckle_NoiseImg = image.copy()
    # 随机生成一个服从分布的噪声
    gauss = np.random.randn(image.shape[0], image.shape[1], image.shape[2])
    # 给图片添加speckle噪声
    Speckle_NoiseImg = Speckle_NoiseImg + Speckle_NoiseImg * gauss
    # 归一化图像的像素值
    Speckle_NoiseImg = np.clip(Speckle_NoiseImg, a_min=0, a_max=255)
    return Speckle_NoiseImg

def addNoise(image, type):
    """
     :param image:cv2所读取的图片
     :type:四种噪声类型-椒盐噪声、高斯噪声、泊松噪声、散斑噪声
     :return:处理后的散斑噪声图片

    """
    if type == "s&p":
        return SaltAndPepper(image)
    elif type == "gaussian":
        return addSpeckleNoise(image)
    elif type == "poisson":
        return addPoissonNoise(image)
    elif type == "speckle":
        return addSpeckleNoise(image)

if __name__ == "__main__":
    Origin_JPEGImages_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")

    Out_JPEGImages_path = os.path.join(Out_VOCdevkit_path, "VOC2007/JPEGImages")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "VOC2007/Annotations")

    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)

    for img_name in os.listdir(Origin_JPEGImages_path):
        name = img_name.split('.')[0]
        img_src_path = os.path.join(Origin_JPEGImages_path, img_name)
        img_dst_path = os.path.join(Out_JPEGImages_path, img_name)
        img = cv2.imread(img_src_path)
        xml_src_path = os.path.join(Origin_Annotations_path, name+'.xml')
        xml_dst_path = os.path.join(Out_Annotations_path, name+'xml')


        type = noise_type[np.random.randint(4)]
        print(type)
        noise_img = addNoise(img, type)
        cv2.imwrite(img_dst_path, noise_img)
        shutil.copyfile(xml_src_path, xml_dst_path)

