from Object_Detection_Augmentation.noise_aug import NoiseAugmentation
from Object_Detection_Augmentation.cutmix_aug import CutmixAugmentation
from Object_Detection_Augmentation.small_object import Samll_object_Augmentation
from Object_Detection_Augmentation.CycleGAN.predict import cycleGANAugmentaion

if __name__ == '__main__':
    """
    # -----------------------------------#
    #   Part1路径修改：
    #       输入图片、XML路径应为数据集路径
    #       输出图片、XML路径应为新路径
    #       未防止混淆，建议提前备份
    # -----------------------------------#
    """
    img_path = "Data/JPEGImages"  # 原始图片文件夹路径
    xml_path = "Data/Annotations"  # 原始图片对应的标注文件xml文件夹的路径
    img_save_path = "Data/aug_JPEGImages"  # 增强的图片文件夹路径
    xml_save_xml = "Data/aug_Annotations"  # 增强的图片对应的标注文件xml的文件夹路径

    """
    # -----------------------------------#
    #   Part2增强方法调用
    # -----------------------------------#
    """
    # -----------------------------------#
    #   （1）噪声增强
    # -----------------------------------#
    gs_mean = 0  # 高斯噪声均值
    gs_var = 0.01  # 高斯噪声var
    NoiseAugmentation(img_path, xml_path, img_save_path, xml_save_xml, gs_mean, gs_var)

    # -----------------------------------#
    #   （2）cutmix增强
    # -----------------------------------#
    CutmixAugmentation(img_path, xml_path, img_save_path, xml_save_xml)

    # -----------------------------------#
    #   （3）小目标增强
    # -----------------------------------#
    """Low_SOA_THRESH = 128*128
    SOA_THRESH = 256 * 256  # 复制最大尺寸(如果尺寸小于64*64就不复制)
    SOA_PROB = 1  # 百分之百复制
    SOA_COPY_TIMES = 3  # 复制的个数。(如果小于64*64就会复制3个)
    SOA_EPOCHS = 30  # 轮次
    Samll_object_Augmentation(img_path, xml_path, img_save_path,xml_save_xml, SOA_THRESH, SOA_PROB,
                              SOA_COPY_TIMES, SOA_EPOCHS,Low_SOA_THRESH)"""

    # -----------------------------------#
    #   （4）cyclegan增强
    # -----------------------------------#
    cycleGANAugmentaion(img_path, xml_path, img_save_path, xml_save_xml)
