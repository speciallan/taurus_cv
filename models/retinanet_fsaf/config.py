#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
系统级默认配置文件
"""

import numpy as np
import configparser
import os
from taurus_cv.pretrained_models.get import get as get_pretrained_model

# Faster_rcnn 基础配置
class Config(object):

    # 配置名称
    NAME = 'retinanet_fsaf'

    # 并行GPU数量
    GPU_COUNT = 1

    # 每次训练的图片数量 1080ti可以2 8g显存为1
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_INPUT_SHAPE = (IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

    # batch_size
    BATCH_SIZE = IMAGES_PER_GPU

    NUM_CLASSES = 8
    CLASS_MAPPING = {'0': 0,
                     '1': 1,
                     '2': 2,
                     '3': 3,
                     '4': 4,
                     '5': 5,
                     '6': 6,
                     '7': 7
                     }

    # 每个epoch需要训练的次数
    STEPS_PER_EPOCH = 1000

    # CNN的架构
    BACKBONE = 'resnet50'

    # FPN
    TOP_DOWN_PYRAMID_SIZE = 256

    # 用于FPN（目前没用）
    FPN_CLF_FC_SIZE = 1024

    # FPN
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # 网络步长
    BACKBONE_STRIDE = 16

    # anchors
    RPN_ANCHOR_BASE_SIZE = 64
    RPN_ANCHOR_SCALES = [1, 2 ** 1, 2 ** 2]
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_NUM = len(RPN_ANCHOR_SCALES) * len(RPN_ANCHOR_RATIOS)
    # RPN_ANCHOR_STRIDE = 1

    # 不同数据集这个阈值不同
    RPN_SCORE_THRESHOLD = 0.1

    # RPN提议框非极大抑制阈值(训练时可以增加该值来增加提议框) 越小框越少 训练时用0.9有很多rois
    RPN_NMS_THRESHOLD_TRAIN = 0.9 # 获取几乎所有正确rois
    RPN_NMS_THRESHOLD_INFERENCE = 0.01 # 获取最好的无重叠roi

    # 每张图像训练anchors个数
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    RPN_POSITIVE_THRESHOLD=0.2
    RPN_NEGATIVE_THRESHOLD=0.2

    # 训练和预测阶段NMS后保留的ROIs数
    POST_NMS_ROIS_TRAIN = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # 检测网络训练rois数和正样本比
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = 0.33

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # 最大ground_truth实例
    MAX_GT_INSTANCES = 100

    # 最大最终检测实例数
    DETECTION_MAX_INSTANCES = 100

    # 训练时通过roi和gt 的iou生成正负例 <0.2负 >=0.2正
    DETECTION_POSITIVE_THRESHOLD=0.2
    DETECTION_NEGATIVE_THRESHOLD=0.2

    # 检测最小置信度
    DETECTION_MIN_CONFIDENCE = 0.7

    # 检测MNS阈值
    DETECTION_NMS_THRESHOLD = 0.3

    # 训练参数
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # 权重衰减
    WEIGHT_DECAY = 0.0001

    # 梯度裁剪
    GRADIENT_CLIP_NORM = 1.0

    # 损失函数权重
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "rcnn_class_loss": 1.,
        "rcnn_bbox_loss": 1.
    }


# PASCAL VOC2007数据集配置
class VocConfig(Config):

    NAME = 'retinanet_fsaf'

    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_INPUT_SHAPE = (IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

    # batch_size
    BATCH_SIZE = IMAGES_PER_GPU

    NUM_CLASSES = 8
    CLASS_MAPPING = {'0': 0,
                     '1': 1,
                     '2': 2,
                     '3': 3,
                     '4': 4,
                     '5': 5,
                     '6': 6,
                     '7': 7
                     }

    backbone_weight_path = ''
    pretrained_weights = get_pretrained_model(weight_path=backbone_weight_path, network=Config.BACKBONE)

    config_filepath = './config.ini'
    rpn_weights = '/tmp/faster-rcnn-rpn.h5'
    rcnn_weights = '/tmp/faster-rcnn-rcnn.h5'
    voc_path = '/home/speciallan/Documents/python/data/VOCdevkit'
    voc_sub_dir = 'VOC2007'
    log_path = './logs'

    # 特征提取的层
    backbone_output_layer_name = None

    if not os.path.exists(config_filepath):
        print('找不到实验的config.ini')

    if not os.path.exists(log_path):
        os.mkdir(log_path)

# Coco数据集配置
class CocoConfig(Config):
    pass

# 家里的mac voc配置
class MacVocConfig(VocConfig):
    pass

# 家里的linux voc配置
class LinuxVocConfig(VocConfig):
    username = ''
    voc_path = '/home/speciallan/Documents/python/data/VOCdevkit'

# 实验室voc配置
class LabVocConfig(VocConfig):
    VocConfig.voc_path = ''
    pass

# 当前配置
current_config = LinuxVocConfig()

# 获取用户配置
cf = configparser.ConfigParser()
cf.read(current_config.config_filepath)
sections = cf.sections()

for k,section in enumerate(sections):
    user_config = cf.items(section)
    for k2,v in enumerate(user_config):
        current_config.__setattr__(v[0], v[1])
