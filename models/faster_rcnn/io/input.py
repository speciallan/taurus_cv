#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
Faster-rcnn 输入数据准备
"""

from datasets.dataset import VocDetectionDataset

def get_prepared_detection_dataset(config):
    """
    准备目标检测数据集
    :param config:
    :return:
    """

    return VocDetectionDataset(config.voc_path, class_mapping=config.CLASS_MAPPING).prepare()
