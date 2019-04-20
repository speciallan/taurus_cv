#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
模型的输入
"""

import numpy as np
from .pascal_voc import get_voc_dataset


class Dataset(object):
    """
    基础数据集
    """

    def __init__(self, stage='train', class_mapping=None):
        """
        数据集初始化
        :param stage: 分阶段获取数据集
        :param class_mapping: 分类映射图
        """
        self.stage = stage
        self.class_mapping = class_mapping

        # 图片字典列表，包含的key有boxes,labels,filename,filepath,type等
        self.image_list = []

    # 获取图片
    def get_image_list(self):
        return self.image_list

    def prepare(self):
        """
        将数据集转为标准的目标检测数据集格式
        :return:
        """
        raise NotImplementedError('请实现具体的数据准备方法')

    def get_train_data(self):
        """
        获取训练集数据
        :return:
        """
        raise NotImplementedError('请实现具体的训练集方法')

    def get_test_data(self):
        """
        获取测试集数据
        :return:
        """
        raise NotImplementedError('请实现具体的测试集方法')

class VocDataset(Dataset):

    def __init__(self, path, **kwargs):
        """
        VOC数据集初始化
        :param path: 数据集路径，对于不同的数据集增加了path属性
        :param kwargs:
        """

        # 训练集，测试集标记
        self.TRAIN_LABEL = 'trainval'
        self.TEST_LABEL  = 'test'

        self.path = path

        super(VocDataset, self).__init__(**kwargs)


class VocClassificationDataset(VocDataset):
    pass


class VocSegmentDataset(VocDataset):
    pass


class VocDetectionDataset(VocDataset):
    """
    VOC目标检测数据准备
    """

    def __init__(self, path, **kwargs):
        super(VocDetectionDataset, self).__init__(path, **kwargs)

    def prepare(self):

        img_info_list, classes_count, class_mapping = get_voc_dataset(self.path, self.class_mapping)

        for img_info in img_info_list:

            image_info = {"filename": img_info['filename'],
                          "filepath": img_info['filepath'],
                          "type"    : img_info['imageset']}
            # GT 边框转换
            boxes, labels = [], []

            # 训练阶段加载边框标注信息
            if self.stage == 'train':
                for bbox in img_info['bboxes']:
                    y1, x1, y2, x2 = bbox['y1'], bbox['x1'], bbox['y2'], bbox['x2']
                    boxes.append([y1, x1, y2, x2])
                    labels.append(bbox['class_id'])

            image_info['boxes'] = np.array(boxes)
            image_info['labels'] = np.array(labels)

            self.image_list.append(image_info)

        return self

    def get_train_data(self):

        return [info for info in self.get_image_list() if info['type'] == self.TRAIN_LABEL]

    def get_test_data(self):

        return [info for info in self.get_image_list() if info['type'] == self.TEST_LABEL]
