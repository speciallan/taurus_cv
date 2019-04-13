#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
数据生成器
"""

import random
import numpy as np
from ..utils import np_utils
from utils import image as image_util


def image_generator(image_list, batch_size, max_output_dim, max_gt_num, stage='train'):
    """
    生成器
    :param image_list: 字典列表
    :param batch_size: 批数据尺寸
    :param max_output_dim:
    :param max_gt_num:
    :param stage:
    :return:
    """
    image_length = len(image_list)
    id_list = range(image_length)

    while True:
        ids = random.sample(id_list, batch_size)
        batch_image = []
        batch_image_meta = []
        batch_class_ids = []
        batch_bbox = []

        for id in ids:
            # 图像数据，图像元数据，回归框
            image, image_meta, bbox = image_util.load_image_gt(id,
                                                               image_list[id]['filepath'],
                                                               max_output_dim,
                                                               image_list[id]['boxes'])
            batch_image.append(image)
            batch_image_meta.append(image_meta)

            if stage == 'train':
                # gt个数固定
                batch_class_ids.append(np_utils.pad_to_fixed_size(np.expand_dims(image_list[id]['labels'], axis=1), max_gt_num))
                batch_bbox.append(np_utils.pad_to_fixed_size(bbox, max_gt_num))

        if stage == 'train':
            yield [np.asarray(batch_image),
                   np.asarray(batch_image_meta),
                   np.asarray(batch_class_ids),
                   np.asarray(batch_bbox)], None
        else:
            yield [np.asarray(batch_image),
                   np.asarray(batch_image_meta)]

