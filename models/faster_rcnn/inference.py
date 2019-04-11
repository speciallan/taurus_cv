#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
faster-rcnn推理
"""

import numpy as np
import matplotlib
import os

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from .io.input import get_prepared_detection_dataset
from .config import current_config as config
from .layers import network
from .utils import visualize, np_utils
from utils.image import load_image_gt


def inference(output_dir):

    # 加载数据
    dataset = get_prepared_detection_dataset(config)

    # 获取所有测试集图片
    all_img_list = [info for info in dataset.get_image_list() if info['type'] == dataset.TEST_LABEL]  # 测试集

    # 加载模型
    m = network.faster_rcnn(config, stage='test')
    m.load_weights(config.rcnn_weights, by_name=True)
    # m.summary()

    # class map 转为 id map
    id_mapping = class_map_to_id_map(config.CLASS_MAPPING)

    def _show_inference(id, ax=None):
        image, image_meta, _ = load_image_gt(id,
                                             all_img_list[id]['filepath'],
                                             config.IMAGE_MAX_DIM,
                                             all_img_list[id]['boxes'])
        # 预测
        boxes, scores, class_ids, class_logits = m.predict([np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])

        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        class_ids = np_utils.remove_pad(class_ids[0])[:, 0]

        # 只取score大于0.7的框
        # right_idx = [i for i in range(len(scores)) if scores[i] < 0.7]
        # scores = np.delete(scores, right_idx, axis=0)
        # boxes = np.delete(boxes, right_idx, axis=0)
        # class_ids = np.delete(class_ids, right_idx, axis=0)

        # 画框到原图
        visualize.display_instances(image, boxes, class_ids, id_mapping, scores=scores, ax=ax)

    # 随机展示9张图像
    image_ids = np.random.choice(len(all_img_list), 16, replace=False)

    # 画图并展示
    fig = plt.figure(figsize=(20, 20))
    for idx, image_id in enumerate(image_ids):
        ax = fig.add_subplot(4, 4, idx + 1)
        _show_inference(image_id, ax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fig.savefig(output_dir + '/inferece_examples.{}.png'.format(np.random.randint(10)))


def class_map_to_id_map(class_mapping):

    id_map = {}
    for k, v in class_mapping.items():
        id_map[v] = k

    return id_map


if __name__ == '__main__':
    pass

