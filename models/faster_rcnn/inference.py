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

from taurus_cv.models.faster_rcnn.io.input import get_prepared_detection_dataset
from taurus_cv.models.faster_rcnn.layers import network
from taurus_cv.models.faster_rcnn.utils import visualize, np_utils
from taurus_cv.models.faster_rcnn.preprocessing.image import load_image_gt
from taurus_cv.models.faster_rcnn.training import trainer


def inference(config, output_dir):

    # 设置运行时环境 / training.trainer模块
    trainer.set_runtime_environment()

    # 加载数据
    test_img_list = get_prepared_detection_dataset(config).get_test_data()

    # 加载模型
    model = network.faster_rcnn(config, stage='test')
    model.load_weights(config.rcnn_weights, by_name=True)

    # model.summary()

    # class map 转为 id map
    id_mapping = class_map_to_id_map(config.CLASS_MAPPING)

    def _show_inference(id, ax=None):

        image, image_meta, _ = load_image_gt(id,
                                             test_img_list[id]['filepath'],
                                             config.IMAGE_MAX_DIM,
                                             test_img_list[id]['boxes'])
        # 预测
        boxes, scores, class_ids, class_logits = model.predict([np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])

        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        class_ids = np_utils.remove_pad(class_ids[0])[:, 0]

        # 只取score大于0.7的框
        # right_idx = [i for i in range(len(scores)) if scores[i] < 0.7]
        # scores = np.delete(scores, right_idx, axis=0)
        # boxes = np.delete(boxes, right_idx, axis=0)
        # class_ids = np.delete(class_ids, right_idx, axis=0)

        # 打印box score class_id数据
        # print(boxes.shape, scores.shape, class_ids.shape)
        # print(boxes, scores, class_ids)

        # 画框到原图
        visualize.display_instances(image, boxes, class_ids, id_mapping, scores=scores, ax=ax)

    # 随机展示16张图像
    image_ids = np.random.choice(len(test_img_list), 16, replace=False)

    # 画图并展示
    fig = plt.figure(figsize=(20, 20))
    for idx, image_id in enumerate(image_ids):
        ax = fig.add_subplot(4, 4, idx + 1)
        _show_inference(image_id, ax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_img_filename = output_dir + '/inference_examples_{}.png'.format(np.random.randint(10))
    fig.savefig(save_img_filename)
    print('可视化到:{}'.format(save_img_filename))


def inference_rpn(config, output_dir):

    # 设置运行时环境 / training.trainer模块
    trainer.set_runtime_environment()

    # 加载数据
    test_img_list = get_prepared_detection_dataset(config).get_test_data()

    # 加载模型
    model = network.rpn_net(config, stage='test')
    model.load_weights(config.rpn_weights, by_name=True)

    # class map 转为 id map
    id_mapping = class_map_to_id_map(config.CLASS_MAPPING)

    def _show_inference(id, ax=None):

        image, image_meta, _ = load_image_gt(id,
                                             test_img_list[id]['filepath'],
                                             config.IMAGE_MAX_DIM,
                                             test_img_list[id]['boxes'])
        # 预测
        X_test = [np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)]

        detect_boxes, class_scores = model.predict(X_test)

        rois_boxes = detect_boxes[0][:20][:, :4] # 取4个坐标
        rois_scores = class_scores[0][:20][:, :1] # 取前景分
        class_ids = np.ones(rois_boxes.shape[0]).astype('int') # 整数用于mapping字典
        print(rois_boxes.shape)

        # print(rois_boxes, '\n', rois_scores)
        id_mapping = {0:'bg', 1:'object'}

        # 画框到原图
        visualize.display_instances(image, rois_boxes, class_ids, id_mapping, scores=rois_scores, ax=ax)

    # 随机展示16张图像
    image_ids = np.random.choice(len(test_img_list), 1, replace=False)

    # 画图并展示
    fig = plt.figure(figsize=(20, 20))
    for idx, image_id in enumerate(image_ids):
        ax = fig.add_subplot(1, 1, idx + 1)
        _show_inference(image_id, ax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_img_filename = output_dir + '/inference_rpn_examples_{}.png'.format(np.random.randint(10))
    fig.savefig(save_img_filename)
    print('可视化到:{}'.format(save_img_filename))

def class_map_to_id_map(class_mapping):

    id_map = {}
    for k, v in class_mapping.items():
        id_map[v] = k

    return id_map


if __name__ == '__main__':
    pass

