#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
评估模型
"""

import argparse
import sys
import numpy as np

from utils import image as image_utils

from .io.input import get_prepared_detection_dataset
from .config import current_config as config
from .utils import np_utils, eval_utils
from .layers import network


def evaluate(args, image_num=200):

    # 加载数据集
    dataset = get_prepared_detection_dataset(config)

    test_image_list = [info for info in dataset.get_image_list() if info['type'] == dataset.TEST_LABEL]

    print("测试集图片数量:{}".format(len(test_image_list)))

    # 加载模型
    model = network.faster_rcnn(config, stage='test')

    if args.weight_path is not None:
        model.load_weights(args.weight_path, by_name=True)
    else:
        model.load_weights(config.rcnn_weights, by_name=True)

    # m.summary()
    # 预测边框、得分、类别
    predict_boxes = []
    predict_scores = []
    predict_labels = []

    # 一次最多只能跑200张
    test_image_list = test_image_list[:image_num]

    # 通过测试集验证模型
    # 内存不够，要使用生成器载入，一次100-200张
    for id in range(len(test_image_list)):
        image, image_meta, _ = image_utils.load_image_gt(id, test_image_list[id]['filepath'], config.IMAGE_MAX_DIM, test_image_list[id]['boxes'])

        # 预测结果，每次预测一张图
        boxes, scores, class_ids, class_logits = model.predict([np.expand_dims(image, axis=0), np.expand_dims(image_meta, axis=0)])

        boxes = np_utils.remove_pad(boxes[0])
        scores = np_utils.remove_pad(scores[0])[:, 0]
        class_ids = np_utils.remove_pad(class_ids[0])[:, 0]

        # 还原检测边框到
        window = image_meta[7:11]
        scale = image_meta[11]
        boxes = image_utils.recover_detect_boxes(boxes, window, scale)

        # 添加到列表中
        predict_boxes.append(boxes)
        predict_scores.append(scores)
        predict_labels.append(class_ids)

        if id % 100 == 0:
            print('预测完成：{}'.format(id + 1))

    # 以下是评估过程
    annotations = eval_utils.get_annotations(test_image_list, config.NUM_CLASSES)
    detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, config.NUM_CLASSES)
    average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.5, use_07_metric=True)

    print("ap:{}".format(average_precisions))

    # 求mean ap 去除背景类
    mAP = np.mean(np.array(list(average_precisions.values()))[1:])
    print("mAP:{}".format(mAP))

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    evaluate(argments)
