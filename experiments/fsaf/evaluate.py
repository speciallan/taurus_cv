#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import argparse
import sys
import os
import time
sys.path.append('../../..')

import cv2
import numpy as np

from taurus_cv.models.fsaf.io.input import get_prepared_detection_dataset
from taurus_cv.models.fsaf.networks.retinanet import retinanet
from taurus_cv.models.fsaf.config import current_config as config
from taurus_cv.models.fsaf.utils import np_utils, eval_utils
from taurus_cv.utils.spe import spe


def evaluate(args):

    time_start = time.time()

    model = retinanet(config)
    model.load_weights(config.retinanet_weights, by_name=True)

    time_load_model = time.time() - time_start
    time_start = time.time()

    test_img_list = get_prepared_detection_dataset(config).get_all_data()
    test_img_list = test_img_list[:10]

    # 预测边框、得分、类别
    predict_boxes = []
    predict_scores = []
    predict_labels = []
    exist_img_info = []

    for id, img_info in enumerate(test_img_list):

        if os.path.exists(img_info['filepath']):

            img = cv2.imread(img_info['filepath'])
            _, _, detections = model.predict(np.expand_dims(img, axis=0))

            scores = detections[0, :, 4:]

            # 选取置信度大于0.3
            indices = np.where(detections[0, :, 4:] >= 0.3)
            scores = scores[indices]

            # 排序选择前100个大的, 得到排序方式 比如[1,0,2]
            sort_by_scores = np.argsort(-scores)[:100]
            indices_sorted = [indices[0][sort_by_scores], indices[1][sort_by_scores]]

            img_boxes = detections[0, indices_sorted[0], :4]

            # 取detection后面类别位，前4位是坐标
            img_scores = np.expand_dims(detections[0, indices_sorted[0], 4 + indices_sorted[1]], axis=1)
            img_detections = np.append(img_boxes, img_scores, axis=1)
            img_labels = indices_sorted[1]

            # 添加到列表中
            predict_boxes.append(img_boxes)
            predict_scores.append(img_scores)
            predict_labels.append(img_labels)
            exist_img_info.append(test_img_list[id])

            if id % 100 == 0:
                print('预测完成：{}'.format(id + 1))

        else:
            print('图片 {} 不存在'.format(img_info['filename']))


    # 以下是评估过程 这里img_info是y1,x1,y2,x2
    # 找到问题了 anno 和 pre_boxes 没对应， 导致后面detection错误 修改了get_annotations 里面-1
    classes = config.CLASS_MAPPING
    annotations = eval_utils.get_annotations(exist_img_info, len(classes), cordinate_order=True)
    detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, len(classes))
    average_precisions = eval_utils.voc_eval(annotations, detections, img_info=exist_img_info, iou_threshold=0.05, use_07_metric=True)


    if 1 == 0:
        # y1,x1,y2,x2 | 2 1 25 62 | 107 100 120 115, 通过order=True修改顺序为下面那种 1,2,62,25
        for k,v in enumerate(annotations):
            for k2,v2 in enumerate(v):
                print('img id:{}, class id:{}, boxes:{}'.format(test_img_list[k]['filename'], k2, v2))

        # x1,y1,x2,y2 | 2.6678467    0.88687134  62.456413    24.451286 | 99.203384   105.607574   112.44885    119.33241
        for k,v in enumerate(predict_boxes):
            print(predict_labels[k], v)
        spe(11)
        # spe(predict_boxes, predict_scores, predict_labels)

    print("ap:{}".format(average_precisions))

    # 求mean ap 去除背景类
    mAP = np.mean(np.array(list(average_precisions.values()))[1:])
    print("mAP:{}".format(mAP))

    time_inference = (time.time() - time_start) / len(test_img_list)
    print('load_model_time:{}'.format(time_load_model))
    print('inference_time:{}'.format(time_inference))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])

    evaluate(args)
