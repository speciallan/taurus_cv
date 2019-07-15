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
from taurus_cv.models.fsaf.preprocessing.image import preprocess_image, resize_image
from taurus_cv.models.fsaf.config import current_config as config
from taurus_cv.utils.spe import spe


def inference(args):

    start_time = time.time()

    model = retinanet(config)
    model.load_weights(config.retinanet_weights, by_name=True)

    test_img_list = get_prepared_detection_dataset(config).get_all_data()
    # test_img_list = test_img_list[:100]

    # 基本设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    classes = config.CLASS_MAPPING
    classes = dict(zip(classes.values(), classes.keys()))

    for idx, img_info in enumerate(test_img_list):
        if os.path.exists(img_info['filepath']):

            img = cv2.imread(img_info['filepath'])
            # img = preprocess_image(img.copy())
            # img, scale = resize_image(img, min_side=config.IMAGE_MIN_DIM, max_side=config.IMAGE_MAX_DIM)
            _, _, detections = model.predict(np.expand_dims(img, axis=0))

            scores = detections[0, :, 4:]

            # 选取置信度大于0.3
            # ([0, 0, 1, 2, 3, 4, 5, 6], [2, 3, 3, 2, 3, 6, 6, 3])
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

            # 展示图片
            show_img = img.copy()

            # 画框
            if len(img_boxes) > 0:
                for i, box in enumerate(img_boxes):
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])

                    show_img = cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), color, 1)
                    show_img = cv2.putText(show_img, '{} {:2f}'.format(classes[img_labels[i]], img_scores[i][0]), (xmin, ymin-2), font, 0.5, color, 1)

            cv2.imwrite(img_info['filepath'].replace('JPEGImages', 'results'), show_img)
            print('生成图片 {} time:{}'.format(img_info['filename'], time.time() - start_time))
            start_time = time.time()

        else:
            print('图片 {} 不存在'.format(img_info['filename']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])

    inference(args)
