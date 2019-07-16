#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import argparse
import time
import os
sys.path.append('../../..')

import cv2
import numpy as np

from taurus_cv.models.fsaf.io.input import get_prepared_detection_dataset
from taurus_cv.models.fsaf.networks.retinanet import retinanet
from taurus_cv.models.fsaf.config import current_config as config
from taurus_cv.models.faster_rcnn.utils import np_utils, eval_utils
from taurus_cv.utils.spe import spe


def show(args):

    test_img_list = get_prepared_detection_dataset(config).get_all_data()

    # 基本设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    classes = config.CLASS_MAPPING
    classes = dict(zip(classes.values(), classes.keys()))

    # 并不是testimg里面所有图片都存在，会出现问题
    exist_img_info = []
    for id, img_info in enumerate(test_img_list):

        imgfp = img_info['filepath']

        if os.path.isfile(imgfp):
            exist_img_info.append(test_img_list[id])
        else:
            print('图片不存在:', imgfp)


    annotations = eval_utils.get_annotations(exist_img_info, len(classes), order=True)

    for id, info in enumerate(exist_img_info):

        origin_path = info['filepath']
        print(origin_path)
        results_path = origin_path.replace('JPEGImages', 'results')
        show_path = origin_path.replace('JPEGImages', 'show')

        origin_img = cv2.imread(origin_path)
        show_img = origin_img.copy()
        results_img = cv2.imread(results_path)

        # 可视化gt
        all_boxes = annotations[id]

        for classid, boxes in enumerate(all_boxes):

            if len(boxes) == 0:
                continue

            # print(classid, boxes)

            for k,v in enumerate(boxes):

                if len(v) == 0:
                    continue

                x1, y1, x2, y2 = v

                show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), color, 1)
                show_img = cv2.putText(show_img, str(classid), (x1 - 2, y1 - 2), font, 0.5, color, 1)

        combine = np.array(np.zeros((512, 512*3, 3)))
        combine[:, 0:512, :] = origin_img
        combine[:, 512:512*2, :] = show_img
        combine[:, 512*2:512*3, :] = results_img
        # combine = cv2.vconcat(origin_img, show_img)
        cv2.imwrite(show_path, combine)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])

    show(args)
