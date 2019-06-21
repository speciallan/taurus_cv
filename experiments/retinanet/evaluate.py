
import sys
import argparse
sys.path.append('../../..')

import matplotlib as mpl

mpl.use('Agg')

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from taurus_cv.models.retinanet.model.pascal_voc import save_annotations
from taurus_cv.models.retinanet.model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb
from taurus_cv.models.retinanet.model.resnet import resnet_retinanet
from taurus_cv.models.retinanet.config import Config
from taurus_cv.models.faster_rcnn.utils import np_utils, eval_utils
from taurus_cv.utils.spe import spe

config = Config('configRetinaNet.json')

wname = 'BASE'
wpath = config.base_weights_path
classes = ['0', '1', '2', '3', '4', '5', '6']

if os.path.isfile(config.trained_weights_path):
    wname = "DEFINITIVI"
    wpath = config.trained_weights_path
    classes = config.classes
if os.path.isfile(config.pretrained_weights_path):
    wname = 'PRETRAINED'
    wpath = config.pretrained_weights_path
    classes = config.classes

if config.type.startswith('resnet'):
    model, _ = resnet_retinanet(len(classes), backbone=config.type, weights='imagenet', nms=True)
else:
    model = None
    print("模型 ({})".format(config.type))
    exit(1)

print("backend: ", config.type)

if os.path.isfile(wpath):
    model.load_weights(wpath, by_name=True, skip_mismatch=True)
    print("权重" + wname)
else:
    print("None")

# 预测边框、得分、类别
predict_boxes = []
predict_scores = []
predict_labels = []

start_index = config.test_start_index

from taurus_cv.models.faster_rcnn.config import current_config
from taurus_cv.models.faster_rcnn.io.input import get_prepared_detection_dataset
from taurus_cv.datasets.pascal_voc import get_voc_dataset

current_config.voc_sub_dir = 'dd'
# current_config.NUM_CLASSES = 7
# current_config.CLASS_MAPPING = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6}
test_img_list = get_prepared_detection_dataset(current_config).get_all_data()
# test_img_list = get_voc_dataset('../../../../data/VOCdevkit', 'dd', class_mapping=classes)

test_img_list = test_img_list[:100]
test_img_list = test_img_list[:5]


for id, imgf in enumerate(test_img_list):


    # imgfp = os.path.join(config.test_images_path, imgf)
    imgfp = imgf['filepath']

    if os.path.isfile(imgfp):

        try:
            img = read_image_bgr(imgfp)
        except:
            continue

        img = preprocess_image(img.copy())
        img, scale = resize_image(img, min_side=config.img_min_size, max_side=config.img_max_size)

        orig_image = read_image_rgb(imgfp)

        _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))


        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(img.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(img.shape[0], detections[:, :, 3])

        detections[0, :, :4] /= scale

        scores = detections[0, :, 4:]

        # 推测置信度 indices = [[0,1,2,3], [6,6,3,3]] idx + cls_labels
        indices = np.where(detections[0, :, 4:] >= 0.3)

        scores = scores[indices]

        # 取前100个idx [0,1,2,3]
        scores_sort = np.argsort(-scores)[:100]

        # 一张图的预测框
        image_boxes = detections[0, indices[0][scores_sort], :4]
        # spe(image_boxes, image_scores, image_detections)
        image_scores = detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]]
        image_predicted_labels = indices[1][scores_sort]

        # 添加到列表中
        predict_boxes.append(image_boxes)
        predict_scores.append(image_scores)

        image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        predict_labels.append(image_predicted_labels)

        if id % 100 == 0:
            print('预测完成：{}'.format(id + 1))


# 以下是评估过程
annotations = eval_utils.get_annotations(test_img_list, len(classes))
# spe(predict_boxes, predict_scores, predict_labels)
detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, len(classes))
average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.2, use_07_metric=True)

print("ap:{}".format(average_precisions))

# 求mean ap 去除背景类
mAP = np.mean(np.array(list(average_precisions.values()))[1:])
print("mAP:{}".format(mAP))

