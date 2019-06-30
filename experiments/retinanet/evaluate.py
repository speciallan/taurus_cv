
import sys
import argparse
sys.path.append('../../..')

import matplotlib as mpl

mpl.use('Agg')

import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from taurus_cv.models.retinanet.model.pascal_voc import save_annotations
from taurus_cv.models.retinanet.model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb
from taurus_cv.models.retinanet.model.resnet import resnet_retinanet
from taurus_cv.models.retinanet.config import Config
from taurus_cv.models.faster_rcnn.utils import np_utils, eval_utils
from taurus_cv.utils.spe import spe

time_start = time.time()

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

time_load_model = time.time() - time_start
time_start = time.time()

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

# test_img_list = test_img_list[:100]


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
        indices = np.where(detections[0, :, 4:] >= 0.25)

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
        predict_labels.append(image_predicted_labels)

        image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)

        if id % 100 == 0:
            print('预测完成：{}'.format(id + 1))


# 以下是评估过程
annotations = eval_utils.get_annotations(test_img_list, len(classes), order=True)
detections = eval_utils.get_detections(predict_boxes, predict_scores, predict_labels, len(classes))
average_precisions = eval_utils.voc_eval(annotations, detections, iou_threshold=0.05, use_07_metric=True)

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
