
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
classes = ['0', '1', '2', '3', '4', '5', '6', '7']

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
img_info = []

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

    # if test_img_list[id]['filename'] == '000227.jpg':
    #     print(id, test_img_list[id])
    #     exit()

    if os.path.isfile(imgfp):

        try:
            img = read_image_bgr(imgfp)
        except:
            continue

        img = preprocess_image(img.copy())
        img_info.append(test_img_list[id])

    else:
        print('not exist:', imgfp)


annotations = eval_utils.get_annotations(img_info, len(classes), order=True)
font = cv2.FONT_HERSHEY_SIMPLEX

for id, info in enumerate(img_info):

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

            show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            show_img = cv2.putText(show_img, str(classid), (x1 - 2, y1 - 2), font, 0.5, (0, 255, 0), 1)

    combine = np.array(np.zeros((512, 512*3, 3)))
    combine[:, 0:512, :] = origin_img
    combine[:, 512:512*2, :] = show_img
    combine[:, 512*2:512*3, :] = results_img
    # combine = cv2.vconcat(origin_img, show_img)
    cv2.imwrite(show_path, combine)
