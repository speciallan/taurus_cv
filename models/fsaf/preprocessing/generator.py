#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import random
import numpy as np
import cv2
import keras

from taurus_cv.models.fsaf.preprocessing.anchors import anchor_targets_bbox, bbox_transform
from taurus_cv.models.fsaf.preprocessing.image import preprocess_image
from taurus_cv.utils.spe import spe


def generator(image_list, num_classes, batch_size, image_size=(512,512), stage='train'):

    length = len(image_list)
    idx_list = range(length)

    while True:

        ids = random.sample(idx_list, batch_size)
        batch_images = []
        batch_annotations = []

        for id in ids:

            # img
            image = cv2.imread(image_list[id]['filepath'])
            image = cv2.resize(image, image_size)

            # anno
            gt_bbox = image_list[id]['boxes']
            label = np.array([image_list[id]['labels']], dtype='float').T
            gt_bbox_with_label = np.append(gt_bbox, label, axis=1)

            # 预处理
            img_preprocessed = preprocess_image(image)

            batch_images.append(img_preprocessed)
            batch_annotations.append(gt_bbox_with_label)

        # model inputs
        inputs = np.array(batch_images)

        # model outputs
        outputs = get_outputs(batch_images, batch_annotations, batch_size, num_classes)
        # spe(inputs[0].shape, outputs[0].shape, outputs[1].shape, outputs[0][0][:5], outputs[1][0][:5])

        if stage == 'train':
            yield [inputs, outputs]
        else:
            yield [inputs]


# image_group (1,512,512,3) anno_group (1,2,5) 1张图，2个gtbox, 4个坐标+置信度
def get_outputs(image_batch, annotations_batch, batch_size, num_classes):

    # 512
    max_shape = tuple(max(image.shape[x] for image in image_batch) for x in range(3))

    labels_group = [None] * batch_size
    regression_group = [None] * batch_size

    for index, (image, annotations) in enumerate(zip(image_batch, annotations_batch)):

        labels_group[index], annotations, anchors = anchor_targets(max_shape, annotations, num_classes, mask_shape=image.shape)
        regression_group[index] = bbox_transform(anchors, annotations)

        anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
        regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

    # (1, 196416, 5) regression_group (1, 196416, 8) labels_group
    # print(regression_group[0].shape, labels_group[0][:5])
    # exit()

    labels_batch = np.zeros((batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
    regression_batch = np.zeros((batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

    for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
        labels_batch[index, ...] = labels
        regression_batch[index, ...] = regression

    # spe(regression_batch.shape, labels_batch.shape)
    # (1, 196416, 5) (1, 196416, 8)
    return [regression_batch, labels_batch]


def anchor_targets(image_shape,
                   annotations,
                   num_classes,
                   mask_shape=None,
                   negative_overlap=0.4,
                   positive_overlap=0.5,
                   **kwargs):
    return anchor_targets_bbox(image_shape, annotations, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)





if __name__ == '__main__':

    train_image_list = [
        {'filename': '000152.jpg',
         'filepath': '/home/speciallan/Documents/python/data/VOCdevkit/dd/JPEGImages/000152.jpg', 'type': 'trainval',
         'height': 512, 'width': 512, 'boxes':[[107, 100, 120, 115],
                                                      [245, 149, 259, 161],
                                                      [3, 20, 512, 142],
                                                      [2, 1, 25, 62]],
         'labels': ['6', '6', '4', '2']}
    ]

