#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import random
import numpy as np
import cv2


def generator(image_list, batch_size, image_size=(512,512), stage='train'):

    print('generator')

    length = len(image_list)
    idx_list = range(length)

    while True:

        ids = random.sample(idx_list, batch_size)
        batch_image = []
        batch_class_ids = []
        batch_bbox = []

        for id in ids:

            image = cv2.imread(image_list[id]['filepath'])
            image = cv2.resize(image, image_size)
            bbox = image_list[id]['boxes']

            batch_image.append(image)
            batch_bbox.append(bbox)
            batch_class_ids.append(image_list[id]['labels'])

        if stage == 'train':
            # yield [np.asarray(batch_image),
            #        np.asarray(batch_class_ids),
            #        np.asarray(batch_bbox)]
            yield [np.asarray(batch_image),
                   np.asarray(batch_class_ids),
                   np.asarray(batch_bbox)]
        else:
            yield [np.asarray(batch_image)]


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

