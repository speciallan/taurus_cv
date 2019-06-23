#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import numpy as np


def gen_all():

    all_img = sorted(os.listdir('../../../../data/VOCdevkit/dd/JPEGImages'))

    f = open('../../../../data/VOCdevkit/dd/ImageSets/Main/all.txt', 'w')

    for k,v in enumerate(all_img):
        f.write('{}\n'.format(v.replace('.jpg', '')))

    f.close()
    print(all_img)


def test():
    a = np.array([[1, 2, 3], [2, 3, 4], [1, 6, 5], [9, 3, 4]])
    print(a)
    print(a[:, [1, 0, 2]])


if __name__ == '__main__':
    gen_all()
