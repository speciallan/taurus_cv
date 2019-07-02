#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
去掉train.txt中 JPEGImages没有的图片
"""

import os
import shutil

old_data_path = '../../../../data/VOCdevkit/dd2/ImageSets/Main/'

filelist = ['train', 'val', 'trainval', 'test']

num = [str(i) for i in range(397, 463 + 1)]
# t = '000397\n'
# t = t[3:6]
# print(t in num)
# exit()

for k,v in enumerate(filelist):
    filename = old_data_path + v + '.txt'
    file = open(filename)
    content = file.readlines()

    new_content = []

    for k2,v2 in enumerate(content):
        id = v2[3:6]
        if id in num:
           continue
        new_content.append(v2)

    new_filename = old_data_path + v + '_new.txt'
    new_file = open(new_filename, 'w')
    new_file.writelines(new_content)

