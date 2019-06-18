#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import shutil

old_data_path = '../../../../data/VOCdevkit/gangdaiyin_new'
new_data_path = '../../../../data/VOCdevkit/gangdaiyin_new2'

old_anno = old_data_path + '/Annotations/'
new_anno = new_data_path + '/Annotations/'
anno = os.listdir(old_anno)

for k,filename in enumerate(anno):

    new_filename = '4' + filename[1:]

    # print(new_anno + new_filename)
    # exit()
    shutil.copy(old_anno + filename, new_anno + new_filename)

old_img = old_data_path + '/JPEGImages/'
new_img = new_data_path + '/JPEGImages/'
img = os.listdir(old_img)

for k,filename in enumerate(img):

    new_filename = '4' + filename[1:]

    # print(new_anno + new_filename)
    # exit()
    shutil.copy(old_img + filename, new_img + new_filename)
