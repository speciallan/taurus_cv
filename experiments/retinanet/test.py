#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

all_img = os.listdir('../../../../data/VOCdevkit/dd/JPEGImages')

f = open('../../../../data/VOCdevkit/dd/ImageSets/Main/all.txt', 'w')

for k,v in enumerate(all_img):
    f.write('{}\n'.format(v.replace('.jpg', '')))

f.close()
print(all_img)