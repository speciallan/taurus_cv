#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../..')

from models.faster_rcnn.inference import *
from utils.spe import *

if __name__ == '__main__':

    # 预测结果输出到当前目录
    inference(output_dir='./predicted_images')

