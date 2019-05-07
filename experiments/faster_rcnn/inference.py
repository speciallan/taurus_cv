#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import argparse
sys.path.append('../../..')

from taurus_cv.models.faster_rcnn.inference import inference, inference_rpn
from taurus_cv.models.faster_rcnn.config import current_config as config
from taurus_cv.utils.spe import *

if __name__ == '__main__':

    # 预测结果输出到当前目录
    parse = argparse.ArgumentParser()
    parse.add_argument("--stages", type=str, nargs='+', default=['rcnn'], help="stages: rpn、rcnn")
    argments = parse.parse_args(sys.argv[1:])

    if 'rpn' in argments.stages:
        inference_rpn(config, output_dir='./predicted_images')
    else:
        inference(config, output_dir='./predicted_images')

