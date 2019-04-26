#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../../..')

from taurus_cv.models.faster_rcnn.evaluate import *
from taurus_cv.utils.spe import *

if __name__ == '__main__':

    # 预测结果输出到当前目录
    parse = argparse.ArgumentParser()
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])

    # 执行评估
    evaluate(argments, image_num=4000)

