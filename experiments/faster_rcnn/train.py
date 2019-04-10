#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import argparse
sys.path.append('../..')

from models.faster_rcnn.train import *
from utils.spe import *

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--stages", type=str, nargs='+', default=['rcnn'], help="stages: rpn、rcnn")
    parse.add_argument("--epochs", type=int, default=50, help="epochs")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--init_weight_path", type=str, default=None, help="weight path")
    parse.add_argument("--init_epochs", type=int, default=0, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    train(argments)

