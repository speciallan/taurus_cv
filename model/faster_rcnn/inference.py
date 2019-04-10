#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

# faster_rcnn推理

import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from faster_rcnn.preprocess.input import VocDataset
from faster_rcnn.utils.image import load_image_gt
from faster_rcnn.config import current_config as config
from faster_rcnn.utils import visualize, np_utils
from faster_rcnn.layers import models
