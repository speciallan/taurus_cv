#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
骨干网络用作特征提取器
"""

from taurus_cv.models.resnet.resnet import resnet50


def feature_extractor(input, is_extractor=True, model=None):
    """
    ResNet40特征提取器，40层是为了加速训练，节约显存
    单词训练7891M显存可以节省到6635M，epoch训练时间会从40min降到20min
    :param input:
    :return:
    """

    if not model:
        x = resnet50(input, layer_num=40, is_extractor=is_extractor)
        return x

    else:

        x = model(input, is_extractor=is_extractor)
        return x

