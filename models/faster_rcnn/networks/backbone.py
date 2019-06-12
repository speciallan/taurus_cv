#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
骨干网络用作特征提取器
"""

from taurus_cv.models.resnet.resnet import resnet50, resnet50_fpn


def feature_extractor(input, is_extractor=True, model=None, output_layer_name=None):

    """
    ResNet40特征提取器，40层是为了加速训练，节约显存
    单词训练7891M显存可以节省到6635M，epoch训练时间会从40min降到20min
    :param input:
    :return:
    """

    if not model:
        x = resnet50(input,
                     layer_num=40,
                     is_extractor=is_extractor,
                     output_layer_name=output_layer_name)
        return x

    else:

        x = model(input,
                  is_extractor=is_extractor,
                  output_layer_name=output_layer_name)
        return x


def feature_extractor_with_fpn(input, is_extractor=True, model=None, output_layer_name=None):

    if not model:
        x = resnet50_fpn(input,
                         layer_num=50,
                         is_extractor=is_extractor,
                         output_layer_name=output_layer_name)
        return x

