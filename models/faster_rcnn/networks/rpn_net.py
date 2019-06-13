#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras import layers
from keras.layers import Conv2D, Reshape, Concatenate, Add


def rpn(features, num_anchors):
    """
    RPN基础网络
    :param base_layers:
    :param num_anchors:
    :return:
    """

    # 3x3卷积
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(features)

    # 再接1x1x512卷积，再展平，作二分类，是否是roi
    x_class = Conv2D(num_anchors * 2, (1, 1), kernel_initializer='uniform', activation='linear', name='rpn_class_logits')(x)
    x_class = Reshape((-1, 2))(x_class)

    # 再接1x1x1024卷积，作回归，4个点
    x_regr = Conv2D(num_anchors * 4, (1, 1), kernel_initializer='normal', name='rpn_deltas')(x)
    x_regr = Reshape((-1, 4))(x_regr)

    return x_regr, x_class
