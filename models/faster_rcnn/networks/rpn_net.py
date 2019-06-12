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

    # 特征是TensorList，走fpn
    if isinstance(features, list):

        P2, P3, P4, P5, P6 = features

        top_down_pyramid_size = 256

        # 每一层128个通道 一共512
        head1 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head1_conv")(P2)
        head1 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head1_conv_2")(head1)

        head2 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head2_conv")(P3)
        head2 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head2_conv_2")(head2)

        head3 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head3_conv")(P4)
        head3 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head3_conv_2")(head3)

        head4 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head4_conv")(P5)
        head4 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head4_conv_2")(head4)

        head5 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head4_conv")(P6)
        head5 = layers.Conv2D(top_down_pyramid_size // 2, (3, 3), padding="SAME", name="head4_conv_2")(head5)

        # f_p1 = layers.UpSampling2D(size=(16, 16), name="pre_cat_1")(head5)
        f_p2 = layers.UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
        f_p3 = layers.UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
        f_p4 = layers.UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
        f_p5 = head1

        x = layers.Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])

        # print(f_p2,f_p3,f_p4,f_p5,x)
        # exit()

    else:

        # 3x3卷积
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv')(features)

    # 再接1x1x512卷积，再展平，作二分类，是否是roi
    x_class = Conv2D(num_anchors * 2, (1, 1), kernel_initializer='uniform', activation='linear', name='rpn_class_logits')(x)
    x_class = Reshape((-1, 2))(x_class)

    # 再接1x1x1024卷积，作回归，4个点
    x_regr = Conv2D(num_anchors * 4, (1, 1), kernel_initializer='normal', name='rpn_deltas')(x)
    x_regr = Reshape((-1, 4))(x_regr)

    # print(x_regr, x_class)
    # exit()

    return x_regr, x_class

