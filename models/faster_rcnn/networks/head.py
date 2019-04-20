#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
RoiHead
"""

import tensorflow as tf
from keras import backend
from keras.layers import TimeDistributed, Conv2D, BatchNormalization, Activation, Lambda, Dense, Reshape
from ..layers.roi_align import RoiAlign


def roi_head(base_layers, rois, num_classes, image_max_dim, pool_size=(7, 7), fc_layers_size=1024):

    # 候选框投影到特征图
    x = RoiAlign(image_max_dim)([base_layers, rois])  #

    # 用卷积来实现两个全连接
    x = TimeDistributed(Conv2D(fc_layers_size, pool_size, padding='valid'), name='rcnn_fc1')(x)  # 变为(batch_size,roi_num,1,1,channels)
    x = TimeDistributed(BatchNormalization(), name='rcnn_class_bn1')(x)
    x = Activation(activation='relu')(x)

    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1), padding='valid'), name='rcnn_fc2')(x)
    x = TimeDistributed(BatchNormalization(), name='rcnn_class_bn2')(x)
    x = Activation(activation='relu')(x)

    # 收缩维度
    shared_layer = Lambda(lambda a: tf.squeeze(tf.squeeze(a, 3), 2))(x)  # 变为(batch_size,roi_num,channels)

    # 分类
    class_logits = TimeDistributed(Dense(num_classes, activation='linear'), name='rcnn_class_logits')(shared_layer)

    # 回归(类别相关)
    deltas = TimeDistributed(Dense(4 * num_classes, activation='linear'), name='rcnn_deltas')(shared_layer)  # shape (batch_size,roi_num,4*num_classes)

    # 变为(batch_size,roi_num,num_classes,4)
    roi_num = backend.int_shape(deltas)[1]
    deltas = Reshape((roi_num, num_classes, 4))(deltas)

    return deltas, class_logits

