#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
GoogleNet实现，包括inception模块
"""

from keras import layers
from keras.models import Model
from functools import partial


# 1*1卷积
conv1x1 = partial(layers.Conv2D, kernel_size=1, activation='relu')

# 3*3卷积
conv3x3 = partial(layers.Conv2D, kernel_size=3, padding='same', activation='relu')

# 5*5卷积
conv5x5 = partial(layers.Conv2D, kernel_size=5, padding='same', activation='relu')

# inception
def inception_module(in_tensor, c1, c3_1, c3, c5_1, c5, pp):

    # 一个1*1卷积
    conv1 = conv1x1(c1)(in_tensor)

    # 1*1 + 3*3
    conv3_1 = conv1x1(c3_1)(in_tensor)
    conv3 = conv3x3(c3)(conv3_1)

    # 5*5
    conv5_1 = conv1x1(c5_1)(in_tensor)
    conv5 = conv5x5(c5)(conv5_1)

    # 1*1 + 最大池化
    pool_conv = conv1x1(pp)(in_tensor)
    pool = layers.MaxPool2D(3, strides=1, padding='same')(pool_conv)

    # 合并
    merged = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])

    return merged

# auxiliary
def aux_clf(in_tensor):

    avg_pool = layers.AvgPool2D(5, 3)(in_tensor)

    conv = conv1x1(128)(avg_pool)

    flattened = layers.Flatten()(conv)

    dense = layers.Dense(1024, activation='relu')(flattened)
    dropout = layers.Dropout(0.7)(dense)

    out = layers.Dense(1000, activation='softmax')(dropout)

    return out

def googlenet(in_shape=(224,224,3), n_classes=1000, opt='sgd'):

    in_layer = layers.Input(in_shape)

    conv1 = layers.Conv2D(64, 7, strides=2, activation='relu', padding='same')(in_layer)
    pad1 = layers.ZeroPadding2D()(conv1)
    pool1 = layers.MaxPool2D(3, 2)(pad1)

    conv2_1 = conv1x1(64)(pool1)
    conv2_2 = conv3x3(192)(conv2_1)
    pad2 = layers.ZeroPadding2D()(conv2_2)
    pool2 = layers.MaxPool2D(3, 2)(pad2)

    inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32)
    inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64)
    pad3 = layers.ZeroPadding2D()(inception3b)
    pool3 = layers.MaxPool2D(3, 2)(pad3)

    inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64)
    inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64)
    inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64)
    inception4d = inception_module(inception4c, 112, 144, 288, 32, 48, 64)
    inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128)
    pad4 = layers.ZeroPadding2D()(inception4e)
    pool4 = layers.MaxPool2D(3, 2)(pad4)

    aux_clf1 = aux_clf(inception4a)
    aux_clf2 = aux_clf(inception4d)

    inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128)
    inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)
    pad5 = layers.ZeroPadding2D()(inception5b)
    pool5 = layers.MaxPool2D(3, 2)(pad5)

    avg_pool = layers.GlobalAvgPool2D()(pool5)
    dropout = layers.Dropout(0.4)(avg_pool)

    preds = layers.Dense(1000, activation='softmax')(dropout)

    model = Model(in_layer, [preds, aux_clf1, aux_clf2])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

if __name__ == '__main__':

    model = googlenet()
    print(model.summary())
