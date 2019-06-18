#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras import Model, Input
from keras.layers import *

def snet(input, classes_num=2, is_extractor=False, output_layer_name='snet_pool'):

    x = Conv2D(16, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv1')(input)
    x = Conv2D(16, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv2')(x)
    x = Conv2D(16, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv3')(x)
    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv4')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv5')(x)
    x = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv6')(x)
    x = MaxPooling2D((2, 2), strides=1, name='snet_pool')(x)

    if is_extractor:
        return x

    else:

        # x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(classes_num, activation='softmax', kernel_initializer='normal')(x)
        model = Model(input, x)

        return model


def snetplus(input, classes_num=2, is_extractor=False, output_layer_name='snet_conv'):

    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv1_1')(input)
    x = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv1_2')(x)
    x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv1_3')(x)
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv1_4')(x)
    c1 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv1_5')(x)

    x = Conv2D(32, (5, 5), strides=2, activation='relu', padding='same', name='snet_conv2_1')(input)
    x = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv2_2')(x)
    x = Conv2D(64, (5, 5), strides=2, activation='relu', padding='same', name='snet_conv2_3')(x)
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv2_4')(x)
    c2 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv2_5')(x)

    x = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv3_1')(input)
    x = Conv2D(32, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=2, name='snet_pool')(x)
    c3 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv3_3')(x)

    x = Concatenate(axis=-1)([c1, c2, c3])
    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv')(x)

    if is_extractor:
        return x

    else:

        # x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(classes_num, activation='softmax', kernel_initializer='normal')(x)
        model = Model(input, x)

        return model

