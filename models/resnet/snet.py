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



