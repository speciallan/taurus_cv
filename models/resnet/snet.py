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


def snettz(input, classes_num=2, is_extractor=False, multi_outputs=False, output_layer_name='snet_pool'):

    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv1_1')(input)
    # x = ConvOffset2D(32, name='conv1_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    c1 = x

    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv2_1')(x)
    # x = ConvOffset2D(64, name='conv2_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    c2 = x

    x = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv3_1')(x)
    # x = ConvOffset2D(128, name='conv3_offset')(x)
    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    c3 = x

    x = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv4_1')(x)
    # x = ConvOffset2D(256, name='conv4_offset')(x)
    # x = Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', name='snet_conv4_2')(x)
    # x = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv4_3')(x)
    # x = BatchNormalization(name='conv4_bn')(x)
    # x = AveragePooling2D((2, 2), name='snet_pool4')(x)
    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    c4 = x

    if is_extractor:
        if multi_outputs:
            outputs = [c1, c2, c3, c4]
        else:
            outputs = c4
        model = Model(input, outputs=outputs)
        return model

def snetplus(input, classes_num=2, is_extractor=False, multi_outputs=False, output_layer_name='snet_pool'):

    # 第一层尺寸大
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_a1')(input)
    x = MaxPooling2D((2, 2), name='snet_pool1')(x)
    conv1 = x

    b1 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b1')(x)
    b2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    b2 = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_b2')(b2)
    b3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_b3')(x)
    # b3 = ConvOffset2D(64, name='conv2_offset')(b3)
    x = Concatenate(axis=-1)([b1, b2, b3])
    x = Conv2D(64, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool2')(x)
    conv2 = x

    c1 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c1')(x)
    c2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    c2 = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_c2')(c2)
    c3 = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_c3')(x)
    # c3 = ConvOffset2D(128, name='conv3_offset')(c3)
    x = Concatenate(axis=-1)([c1, c2, c3])
    x = Conv2D(128, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool3')(x)
    conv3 = x

    d1 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d1')(x)
    d2 = AveragePooling2D(strides=(1, 1), padding='same')(x)
    d2 = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same', name='snet_conv_d2')(d2)
    d3 = Conv2D(256, (3, 3), strides=1, activation='relu', padding='same', name='snet_conv_d3')(x)
    # d3 = ConvOffset2D(256, name='conv4_offset')(d3)
    x = Concatenate(axis=-1)([d1, d2, d3])
    x = Conv2D(256, (1, 1), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='snet_pool4')(x)
    conv4 = x

    if is_extractor:
        if multi_outputs:
            outputs = [conv1, conv2, conv3, conv4]
        else:
            outputs = conv4
        model = Model(input, outputs=outputs)
        model.summary()
        return model
