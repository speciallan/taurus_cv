#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras import Model, Input
from keras.layers import *


# 定义一个backbone
def ddnet(input, classes_num=2, is_extractor=False):

    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='dd_conv1')(input)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='dd_conv2')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='dd_conv3')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='dd_conv4')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='dd_conv5')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D()(x)

    # x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)

    if is_extractor:
        return x
    else:
        # x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu', kernel_initializer='normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_initializer='normal')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes_num, activation='softmax', kernel_initializer='normal')(x)
        model = Model(input, x)

        return model

