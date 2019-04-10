#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from functools import partial

# 卷积偏函数 3×3卷积核 步长1
conv3 = partial(layers.Conv2D, kernel_size=3, strides=1, padding='same', activation='relu')

# 卷积复用：张量、卷积核、卷积层数
def block(in_tensor, filters, n_convs):

    conv_block = in_tensor

    for _ in range(n_convs):
        conv_block = conv3(filters=filters)(conv_block)

    return conv_block

# 16层
def vgg(in_shape=(227,227,3), n_classes=1000, opt='sgd', n_stages_per_blocks=[2, 2, 3, 3, 3]):

    in_layer = layers.Input(in_shape)

    # 2 227*227*64
    block1 = block(in_layer, 64, n_stages_per_blocks[0])
    pool1 = layers.MaxPool2D()(block1)

    # 2 112*112*128
    block2 = block(pool1, 128, n_stages_per_blocks[1])
    pool2 = layers.MaxPool2D()(block2)

    # 3 56*56*256
    block3 = block(pool2, 256, n_stages_per_blocks[2])
    pool3 = layers.MaxPool2D()(block3)

    # 3 28*28*512
    block4 = block(pool3, 512, n_stages_per_blocks[3])
    pool4 = layers.MaxPool2D()(block4)

    # 3 14*14*512
    block5 = block(pool4, 512, n_stages_per_blocks[4])
    pool5 = layers.MaxPool2D()(block5)

    # 平均池化
    flattened = layers.GlobalAvgPool2D()(pool5)

    # 4097
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    dense2 = layers.Dense(4096, activation='relu')(dense1)

    # 1000
    preds = layers.Dense(n_classes, activation='softmax')(dense2)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model

# 16层
def vgg16(in_shape=(227,227,3), n_classes=1000, opt='sgd'):
    return vgg(in_shape, n_classes, opt)

# 19层
def vgg19(in_shape=(227,227,3), n_classes=1000, opt='sgd'):
    return vgg(in_shape, n_classes, opt, [2, 2, 4, 4, 4])

if __name__ == '__main__':

    model = vgg19()
    print(model.summary())
