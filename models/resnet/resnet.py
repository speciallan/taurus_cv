#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
ResNet实现
"""

from keras import layers, Model, backend
from taurus_cv.models.base_model import BaseModel


def resnet18(input, classes_num=1000, is_extractor=False):
    """
    ResNet18
    :param input: 输入Keras.Input
    :param classes_num: 分类数量
    :param is_extractor: 是否用作特征提取器
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    # conv2 [64,64]*2
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')

    # 确定fine-turning层
    no_train_model = Model(inputs=input, outputs=x)

    # conv3 [128,128]*2
    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')

    # conv4 [256,256]*2
    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')

    # conv5 [512,512]*2
    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = True
            else:
                l.trainable = False

        return x

    # 完整CNN模型
    else:

        dropout = layers.Dropout(0.5)(x)
        preds = layers.Dense(classes_num, activation='softmax')(dropout)

        return Model(input, preds, name='resnet18')


def resnet34(input, classes_num=1000, is_extractor=False):
    """
    ResNet34
    :param input:
    :param classes_num:
    :param is_extractor:
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    # conv2 [64,64]*3
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block='b')
    x = identity_block(x, 3, [64, 64], stage=2, block='c')

    # 确定fine-turning层
    no_train_model = Model(inputs=input, outputs=x)

    # conv3 [128,128]*4
    x = conv_block(x, 3, [128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128], stage=3, block='b')
    x = identity_block(x, 3, [128, 128], stage=3, block='c')
    x = identity_block(x, 3, [128, 128], stage=3, block='d')

    # conv4 [256,256]*6
    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')
    x = identity_block(x, 3, [256, 256], stage=4, block='c')
    x = identity_block(x, 3, [256, 256], stage=4, block='d')
    x = identity_block(x, 3, [256, 256], stage=4, block='e')
    x = identity_block(x, 3, [256, 256], stage=4, block='f')

    # conv5 [512,512]*3
    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')
    x = identity_block(x, 3, [512, 512], stage=5, block='c')

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = True
            else:
                l.trainable = False

        return x

    # 完整CNN模型
    else:

        dropout = layers.Dropout(0.5)(x)
        preds = layers.Dense(classes_num, activation='softmax')(dropout)

        return Model(input, preds, name='resnet34')

def resnet50(input, classes_num=1000, layer_num=50, is_extractor=False, output_layer_name = None, is_transfer_learning=False):
    """
    ResNet50
    :param input: 输入Keras.Input
    :param is_extractor: 是否用于特征提取
    :param layer_num: 可选40、50，40用于训练frcnn的时候速度过慢的问题
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    # 池化
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # conv2 [64,64,256]*3
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv3 [128,128,512]*4
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv4 [256,256,1024]*6
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # conv5 [512,512,2048]*3
    if layer_num == 50:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 确定fine-turning层
    no_train_model = Model(inputs=input, outputs=x)

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            l.trainable = False

            # if isinstance(l, layers.BatchNormalization):
            #     l.trainable = True
            # else:
            #     l.trainable = False

        if output_layer_name:
            return no_train_model.get_layer(output_layer_name).output

        return x

    elif is_transfer_learning:

        x = layers.AveragePooling2D()(x)
        x = layers.Flatten()(x)

        preds = layers.Dense(units=classes_num, activation='softmax', kernel_initializer='he_normal')(x)

        model = Model(input, preds, name='resnet50')

        # 3 4 6 3=16 * 3 前2个block 21层冻结
        for layer in model.layers[:21]:
            layer.trainable = False

        return model


    # 完整CNN模型
    else:

        # x = layers.MaxPooling2D(pool_size=(7, 7))(x)
        x = layers.AveragePooling2D()(x)
        x = layers.Flatten()(x)

        preds = layers.Dense(units=classes_num, activation='softmax', kernel_initializer='he_normal')(x)

        return Model(input, preds, name='resnet50')


def resnet50_fpn(input, classes_num=1000, layer_num=50, is_extractor=False, output_layer_name = None, is_transfer_learning=False):
    """
    ResNet50 with FPN
    :param input: 输入Keras.Input
    :param is_extractor: 是否用于特征提取
    :param layer_num: 可选40、50，40用于训练frcnn的时候速度过慢的问题
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    c1 = x

    # 池化
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv2 [64,64,256]*3
    c2 = x
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv3 [128,128,512]*4
    c3 = x
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv4 [256,256,1024]*6
    c4 = x
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # conv5 [512,512,2048]*3
    c5 = x
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    #FPN
    top_down_pyramid_size = 256

    P5 = layers.Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c5p5')(c5)

    P4 = layers.Add(name="fpn_p4add")([layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                       layers.Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c4p4')(c4)])

    P3 = layers.Add(name="fpn_p3add")([layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                       layers.Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c3p3')(c3)])

    P2 = layers.Add(name="fpn_p2add")([layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                       layers.Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c2p2')(c2)])

    P2 = layers.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = layers.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = layers.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # P5 = layers.Conv2D(top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p5")(P5)

    P6 = layers.Conv2D(top_down_pyramid_size, (3, 3), strides=2, padding='same', name='fpn_p6')(c5)
    # P6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # print(c2,c3,c4,c5)
    # print(P2,P3,P4,P5,P6)
    # exit()

    if is_extractor:
        return Model(input, [P2, P3, P4, P5, P6], name='resnet50_fpn')
    else:
        return False


def resnet101(input, classes_num=1000, is_extractor=False):
    """
    ResNet101
    :param input: 输入Keras.Input
    :param is_extractor: 是否用于特征提取
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    # conv2 [64,64,256]*3
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 确定fine-turning层
    no_train_model = Model(inputs=input, outputs=x)

    # conv3 [128,128,512]*4
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv4 [256,256,1024]*23
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

    for i in range(22):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='')

    # conv5 [512,512,2048]*3
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = True
            else:
                l.trainable = False

        return x

    # 完整CNN模型
    else:

        dropout = layers.Dropout(0.5)(x)
        preds = layers.Dense(classes_num, activation='softmax')(dropout)

        return Model(input, preds, name='resnet101')


def resnet152(input, classes_num=1000, is_extractor=False):
    """
    ResNet152
    :param input: 输入Keras.Input
    :param is_extractor: 是否用于特征提取
    :return:
    """

    bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input)

    # conv1 [64]*1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    # conv2 [64,64,256]*3
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 确定fine-turning层
    no_train_model = Model(inputs=input, outputs=x)

    # conv3 [128,128,512]*8
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='e')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='f')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='g')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='h')

    # conv4 [256,256,1024]*36
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

    for i in range(35):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='')

    # conv5 [512,512,2048]*3
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 用作特征提取器做迁移学习
    if is_extractor:

        # 冻结参数，停止学习
        for l in no_train_model.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = True
            else:
                l.trainable = False

        return x

    # 完整CNN模型
    else:

        dropout = layers.Dropout(0.5)(x)
        preds = layers.Dense(classes_num, activation='softmax')(dropout)

        return Model(input, preds, name='resnet152')


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    残差连接
    :param input_tensor: 输入张量
    :param kernel_size: 卷积核大小
    :param filters: 卷积核个数
    :param stage: 阶段标记
    :param block: 生成层名字
    :return: Tensor
    """

    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """
    膨胀卷积

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)

    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


if __name__ == '__main__':

    input = layers.Input(shape=(224,224,3))
    model = resnet50(input)
    print(model.summary())
