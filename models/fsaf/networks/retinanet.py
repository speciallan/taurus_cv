#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import tensorflow as tf

import keras
from keras.layers import Input

from taurus_cv.models.resnet.resnet import resnet50_fpn
from taurus_cv.models.retinanet.model.misc import RegressBoxes, NonMaximumSuppression, Anchors, UpsampleLike
from taurus_cv.models.fsaf.layers.initializers import PriorProbability
from taurus_cv.utils.spe import spe


def retinanet(config, stage = 'train'):

    input_images = Input(shape=config.IMAGE_INPUT_SHAPE)

    import keras_resnet.models
    backbone = keras_resnet.models.ResNet50(input_images, include_top=False, freeze_bn=True)
    # backbone.load_weights('/home/speciallan/.keras/models/ResNet-50-model.keras.h5', by_name=True, skip_mismatch=True)
    # for l in backbone.layers:
    #     if isinstance(l, keras.layers.BatchNormalization):
    #         l.trainable = False
    c2, c3, c4, c5 = backbone.outputs[0:]
    features = __create_pyramid_features(c2, c3, c4, c5)
    # print(c2,c3,c4,c5,'\n')
    # print(features)

    # backbone = resnet50_fpn(input_images, config.NUM_CLASSES, is_extractor=True)
    # backbone.load_weights(config.pretrained_weights, by_name=True)
    # features = backbone.outputs
    # spe(features)

    # anchor参数
    anchor_parameters = AnchorParameters.default
    submodels = default_submodels(config.NUM_CLASSES, anchor_parameters)

    # 生成FPN所有子模型并连接
    pyramid = __build_pyramid(submodels, features)

    # 生成FPN所有层不同尺度anchor的集合 (1,?,4)
    anchors = __build_anchors(anchor_parameters, features)

    # [batch_size, ?, 4] [b, ?, 8]
    regression_ab, classification_ab = pyramid[:2]
    regression_af, classification_af = pyramid[2:4]
    # spe(regression_ab, classification_ab, regression_af, classification_af)

    boxes = RegressBoxes(name='boxes')([anchors, regression_ab])
    detections_ab = keras.layers.Concatenate(axis=2)([boxes, classification_ab])

    # detections_af = keras.layers.Concatenate(axis=2)([regression_af, classification_af])

    # detections = np.append(detections_ab, detections_af, axis=1)
    # spe(detections)

    # NMS_ab
    detections_ab = NonMaximumSuppression(name='nms_ab', nms_threshold=0.05)([boxes, classification_ab, detections_ab])

    # NMS 总检测
    # detections = NonMaximumSuppression(name='nms', nms_threshold=0.05)([detections_ab, detections_af])

    return keras.models.Model(inputs=input_images, outputs=pyramid + [detections_ab], name='retinanet')


def __create_pyramid_features(C2, C3, C4, C5, feature_size=256):

    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])

    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, C2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P2, P3, P4, P5, P6


def default_submodels(num_classes, anchor_parameters):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]


# def __build_model_pyramid(name, model, features):
    # FPN每一层的子模型输出连接
    # return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):

    # regression: reg_model
    # classification: cls_model
    cls_ab, cls_af, reg_ab, reg_af = [], [], [], []

    for f in features:
        for name, sub_model in models:
            # spe(model.inputs, model.outputs)
            outputs = sub_model(f)

            if name == 'classification':
                cls_ab.append(outputs[0])
                cls_af.append(outputs[1])

            if name == 'regression':
                reg_ab.append(outputs[0])
                reg_af.append(outputs[1])

    # spe(reg_ab, '\n', cls_ab,'\n',  reg_af,'\n',  cls_af)

    # reg_ab cls_ab reg_af cls_af
    return [keras.layers.Concatenate(axis=1, name='regression_ab')(reg_ab),
            keras.layers.Concatenate(axis=1, name='classification_ab')(cls_ab),
            keras.layers.Concatenate(axis=1, name='regression_af')(reg_af),
            keras.layers.Concatenate(axis=1, name='classification_af')(cls_af)]

    # return [__build_model_pyramid(name, model, features) for name, model in models]


def __build_anchors(anchor_parameters, features):
    anchors = []
    for i, f in enumerate(features):
        anchors.append(Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f))

    # 将每一层不同尺度anchors拼到一起
    return keras.layers.Concatenate(axis=1)(anchors)


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    # sizes=[8, 16, 32, 64, 128, 256],
    # strides=[2, 4, 8, 16, 32, 64],
    # sizes是步长的4倍
    sizes=[16, 32, 64, 128, 256],
    strides=[4, 8, 16, 32, 64],
    ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def default_classification_model(num_classes,
                                 num_anchors,
                                 pyramid_feature_size=256,
                                 prior_probability=0.01,
                                 classification_feature_size=256,
                                 name='classification_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs

    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    # anchor-based
    outputs_ab = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification_ab',
        **options
    )(outputs)

    # A * K
    outputs_ab = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape_ab')(outputs_ab)
    outputs_ab = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid_ab')(outputs_ab)

    # anchor-free
    outputs_af = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification_af',
        **options
    )(outputs)

    outputs_af = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape_af')(outputs_af)
    outputs_af = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid_af')(outputs_af)

    return keras.models.Model(inputs=inputs, outputs=[outputs_ab, outputs_af], name=name)


def default_regression_model(num_anchors,
                             pyramid_feature_size=256,
                             regression_feature_size=256,
                             name='regression_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs

    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs_ab = outputs
    outputs_af = outputs

    # anchor-based
    outputs_ab = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression_ab', **options)(outputs_ab)
    outputs_ab = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape_ab')(outputs_ab)

    # anchor-free
    outputs_af = keras.layers.Conv2D(4, name='pyramid_regression_af', **options)(outputs_af)
    outputs_af = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape_af')(outputs_af)

    return keras.models.Model(inputs=inputs, outputs=[outputs_ab, outputs_af], name=name)