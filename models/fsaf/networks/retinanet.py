#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import tensorflow as tf

import keras
from keras.layers import Input

from taurus_cv.models.resnet.resnet import resnet50_fpn
from taurus_cv.models.retinanet.model.misc import UpsampleLike, RegressBoxes, NonMaximumSuppression, Anchors


def retinanet(config, stage = 'train'):

    batch_size = config.IMAGES_PER_GPU

    input_images = Input(shape=config.IMAGE_INPUT_SHAPE)
    gt_class_ids = Input(shape=(config.MAX_GT_INSTANCES, config.NUM_CLASSES))
    gt_boxes = Input(shape=(config.MAX_GT_INSTANCES, 4 + 1))

    fpn_model = resnet50_fpn(input_images, config.NUM_CLASSES, is_extractor=True)
    features = fpn_model.outputs

    # anchor参数
    anchor_parameters = AnchorParameters.default
    submodels = default_submodels(config.NUM_CLASSES, anchor_parameters)

    pyramid = __build_pyramid(submodels, features)
    anchors = __build_anchors(anchor_parameters, features)

    outputs = [anchors] + pyramid
    # return keras.models.Model(inputs=input_images, outputs=[anchors] + pyramid, name='retinanet')

    # [batch_size, ?, 4]
    anchors = outputs[0]
    regression = outputs[1]
    classification = outputs[2]

    boxes = RegressBoxes(name='boxes')([anchors, regression])
    detections = keras.layers.Concatenate(axis=2)([boxes, classification])

    # NMS
    detections = NonMaximumSuppression(name='nms', nms_threshold=0.5)([boxes, classification, detections])

    return keras.models.Model(inputs=input_images, outputs=[regression, classification, detections], name='retinanet')


def default_submodels(num_classes, anchor_parameters):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]


def __build_model_pyramid(name, model, features):
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    return [__build_model_pyramid(name, model, features) for name, model in models]


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
    # sizes=[32, 64, 128, 256, 512],
    # strides=[8, 16, 32, 64, 128],
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

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=keras.initializers.he_normal(),
        name='pyramid_classification',
        **options
    )(outputs)

    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


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

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)