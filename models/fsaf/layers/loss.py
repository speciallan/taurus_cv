#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import keras
import tensorflow as tf

from taurus_cv.models.retinanet.model import tensorflow_backend

# 自定义损失，字典，总损失是每个loss之和
def get_loss():

    return {
        'regression_ab': smooth_l1(),
        'classification_ab': focal(),
        'regression_af': smooth_l1(),
        'classification_af': focal()
    }


def focal(alpha=0.25, gamma=2.0):
    """
    Focal Loss
    :param alpha:
    :param gamma:
    :return:
    """
    def _focal(y_true, y_pred):

        labels         = y_true
        classification = y_pred

        # filter out "ignore" anchors
        anchor_state   = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices        = tensorflow_backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = tensorflow_backend.gather_nd(labels, indices)
        classification = tensorflow_backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tensorflow_backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tensorflow_backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tensorflow_backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):

    sigma_squared = sigma ** 2

    # (1,5,4) (1,?,4)
    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]

        # filter out "ignore" anchors 只取anchor_start==1的
        indices           = tensorflow_backend.where(keras.backend.equal(anchor_state, 1))
        regression        = tensorflow_backend.gather_nd(regression, indices)
        regression_target = tensorflow_backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tensorflow_backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
