#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
训练观测值
"""

from taurus_cv.models.faster_rcnn.layers import network


def add_rpn_observer(model):
    """
    增加rpn训练观察者
    :param model:
    :return:
    """

    # 增加观测值
    layer = model.get_layer('rpn_target')
    metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
    network.add_metrics(model, metric_names, layer.output[-4:])

    return model


def add_rcnn_observer(model):
    """
    增加rcnn训练观察者
    :param model:
    :return:
    """

    # 增加观测值 rpn_target的倒数四层
    layer = model.get_layer('rpn_target')
    metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
    network.add_metrics(model, metric_names, layer.output[-4:])

    # rcnn最后的检测框
    layer = model.get_layer('rcnn_target')
    metric_names = ['rcnn_miss_match_gt_num']
    network.add_metrics(model, metric_names, layer.output[-1:])

    return model
