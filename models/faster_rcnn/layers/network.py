#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
组合网络 包括rpn和faster_rcnn网络
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Lambda

from taurus_cv.models.faster_rcnn.networks.backbone import feature_extractor
from taurus_cv.models.faster_rcnn.networks.rpn_net import rpn
from taurus_cv.models.faster_rcnn.networks.head import roi_head
from taurus_cv.models.faster_rcnn.layers.anchors import Anchor
from taurus_cv.models.faster_rcnn.layers.target import RpnTarget, DetectTarget
from taurus_cv.models.faster_rcnn.layers.proposals import RpnToProposal
from taurus_cv.models.faster_rcnn.layers.losses import rpn_cls_loss, rpn_regress_loss, detect_regress_loss, detect_cls_loss
from taurus_cv.models.faster_rcnn.layers.specific_to_agnostic import deal_delta
from taurus_cv.models.faster_rcnn.layers.detect_boxes import ProposalToDetectBox
from taurus_cv.models.faster_rcnn.layers.clip_boxes import ClipBoxes, UniqueClipBoxes
from taurus_cv.utils.spe import spe


def rpn_net(config, stage='train', backbone=None):
    """
    单独训练rpn
    :param config:
    :param stage:
    :return:
    """

    batch_size = config.IMAGES_PER_GPU

    # 图片尺寸
    input_image = Input(batch_shape=(batch_size,) + config.IMAGE_INPUT_SHAPE)

    # 二分类
    input_class_ids = Input(batch_shape=(batch_size, config.MAX_GT_INSTANCES, 1 + 1))

    # 回归框4个坐标和边框总数N
    input_boxes = Input(batch_shape=(batch_size, config.MAX_GT_INSTANCES, 4 + 1))

    """
    input_image_meta包括以下，一共12个值
    :param image_id:
    :param original_image_shape: 原始图像形状，tuple(H,W,3)
    :param image_shape: 缩放后图像形状tuple(H,W,3)
    :param window: 原始图像在缩放图像上的窗口位置（y1,x1,y2,x2)
    :param scale: 缩放因子
    """
    input_image_meta = Input(batch_shape=(batch_size, 12))

    # 特征及预测结果 (1,32,32,1024)
    features = feature_extractor(input_image, model=backbone, output_layer_name=config.backbone_output_layer_name)

    # 定义rpn网络 得到分类和回归值
    boxes_regress, class_logits = rpn(features, config.RPN_ANCHOR_NUM)

    # 生成anchor (1,6144,4) 1是batch_size, 6144是anchors数量
    # 占满了（512，512）
    anchors = Anchor(config.RPN_ANCHOR_BASE_SIZE,
                     config.RPN_ANCHOR_RATIOS,
                     config.RPN_ANCHOR_SCALES,
                     config.BACKBONE_STRIDE, name='gen_anchors')(features)

    # 裁剪到窗口内 得到512x512内的anchors
    anchors = UniqueClipBoxes(config.IMAGE_INPUT_SHAPE, name='clip_anchors')(anchors)

    # windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # anchors = ClipBoxes()([anchors, windows])

    if stage == 'train':

        # 生成分类和回归目标 deltas为处理后的box
        rpn_targets = RpnTarget(batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')([input_boxes, input_class_ids, anchors])  # [deltas,cls_ids,indices,..]

        # gt_boxs,gt_cls_ids,anchors
        deltas, cls_ids, anchor_indices = rpn_targets[:3]

        # 定义损失layer
        cls_loss = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')([class_logits, cls_ids, anchor_indices])
        regress_loss = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')([boxes_regress, deltas, anchor_indices])

        # 训练阶段，得到分类和回归损失
        return Model(inputs=[input_image, input_image_meta, input_class_ids, input_boxes], outputs=[cls_loss, regress_loss])

    else:

        # 得到rpn出来的box和anchors求iou和nms排除后的box
        # (1,?,5) (1,?,2)
        detect_boxes, class_scores, _ = RpnToProposal(batch_size,
                                                      output_box_num=config.POST_NMS_ROIS_INFERENCE,
                                                      iou_threshold=config.RPN_NMS_THRESHOLD,
                                                      name='rpn2proposals')([boxes_regress, class_logits, anchors])

        # 预测阶段，通过rpn获取候选框，得到候选框和置信度
        return Model(inputs=[input_image, input_image_meta], outputs=[detect_boxes, class_scores])


def faster_rcnn(config, stage='train', backbone=None):
    """
    Faster-rcnn网络
    :param config:
    :param stage:
    :return:
    """

    batch_size = config.IMAGES_PER_GPU

    input_image = Input(shape=config.IMAGE_INPUT_SHAPE)
    gt_class_ids = Input(shape=(config.MAX_GT_INSTANCES, 1 + 1))
    gt_boxes = Input(shape=(config.MAX_GT_INSTANCES, 4 + 1))
    input_image_meta = Input(shape=(12,))

    # 通过CNN提取特征
    features = feature_extractor(input_image, model=backbone)

    # 训练rpn 得到回归和分类分
    boxes_regress, class_logits = rpn(features, config.RPN_ANCHOR_NUM)

    # 生成基础anchors(batch_size,M*N,ANCHOR_NUM,4)
    anchors = Anchor(config.RPN_ANCHOR_BASE_SIZE,
                     config.RPN_ANCHOR_RATIOS,
                     config.RPN_ANCHOR_SCALES,
                     config.BACKBONE_STRIDE, name='gen_anchors')(features)

    # 裁剪到输入形状内
    anchors = UniqueClipBoxes(clip_box_shape=config.IMAGE_INPUT_SHAPE, name='clip_anchors')(anchors)

    # 取图片元数据的后4个数据，作为窗口
    windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # anchors = ClipBoxes()([anchors, windows])

    # 应用分类和回归生成proposal，通过NMS后保留2000个候选框
    output_box_num = config.POST_NMS_ROIS_TRAIN if stage == 'train' else config.POST_NMS_ROIS_INFERENCE

    # 通过rpn后的候选框和anchors计算iou淘汰一部分，再走NMS过滤，得到最后的候选框rois [proprosal_boxes,fg_scores,class_logits]
    proposal_boxes, _, _ = RpnToProposal(batch_size,
                                         output_box_num=output_box_num,
                                         iou_threshold=config.RPN_NMS_THRESHOLD,
                                         name='rpn2proposals')([boxes_regress, class_logits, anchors])
    # proposal裁剪到图像窗口内
    proposal_boxes_coordinate, proposal_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(proposal_boxes)

    # windows = Lambda(lambda x: x[:, 7:11])(input_image_meta)
    # 边框裁剪层，取出目标候选框处理，再放回去
    proposal_boxes_coordinate = ClipBoxes()([proposal_boxes_coordinate, windows])

    # 最后再合并tag返回
    proposal_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([proposal_boxes_coordinate, proposal_boxes_tag])

    if stage == 'train':

        # 生成分类和回归目标
        rpn_targets = RpnTarget(batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, name='rpn_target')([gt_boxes, gt_class_ids, anchors])  # [deltas,cls_ids,indices,..]

        rpn_deltas, rpn_cls_ids, anchor_indices = rpn_targets[:3]

        # 定义rpn损失layer
        cls_loss_rpn = Lambda(lambda x: rpn_cls_loss(*x), name='rpn_class_loss')([class_logits, rpn_cls_ids, anchor_indices])
        regress_loss_rpn = Lambda(lambda x: rpn_regress_loss(*x), name='rpn_bbox_loss')([boxes_regress, rpn_deltas, anchor_indices])

        # 检测网络的分类和回归目标，根据gt和iou淘汰一部分
        # IoU>=0.5的为正样本；IoU<0.5的为负样本
        roi_deltas, roi_class_ids, train_rois, _ = DetectTarget(batch_size, config.TRAIN_ROIS_PER_IMAGE, config.ROI_POSITIVE_RATIO, name='rcnn_target')([gt_boxes, gt_class_ids, proposal_boxes])

        # 检测网络 RoiHead
        rcnn_deltas, rcnn_class_logits = roi_head(features, train_rois, config.NUM_CLASSES, config.IMAGE_MIN_DIM, pool_size=(7, 7), fc_layers_size=1024)

        # 检测网络损失函数 rcnn_deltas是gt_bbox,roi_deltas是bbox
        regress_loss_rcnn = Lambda(lambda x: detect_regress_loss(*x), name='rcnn_bbox_loss')([rcnn_deltas, roi_deltas, roi_class_ids])
        cls_loss_rcnn = Lambda(lambda x: detect_cls_loss(*x), name='rcnn_class_loss')([rcnn_class_logits, roi_class_ids])

        # 端到端训练，得到4类损失
        return Model(inputs=[input_image, input_image_meta, gt_class_ids, gt_boxes],
                     outputs=[cls_loss_rpn, regress_loss_rpn, regress_loss_rcnn, cls_loss_rcnn])

    else:

        # 预测网络
        rcnn_deltas, rcnn_class_logits = roi_head(features, proposal_boxes, config.NUM_CLASSES, config.IMAGE_MIN_DIM, pool_size=(7, 7), fc_layers_size=1024)

        # 处理类别相关
        rcnn_deltas = Lambda(lambda x: deal_delta(*x), name='deal_delta')([rcnn_deltas, rcnn_class_logits])

        # 应用分类和回归生成最终检测框
        detect_boxes, class_scores, detect_class_ids, detect_class_logits = ProposalToDetectBox(
            score_threshold=0.05,
            output_box_num=100,
            name='proposals2detectboxes'
        )([rcnn_deltas, rcnn_class_logits, proposal_boxes])

        # 裁剪到窗口内部
        detect_boxes_coordinate, detect_boxes_tag = Lambda(lambda x: [x[..., :4], x[..., 4:]])(detect_boxes)
        detect_boxes_coordinate = ClipBoxes()([detect_boxes_coordinate, windows])

        # 最后再合并tag返回
        detect_boxes = Lambda(lambda x: tf.concat(x, axis=-1))([detect_boxes_coordinate, detect_boxes_tag])

        # detect_boxes (?,?,5) boxes_regress(?,?,4) proposal_boxes(1,?,5)
        # detect_boxes = boxes_regress[:,:,:4]
        # detect_boxes = proposal_boxes
        # spe(detect_boxes, boxes_regress, proposal_boxes)

        # 预测阶段，得到检测框、置信度、分类id、分类预测概率
        return Model(inputs=[input_image, input_image_meta],
                     outputs=[detect_boxes, class_scores, detect_class_ids, detect_class_logits])


def compile(keras_model, config, loss_names=[]):
    """
    编译模型，增加损失函数，L2正则化以
    :param keras_model:
    :param config:
    :param loss_names: 损失函数列表
    :return:
    """
    # 优化目标
    optimizer = keras.optimizers.SGD(lr=config.LEARNING_RATE,
                                     momentum=config.LEARNING_MOMENTUM,
                                     clipnorm=config.GRADIENT_CLIP_NORM)

    # 增加损失函数，首先清除之前的，防止重复
    keras_model._losses = []
    keras_model._per_input_losses = {}

    # 层输出加上权重，添加损失到keras
    for name in loss_names:
        layer = keras_model.get_layer(name)
        if layer is None or layer.output in keras_model.losses:
            continue
        loss = (tf.reduce_mean(layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))
        keras_model.add_loss(loss)

    # 增加L2正则化
    # 跳过批标准化层的 gamma 和 beta 权重
    reg_losses = [keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                  for w in keras_model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name]

    keras_model.add_loss(tf.add_n(reg_losses))

    # 编译
    keras_model.compile(optimizer=optimizer, loss=[None] * len(keras_model.outputs))  # 使用虚拟损失

    # 为每个损失函数增加度量
    for name in loss_names:

        if name in keras_model.metrics_names:
            continue

        layer = keras_model.get_layer(name)
        if layer is None:
            continue

        keras_model.metrics_names.append(name)

        loss = (tf.reduce_mean(layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))

        keras_model.metrics_tensors.append(loss)


def add_metrics(keras_model, metric_name_list, metric_tensor_list):
    """
    增加度量
    :param keras_model: 模型
    :param metric_name_list: 度量名称列表
    :param metric_tensor_list: 度量张量列表
    :return: 无
    """
    for name, tensor in zip(metric_name_list, metric_tensor_list):
        keras_model.metrics_names.append(name)
        keras_model.metrics_tensors.append(tf.reduce_mean(tensor, keepdims=True))




if __name__ == '__main__':

    # print(keras.backend.image_data_format())
    # models = resnet50(Input((224, 224, 3)))
    # models.summary()
    x = tf.ones(shape=(5, 4, 1, 1, 3))
    y = tf.squeeze(x, 3)
    sess = tf.Session()
    print(sess.run(y))
