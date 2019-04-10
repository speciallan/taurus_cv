#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
训练模型
"""

import os
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

from .config import current_config as config
from .io.input import VocDataset
from .utils.generator import generator
from .layers import network

def set_gpu_growth():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)
    # keras.backend.set_floatx('float16')

def get_call_back(stage):
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/frcnn-' + stage + '.{epoch:03d}.h5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=False,
                                 period=5)

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=0)

    logs = TensorBoard(log_dir=config.log_path)
    return [checkpoint, lr_reducer, logs]


def train(args):

    set_gpu_growth()
    dataset = VocDataset(config.voc_path, class_mapping=config.CLASS_MAPPING)
    dataset.prepare()
    train_img_info = [info for info in dataset.get_image_info_list() if info['type'] == 'trainval']  # 训练集
    # train_img_info = train_img_info[:100]

    print("all_img_info:{}".format(len(train_img_info)))

    # 生成器 没有做数据增强
    # gen = generator(train_img_info, config.IMAGES_PER_GPU, config.IMAGE_MAX_DIM, 50)
    gen = generator(image_info_list=train_img_info, batch_size=config.IMAGES_PER_GPU, max_output_dim=config.IMAGE_MAX_DIM, max_gt_num=100)

    # 训练rpn
    if 'rpn' in args.stages:

        model = network.rpn_net(config)
        model.load_weights(config.pretrained_weights, by_name=True)
        loss_names = ["rpn_bbox_loss", "rpn_class_loss"]
        network.compile(model, config, loss_names)

        # 增加个性化度量
        layer = model.get_layer('rpn_target')
        metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
        network.add_metrics(model, metric_names, layer.output[-4:])

        # m.summary()
        model.fit_generator(gen,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_img_info) // config.IMAGES_PER_GPU,
                        verbose=1,
                        initial_epoch=args.init_epochs,
                        callbacks=get_call_back('rpn'))

        model.save(config.rpn_weights)

    if 'rcnn' in args.stages:

        model = network.frcnn(config=config)
        # m = models.frcnn((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3), config.BATCH_SIZE, config.NUM_CLASSES, 50, config.IMAGE_MAX_DIM, config.TRAIN_ROIS_PER_IMAGE, config.ROI_POSITIVE_RATIO)

        loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss", "rcnn_class_loss"]
        network.compile(model, config, loss_names)

        # 增加个性化度量
        layer = model.get_layer('rpn_target')
        metric_names = ['gt_num', 'positive_anchor_num', 'miss_match_gt_num', 'gt_match_min_iou']
        network.add_metrics(model, metric_names, layer.output[-4:])

        layer = model.get_layer('rcnn_target')
        metric_names = ['rcnn_miss_match_gt_num']
        network.add_metrics(model, metric_names, layer.output[-1:])

        # 加载预训练模型
        if args.init_epochs > 0:
            model.load_weights(args.init_weight_path, by_name=True)
        elif os.path.exists(config.rpn_weights):  # 有rpn预训练模型就加载，没有直接加载resnet50预训练模型
            model.load_weights(config.rpn_weights, by_name=True)
        else:
            model.load_weights(config.pretrained_weights, by_name=True)

        # m.summary()
        # spe(args.epochs, len(train_img_info))

        # 训练

        model.fit_generator(gen,
                        epochs=args.epochs,
                        steps_per_epoch=len(train_img_info) // config.IMAGES_PER_GPU,
                        verbose=1,
                        initial_epoch=args.init_epochs,
                        callbacks=get_call_back('rcnn'))

        if args.weight_path is not None:
            model.save(args.weight_path)
        else:
            model.save(config.rcnn_weights)


if __name__ == '__main__':
    pass

