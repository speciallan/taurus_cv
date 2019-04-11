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

from .io.input import get_prepared_detection_dataset
from .config import current_config as config
from .utils.generator import image_generator
from .layers import network

def set_gpu_growth():
    """
    GPU设置，设置后端，包括字符精度
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True  # 不要启动的时候占满gpu显存，按需申请空间
    session = tf.Session(config=cfg)     # 生成tf.session
    keras.backend.set_session(session)   # 设置后端为tensorflow
    # keras.backend.set_floatx('float16')  # 设置字符精度，默认float32，使用float16会提高训练效率，但是可能导致精度不够，梯度出现问题。

def get_call_back(stage):
    """
    定义callback，用于每个epoch回调
    包括模型检查点ModelCheckpoint,学习率自减少ReduceLROnPlateau,训练过程可视化TensorBoard，他们都继承自Callback
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/frcnn-' + stage + '.{epoch:03d}.h5', # 模型路径，默认保存在/tmp下
                                 monitor='acc',                                      # 监视值，包括精度acc、损失loss
                                 verbose=1,                                          # 是否显示进度条
                                 save_best_only=False,                               # 知否只保存最好模型
                                 period=5)                                           # checkpoint间隔的epoch数量

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='loss', # 监视值
                                   factor=0.1,     # 减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                   cooldown=0,     # 学习率减少后，会经过cooldown个epoch才重新进行检查
                                   patience=5,     # 经过patience个epoch后，如果检测值没变化，则出发学习率减少
                                   min_lr=0)       # 最小学习率

    # 能够收敛，这里没使用早停
    # early_stopping = EarlyStopping(monitor='loss', # 监视值
    #                                patience=0,     # 早停出发后，经过patience个epoch停止训练
    #                                verbose=1,      # 展示信息
    #                                mode='auto')    # auto,min,max 当监测值不再减小/增加后触发早停

    # 保存训练日志
    logs = TensorBoard(log_dir=config.log_path)    # 日志保存路径，这里的值来自experiments里面的config.ini

    return [checkpoint, lr_reducer, logs]          # 继承自Callback的类都能保存成list返回


def train(args):
    """
    主训练程序
    :param args: 比如通过命令python train.py --stage=rpn --epochs=90 获取的参数值
    :return:
    """

    set_gpu_growth()

    # 获取VOC原始数据集，并根据models/faster_rcnn/config中的分类关联数据，得到最后的训练测试集
    dataset = get_prepared_detection_dataset(config)

    # 根据VOC数据集中的trainval属性筛选出训练集图片
    train_img_list = [info for info in dataset.get_image_list() if info['type'] == dataset.TRAIN_LABEL]

    print("训练集图片数量:{}".format(len(train_img_list)))

    # 生成器 没有做数据增强
    generator = image_generator(image_list=train_img_list, batch_size=config.IMAGES_PER_GPU, max_output_dim=config.IMAGE_MAX_DIM, max_gt_num=100)

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
        model.fit_generator(generator,
                            epochs=args.epochs,
                            steps_per_epoch=len(train_img_list) // config.IMAGES_PER_GPU,
                            verbose=1,
                            initial_epoch=args.init_epochs,
                            callbacks=get_call_back('rpn'))

        model.save(config.rpn_weights)

    if 'rcnn' in args.stages:

        model = network.faster_rcnn(config=config)

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
        model.fit_generator(generator,
                            epochs=args.epochs,
                            steps_per_epoch=len(train_img_list) // config.IMAGES_PER_GPU,
                            verbose=1,
                            initial_epoch=args.init_epochs,
                            callbacks=get_call_back('rcnn'))

        if args.weight_path is not None:
            model.save(args.weight_path)
        else:
            model.save(config.rcnn_weights)


if __name__ == '__main__':
    pass

