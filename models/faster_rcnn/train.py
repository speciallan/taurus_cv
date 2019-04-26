#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
训练模型
"""

from taurus_cv.models.faster_rcnn.io.input import get_prepared_detection_dataset
from taurus_cv.models.faster_rcnn.config import current_config as config
from taurus_cv.models.faster_rcnn.preprocessing.generator import image_generator
from taurus_cv.models.faster_rcnn.layers import network
from taurus_cv.models.faster_rcnn.training import trainer, saver, observer


def train(args):
    """
    主训练程序
    :param args: 比如通过命令python train.py --stage=rpn --epochs=90 获取的参数值
    :return:
    """

    # 设置运行时环境 / training.trainer模块
    trainer.set_runtime_environment()

    # 获取VOC数据集中的训练集数据，并根据models/faster_rcnn/config中的分类关联数据，得到最后的训练测试集 / io模块
    train_img_list = get_prepared_detection_dataset(config).get_train_data()

    print("训练集图片数量:{}".format(len(train_img_list)))

    # 生成器 没有做数据增强 / preprocessing模块
    generator = image_generator(image_list=train_img_list,
                                batch_size=config.IMAGES_PER_GPU,
                                max_output_dim=config.IMAGE_MAX_DIM,
                                max_gt_num=100)

    # 先训练rpn
    if 'rpn' in args.stages:

        # 获取rpn网络 / layers.network模块 / 这里的rpn_net要重构为class，再把compile放进去
        model = network.rpn_net(config)

        # 添加损失，在network中定义好的
        loss_names = ["rpn_bbox_loss", "rpn_class_loss"]

        # 编译模型
        network.compile(model, config, loss_names)

        # 增加观测值
        model = observer.add_rpn_observer(model)

        # 打印模型信息
        # model.summary()

        # 训练模型
        trainer.train_rpn(model=model,
                          generator=generator,
                          iterations=len(train_img_list) // config.BATCH_SIZE,
                          epochs=args.epochs,
                          init_epochs=args.init_epochs)

        # 保存模型
        saver.save_model(model=model, weight_path=config.rpn_weights)

    # 端到端训练rcnn
    if 'rcnn' in args.stages:

        # 定义模型 / layers.network模块
        model = network.faster_rcnn(config=config)

        # 添加损失，在network中定义好的
        loss_names = ["rpn_bbox_loss", "rpn_class_loss", "rcnn_bbox_loss", "rcnn_class_loss"]

        # 编译模型
        network.compile(model, config, loss_names)

        # 增加rcnn观测值
        observer.add_rcnn_observer(model)

        # 打印模型信息
        # model.summary()

        # 训练模型
        trainer.train_rcnn(model=model,
                           generator=generator,
                           iterations=len(train_img_list) // config.BATCH_SIZE,
                           epochs=args.epochs,
                           init_epochs=args.init_epochs,
                           init_weight_path=args.init_weight_path)

        # 保存模型
        if args.weight_path is not None:
            saver.save_model(model=model, weight_path=args.weight_path)
        else:
            saver.save_model(model=model, weight_path=config.rcnn_weights)


if __name__ == '__main__':
    pass

