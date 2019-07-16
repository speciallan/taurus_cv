#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import argparse
import sys
from math import ceil

sys.path.append('../../..')

from keras import metrics

from taurus_cv.models.fsaf.io.input import get_prepared_detection_dataset
from taurus_cv.models.fsaf.networks.retinanet import retinanet as retinanet
from taurus_cv.models.fsaf.preprocessing.generator import generator
from taurus_cv.models.fsaf.training import trainer
from taurus_cv.models.fsaf.config import current_config as config
from taurus_cv.models.fsaf.layers.loss import get_loss
from taurus_cv.models.fsaf.layers.optimizer import get_optimizer
from taurus_cv.models.fsaf.layers.generator import get_generators

from taurus_cv.models.retinanet.config import Config
from taurus_cv.models.retinanet.model.resnet import resnet_retinanet

from taurus_cv.utils.spe import spe


def train(args):

    # 加载配置

    # 设置运行时环境 / training.trainer模块
    trainer.set_runtime_environment()

    # 获取VOC数据集中的训练集数据，并根据models/retinanet/config中的分类关联数据，得到最后的训练测试集 / io模块
    # train_img_list = get_prepared_detection_dataset(config).get_train_data()
    # train_img_list = train_img_list[:100]

    # 生成数据，增加google数据增强 [[1,512,512,3] -> model.outputs for y_pred, [1,?,4+8] for y_true]
    # image_size = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)
    # train_generator = generator(image_list=train_img_list, num_classes=config.NUM_CLASSES, batch_size=config.BATCH_SIZE, image_size=image_size)

    config_json = Config('configRetinaNet.json')
    train_generator, val_generator, n_train_samples, n_val_samples = get_generators(config_json.images_path,
                                                                                    config_json.annotations_path,
                                                                                    config_json.train_val_split,
                                                                                    config_json.batch_size,
                                                                                    config_json.classes,
                                                                                img_min_size=config_json.img_min_size,
                                                                                img_max_size=config_json.img_max_size,
                                                                                transform=config_json.augmentation,
                                                                                debug=False)
    print("训练集图片数量:{}".format(n_train_samples))

    # t = next(gen)
    # print(t[0].shape, t[1][0].shape, t[1][1].shape)
    # exit()

    # 构造模型，加载权重
    model = retinanet(config)
    # model.summary()
    # model, _ = resnet_retinanet(len(config_json.classes), backbone=config_json.type, weights='imagenet', nms=True)
    model.compile(loss=get_loss(), optimizer=get_optimizer(config.LEARNING_RATE), metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=ceil(n_train_samples / config.BATCH_SIZE),
                        epochs=args.epochs,
                        callbacks=trainer.get_callback(config))

    model.save_weights(config.retinanet_weights)

    # 训练模型并保存
    # model = trainer.train_retinanet(model, gen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', default=5, type=int, help='epochs')
    args = parser.parse_args(sys.argv[1:])

    train(args)
