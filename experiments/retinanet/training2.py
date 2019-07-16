
import sys
import argparse
sys.path.append('../../..')

import os
from math import ceil

from taurus_cv.models.retinanet.config import Config
from taurus_cv.models.retinanet.model.callbacks import get_callbacks
from taurus_cv.models.retinanet.model.generator import get_generators
from taurus_cv.models.fsaf.layers.loss import get_loss
from taurus_cv.models.retinanet.model.optimizer import get_optimizer
from taurus_cv.models.retinanet.model.retinanet import retinanet_bbox
from taurus_cv.models.fsaf.io.input import get_prepared_detection_dataset
from taurus_cv.models.fsaf.preprocessing.generator import generator
from taurus_cv.models.fsaf.config import current_config as config2
from taurus_cv.models.fsaf.layers.optimizer import get_optimizer
from taurus_cv.models.fsaf.networks.retinanet import retinanet as retinanet
from taurus_cv.models.fsaf.training import trainer
from taurus_cv.models.retinanet.model.resnet import resnet_retinanet

# 获取配置
config = Config('configRetinaNet.json')

# 如果使用resnet
# model = retinanet(config2)
model, bodyLayers = resnet_retinanet(len(config.classes), backbone=config.type, weights='imagenet', nms=True)
# model = retinanet_bbox()
# model = retinanet(config2)
model.compile(loss=get_loss(), optimizer=get_optimizer(0.0001), metrics=['accuracy'])

config2.voc_path = '/home/speciallan/Documents/python/data/VOCdevkit'
config2.voc_sub_dir = 'dd'
print(config2.voc_path, config2.voc_sub_dir)

# train_img_list = get_prepared_detection_dataset(config2).get_train_data()
# train_img_list = train_img_list[:10]
# n_train_samples = len(train_img_list)
# print("训练集图片数量:{}".format(len(train_img_list)))

# 生成数据，增加google数据增强 [[1,512,512,3] -> model.outputs for y_pred, [1,?,4+8] for y_true]
# image_size = (config2.IMAGE_MAX_DIM, config2.IMAGE_MAX_DIM)
# train_generator = generator(image_list=train_img_list, num_classes=config2.NUM_CLASSES, batch_size=config2.BATCH_SIZE, image_size=image_size)


train_generator, val_generator, n_train_samples, n_val_samples = get_generators(config.images_path,
                                                                                config.annotations_path,
                                                                                config.train_val_split,
                                                                                config.batch_size,
                                                                                config.classes,
                                                                                img_min_size=config.img_min_size,
                                                                                img_max_size=config.img_max_size,
                                                                                transform=config.augmentation,
                                                                                debug=False)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(n_train_samples / config2.BATCH_SIZE),
                    epochs=10,
                    callbacks=trainer.get_callback(config2))

model.save_weights(config2.retinanet_weights)
