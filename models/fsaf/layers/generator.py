import os
import random
import time

from taurus_cv.models.fsaf.layers.pascal_voc import PascalVocGenerator
from taurus_cv.models.fsaf.layers.transform import random_transform_generator


def get_generators(images_path, annotations_path, train_val_split, batch_size, classes, img_min_size, img_max_size, shuffle=True, debug=False, transform=True):
    """
    获取生成器
    :param images_path:
    :param annotations_path:
    :param train_val_split:
    :param batch_size:
    :param classes:
    :param img_min_size:
    :param img_max_size:
    :param shuffle:
    :param debug:
    :param transform:
    :return:
    """
    if transform:
        transform_generator = random_transform_generator(min_rotation=-0.1,
                                                         max_rotation=0.1,
                                                         min_translation=(-0.1, -0.1),
                                                         max_translation=(0.1, 0.1),
                                                         min_shear=-0.1,
                                                         max_shear=0.1,
                                                         min_scaling=(0.9, 0.9),
                                                         max_scaling=(1.1, 1.1),
                                                         flip_x_chance=0.5,
                                                         flip_y_chance=0.5)
    else:
        transform_generator = None

    annotation_files = [os.path.splitext(f)[0] for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))]

    if shuffle:
        random.seed = 19081974
        random.shuffle(annotation_files)
    max_id = int(train_val_split * len(annotation_files))
    train_ids = annotation_files[:max_id]
    if train_val_split < 1.:
        val_ids = annotation_files[max_id:]
    else:
        val_ids = None

    random.seed = int(round(time.time() * 1000))

    train_generator = PascalVocGenerator(
        annotations_path,
        images_path,
        train_ids,
        classes,
        image_min_side=img_min_size,
        image_max_side=img_max_size,
        transform_generator=transform_generator,
        batch_size=batch_size,
        debug=debug
    )

    if val_ids is not None:
        validation_generator = PascalVocGenerator(
            annotations_path,
            images_path,
            val_ids,
            classes,
            image_min_side=img_min_size,
            image_max_side=img_max_size,
            transform_generator=None,
            batch_size=batch_size
        )
        return train_generator, validation_generator, train_generator.size(), validation_generator.size()
    else:
        return train_generator, None, train_generator.size(), 0
