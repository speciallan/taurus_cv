import sys
import keras
import keras_resnet
import keras_resnet.models

sys.path.append('../../..')

from model.retinanet import custom_objects, retinanet_bbox
# from keras.applications import imagenet_utils
from keras.utils import get_file
# from keras.applications.imagenet_utils import imagenet_utils

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)


def download_imagenet(backbone):
    filename = resnet_filename.format(backbone[6:])
    resource = resnet_resource.format(backbone[6:])
    if backbone == 'resnet50':
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 'resnet101':
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 'resnet152':
        checksum = '6ee11ef2b135592f8031058820bb9e71'
    else:
        raise ValueError("Il backbone '{}' non è riconosciuto.".format(backbone))

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # determine which weights to load
    if weights == 'imagenet':
        weights_path = download_imagenet(backbone)
    elif weights is None:
        weights_path = None
    else:
        weights_path = weights

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
        # from taurus_cv.models.resnet.resnet import resnet50_fpn
        # resnet = resnet50_fpn(inputs)
        # resnet.summary()
        # print(resnet.outputs)
        # exit()
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError("Il backbone '{}' non è riconosciuto.".format(backbone))

    # create the full model
    model = retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_outputs=resnet.outputs[1:], **kwargs)

    # optionally load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True, skip_mismatch=skip_mismatch)
        print("Caricati pesi BACKEND")

    return model, resnet.layers
