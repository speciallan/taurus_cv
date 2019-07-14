from __future__ import division

import PIL
import cv2
import keras
import numpy as np

from taurus_cv.models.retinanet.model.transform import change_transform_origin


def read_image_rgb(path):
    image = np.asarray(PIL.Image.open(path).convert('RGB'))
    return image.copy()


def read_image_bgr(path):
    image = np.asarray(PIL.Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x):

    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):

    height, width, channels = image.shape

    result = transform

    if relative_translation:
        result[0:2, 2] *= [width, height]

    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:

    def __init__(
            self,
            fill_mode='nearest',
            interpolation='linear',
            cval=0,
            data_format=None,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

        if data_format is None:
            data_format = keras.backend.image_data_format()
        self.data_format = data_format

        if data_format == 'channels_first':
            self.channel_axis = 0
        elif data_format == 'channels_last':
            self.channel_axis = 2
        else:
            raise ValueError("Il valore di 'data_format' deve essere 'channels_first' o 'channels_last', mentre invece è '{}'".format(data_format))

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):

    if params.channel_axis != 2:
        image = np.moveaxis(image, params.channel_axis, 2)

    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )

    if params.channel_axis != 2:
        output = np.moveaxis(output, 2, params.channel_axis)
    return output


def resize_image(img, min_side, max_side):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # 计算表根据图像的大小
    scale = min_side / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
