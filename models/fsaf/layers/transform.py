import threading

import numpy as np

DEFAULT_PRNG = np.random


def colvec(*args):
    return np.array([args]).T


def transform_aabb(transform, aabb):
    x1, y1, x2, y2 = aabb

    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1, 1, 1, 1],
    ])

    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def _random_vector(min, max, prng=DEFAULT_PRNG):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def rotation(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, prng=DEFAULT_PRNG):
    return rotation(prng.uniform(min, max))


def translation(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng=DEFAULT_PRNG):
    return translation(_random_vector(min, max, prng))


def shear(angle):
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng=DEFAULT_PRNG):
    return shear(prng.uniform(min, max))


def scaling(factor):
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PRNG):
    return scaling(_random_vector(min, max, prng))


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def change_transform_origin(transform, center):
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def random_transform(
        min_rotation=0,
        max_rotation=0,
        min_translation=(0, 0),
        max_translation=(0, 0),
        min_shear=0,
        max_shear=0,
        min_scaling=(1, 1),
        max_scaling=(1, 1),
        flip_x_chance=0,
        flip_y_chance=0,
        prng=DEFAULT_PRNG
):
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


def random_transform_generator(prng=None, **kwargs):

    if prng is None:
        prng = np.random.RandomState()

    lock = threading.Lock()

    while True:
        with lock:
            yield random_transform(prng=prng, **kwargs)
