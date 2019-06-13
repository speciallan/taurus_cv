import threading

import numpy as np

DEFAULT_PRNG = np.random


def colvec(*args):
    # crea un numpy array rappresentante un vettore colonna
    return np.array([args]).T


def transform_aabb(transform, aabb):
    # Applica una trasformazione a un bounding box allineato all'asse (AABB).
    # Il risultato è un nuovo AABB nello stesso sistema di coordinate dell'AABB originale.
    # Il nuovo AABB contiene tutti i punti d'angolo dell'AABB originale dopo l'applicazione della trasformazione indicata.
    x1, y1, x2, y2 = aabb

    # trasforma tutti e 4 gli angoli del AABB
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1, 1, 1, 1],
    ])

    # Estrae minimo e massimo dagli angoli
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def _random_vector(min, max, prng=DEFAULT_PRNG):
    # Costruisce un vettore contenente un mix random dove ogni elemento è compreso tra il relativo elemento di min e max
    # Il vettore finale avrà stessa shape di min e max
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def rotation(angle):
    # Costruisce una matrice di rotazione 2D
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, prng=DEFAULT_PRNG):
    # Esegue una rotazione random tra min e max (scalari in radianti assoluti)
    return rotation(prng.uniform(min, max))


def translation(translation):
    # Costruisce una matrice di traslazione 2D
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng=DEFAULT_PRNG):
    # Esegue una traslazione random tra min e max (vettori 2D con minimo e massimo di traslazione per ogni dimensione)
    return translation(_random_vector(min, max, prng))


def shear(angle):
    # Costruisce una matrice di inclinazione 2D
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng=DEFAULT_PRNG):
    # Esegue una inclinazione random tra min e max (scalari in radianti assoluti)
    return shear(prng.uniform(min, max))


def scaling(factor):
    # Costruisce una matrice di scaling 2D
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PRNG):
    # Esegue uno scaling random tra min e max (vettori 2D con fattori di scala minimo e massimo per ogni dimensione)
    return scaling(_random_vector(min, max, prng))


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    # Esegue un flip random su assi X/Y, considerando la probabilità che succeda (flip_x_chance, flip_y_chance)
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def change_transform_origin(transform, center):
    # Esegue una trasformazione considerando un nuovo punto di origine per la stessa
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
    # Crea trasformazioni random (augmentation)
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


def random_transform_generator(prng=None, **kwargs):
    # Crea un generator di trasformazioni random
    if prng is None:
        prng = np.random.RandomState()

    lock = threading.Lock()

    while True:
        with lock:
            yield random_transform(prng=prng, **kwargs)
