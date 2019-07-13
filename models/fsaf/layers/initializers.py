import math

import keras
import numpy as np


class PriorProbability(keras.initializers.Initializer):
    """
    先验概率初始化
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):

        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result
