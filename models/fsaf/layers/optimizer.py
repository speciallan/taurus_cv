#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import keras


def get_optimizer(base_lr):
    return keras.optimizers.adam(lr=base_lr, clipnorm=0.001)
