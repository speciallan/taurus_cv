#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import tensorflow as tf



def retinanet():
    print('retina')

    input = tf.placeholder(tf.float32, shape=(512,512,3), name='img')