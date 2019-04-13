#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
保存模型
"""

def save_model(model, weight_path):
    """
    通过keras实现的模型保存
    :param model:
    :param weight_path:
    :return:
    """
    model.save(weight_path)
