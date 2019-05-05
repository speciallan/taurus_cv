#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan
"""
获取预训练模型
"""

import os

# 这里获取预训练模型
class PretrainedModels:
    pass

# 获取方法
def get(weight_path, network='resnet50'):

    filepath = './'

    if weight_path and os.path.exists(weight_path):
        return weight_path

    elif network == 'resnet50':
        filename = '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        filepath = os.path.dirname(os.path.abspath(__file__)) + filename

    # 不存在，则下载 speciallan.cn
    if not os.path.exists(filepath):
        print('请下载预训练模型到预训练目录')
        exit()

    return filepath

if __name__ == '__main__':
    print(get())
