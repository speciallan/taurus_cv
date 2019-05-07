#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
可视化组件
"""

import matplotlib.pyplot as plt
from matplotlib import patches, lines
import random
import colorsys
import numpy as np


def random_colors(N, bright=True):
    """
    生成HSV随机颜色并转换成RGB
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


# 画图方法
def display_instances(image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_bbox=True,
                      colors=None, captions=None):
    """
    可视化实例
    boxes: [num_instance, (y1, x1, y2, x2)] in image coordinates.
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** 此图没有实例展示 *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # 生成随机颜色
    colors = colors or random_colors(N)

    # 展示边界外的区域
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('on')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="-",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]

            if isinstance(score, np.ndarray):
                score = score[0]

            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]

        # 写文字
        ax.text(x1, y1 - 4, caption, color='white', size=12, backgroundcolor='none') #none

    ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        plt.show()


if __name__ == '__main__':
    print(random_colors(2))
