#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
评估工具
"""

import numpy as np
from taurus_cv.models.faster_rcnn.utils import np_utils
from taurus_cv.utils.spe import spe


def get_detections(boxes, scores, predict_labels, num_classes, score_shreshold=0.05, max_boxes_num=100):
    """
    获取检测信息
    :param boxes: 检测边框，list of numpy [num_images,[n,(y1,x1,y2,x2)]]
    :param scores: 预测得分，list of numpy [num_images,[n,]]
    :param predict_labels: 预测类别，list of numpy [num_images,[n,]]，
    :param num_classes: 类别数
    :param score_shreshold: 评分阈值
    :param max_boxes_num:
    :return: list of list of numpy(num_boxes,5)  [num_images,[num_classes,[num_boxes,(y1,x1,y2,x2,scores)]]]
             每张图像，每个类别的预测边框；注意num_boxes是变化的；
    """
    # 初始化结果
    num_images = len(boxes)
    all_detections = [[None for j in range(num_classes)] for i in range(num_images)]  # (num_images,num_classes)

    # 逐个图像处理
    for image_idx in range(num_images):

        # 去除padding
        cur_boxes = boxes[image_idx]  # (n,4)
        cur_scores = scores[image_idx]  # (n,)
        cur_predict_labels = predict_labels[image_idx]  # (n,)

        # 过滤排序
        indices = np.where(cur_scores >= score_shreshold)[0]  # 选中的索引号，tuple的第一个值，一个一维numpy数组
        select_scores = cur_scores[indices]
        scores_sort_indices = np.argsort(select_scores * -1)[:max_boxes_num]  # (m,)选中评分排序过滤后的索引号

        # 最终的选中边框的索引号
        indices = indices[scores_sort_indices]

        # 选中的边框，得分，类别
        cur_boxes = cur_boxes[indices]
        cur_scores = cur_scores[indices]
        cur_predict_labels = cur_predict_labels[indices]

        # 合并边框和得分
        cur_detections = np.concatenate([cur_boxes, np.expand_dims(cur_scores, axis=1)], axis=1)

        # print(cur_detections, cur_predict_labels)
        # exit()
        # 逐个类别处理
        for class_id in range(num_classes):
            all_detections[image_idx][class_id] = cur_detections[cur_predict_labels == class_id]

    return all_detections


def get_annotations(image_info_list, num_classes, order=False):
    """
    获取所有的编著
    :param image_info_list: list of dict, 图像数据信息
                        image_info['boxes'] 是(n,4)数组,
                        image_info['labels'] 是(n,1)数组
    :param num_classes: 类别数
    :return: list of list of numpy(num_boxes,4)  [num_images,[num_classes,[num_gt,(y1,x1,y2,x2)]]]
             每张图像，每个类别的GT边框; 注意num_gt是变化的
    """
    num_images = len(image_info_list)
    all_annotations = [[None for j in range(num_classes)] for i in range(num_images)]  # (num_images,num_classes)
    for image_idx in range(num_images):

        gt_boxes = image_info_list[image_idx]['boxes']  # 此图片的GT边框
        if order:
            gt_boxes = gt_boxes[:,[1,0,3,2]]
        for class_id in range(num_classes):
            # [3,3,6,6] 获取class_id=6的索引[2,3], 通过[2,3]获取boxes
            indices = np.where(image_info_list[image_idx]['labels'] == str(class_id))
            all_annotations[image_idx][class_id] = gt_boxes[indices]

    return all_annotations


def voc_ap(rec, prec, use_07_metric=False):
    """
    voc 精度计算，此方法来自：https://github.com/rbgirshick/py-faster-rcnn
    :param rec: 召回率，numpy数组(n,)
    :param prec: 精度，numpy数组(n,)
    :param use_07_metric: 是否使用VOC 07的11点法
    :return: ap
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(all_annotations, all_detections, iou_threshold=0.5, use_07_metric=False, img_info=None):
    """
    voc数据集评估
    :param all_annotations:list of list of numpy(num_boxes,4) [num_images,[num_classes,[num_gt,(y1,x1,y2,x2)]]]
             每张图像，每个类别的GT边框; 注意num_gt是变化的
    :param all_detections:list of list of numpy(num_boxes,5) [num_images,[num_classes,[num_boxes,(y1,x1,y2,x2,scores)]]]
             每张图像，每个类别的预测边框；注意num_boxes是变化的；
    :param iou_threshold: iou阈值
    :param use_07_metric:
    :return: ap numpy数组，(num_classes,)
    """
    num_classes = len(all_annotations[0])
    num_images = len(all_detections)
    average_precisions = {}

    # 逐个类别计算ap
    for class_id in range(num_classes):

        true_positives = np.zeros((0,), dtype=np.float64)
        false_positives = np.zeros((0,), dtype=np.float64)
        scores = np.zeros((0,), dtype=np.float64)
        num_gt_boxes = 0.0

        # 记录错误
        none = []
        wrong = []

        # debug
        # if class_id != 4:
        #     continue

        # 逐个图像处理
        # print('all_detection_num:', num_images)

        for image_id in range(num_images):

            gt_boxes = all_annotations[image_id][class_id]  # (n,x1,y1,x2,y2)
            # print('cid:', class_id, 'imgid:', image_id, img_info[image_id]['filename'], len(gt_boxes))

            # 打印所有gtbox
            # print(img_info[image_id]['filename'], gt_boxes.shape[0], gt_boxes)
            num_gt_boxes += gt_boxes.shape[0]  # gt个数

            detected_gt_boxes = []  # 已经检测匹配过的gt边框

            for detect_box in all_detections[image_id][class_id]:

                scores = np.append(scores, detect_box[4])

                # 有gtbox的时候打印gt和det
                # if img_info != None:
                    # print_detection = detect_box.astype(np.int)
                    # print('img:{}, gt:{}, det:{}'.format(img_info[image_id], gt_boxes, print_detection))

                # 000142
                # if img_info[image_id]['filename'] == '000227.jpg':
                #     print(gt_boxes)
                #     exit()

                # 如果没有GT 边框  1！！！没有为什么要fp+ 因为没有gtbox但是被检测出来了
                if gt_boxes.shape[0] == 0:
                    true_positives = np.append(true_positives, 0)
                    false_positives = np.append(false_positives, 1)

                    # debug
                    if img_info != None:
                        img_info[image_id]['boxes'] = img_info[image_id]['boxes'][:, [1,0,3,2]]
                        # print_detection = [map(int, box) for box in enumerate(detect_box)]
                        print_gtboxes = gt_boxes.astype(np.int)
                        print_detection = detect_box.astype(np.int)
                        none.append('[none] img:{}, class:{}, gt:{}, det:{}'.format(img_info[image_id]['filename'], class_id, gt_boxes, print_detection))

                    continue

                # 计算iou
                # print(gt_boxes, detect_box[:4])
                # exit()
                # if img_info[image_id]['filename'] == '000295.jpg':
                #     print(gt_boxes, detect_box)
                #     exit()

                iou = np_utils.compute_iou(gt_boxes, np.expand_dims(detect_box[:4], axis=0))  # n vs 1
                max_iou = np.max(iou, axis=0)[0]  # 与GT边框的最大iou值
                argmax_iou = np.argmax(iou, axis=0)[0]  # 最大iou值对应的GT

                # 如果超过iou阈值,且之前没有检测框匹配
                if max_iou >= iou_threshold and argmax_iou not in detected_gt_boxes:
                    true_positives = np.append(true_positives, 1)
                    false_positives = np.append(false_positives, 0)
                    detected_gt_boxes.append(argmax_iou)
                else:
                    true_positives = np.append(true_positives, 0)
                    false_positives = np.append(false_positives, 1)

                    # debug
                    if img_info != None:
                        img_info[image_id]['boxes'] = img_info[image_id]['boxes'][:, [1,0,3,2]]
                        # print_detection = [map(int, box) for box in enumerate(detect_box)]
                        print_gtboxes = gt_boxes.astype(np.int)
                        print_detection = detect_box.astype(np.int)
                        wrong.append('[wrong] img:{}, class:{}, gt:{}, det:{}'.format(img_info[image_id]['filename'], class_id, gt_boxes, print_detection))

        if class_id != 0:
            [print(v) for k,v in enumerate(none)]
            [print(v) for k,v in enumerate(wrong)]

        # 每个类别按照得分排序
        indices = np.argsort(scores * -1)

        # [1. 1. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 0. 0. 0. 1. 0. 0.]
        true_positives = true_positives[indices]
        false_positives = false_positives[indices]
        # print(class_id, true_positives)

        # 累加 [ 1.  2.  3.  4.  5.  5.  6.  7.  7.  7.  8.  9. 10. 10. 10. 10. 11. 11. 11.]
        true_positives = np.cumsum(true_positives)
        false_positives = np.cumsum(false_positives)

        # 计算召回率和精度
        recall = true_positives / num_gt_boxes
        # print(recall)
        # print(true_positives, true_positives + false_positives)

        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        print('class:{}'.format(class_id))
        print('gtboxes_num: {}'.format(num_gt_boxes))
        print('recall:{}'.format(recall[-1]) if len(recall) != 0 else recall)
        # 这里preci[0]改成-1
        print('precision:{}'.format(precision[-1] if len(precision) != 0 else precision))
        print('------------------------')

        # 计算ap
        average_precisions[class_id] = voc_ap(recall, precision, use_07_metric=use_07_metric)

    return average_precisions
