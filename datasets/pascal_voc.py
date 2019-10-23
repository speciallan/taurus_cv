#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
通用获取PASCAL VOC公共数据集
"""

import os
import xml.etree.cElementTree as ET

def get_voc_dataset(input_path, sub_dir='VOC2007', class_mapping=[]):
    """
    获取VOC数据
    :param input_path: voc数据集路径
    :param class_mapping: 类别映射
    :return: 所有图片数组，分类计数，分类映射
    """

    all_imgs = []

    classes_count = {}

    # VOC数据集根目录，包括2007、2012
    # data_paths = [os.path.join(input_path, s) for s in ['VOC2007']]
    data_paths = [os.path.join(input_path, sub_dir)]

    print('正在解析标记文件')

    for data_path in data_paths:

        # 解析数据集文件夹
        anno_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')

        # 获取训练、测试集图片文件名
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_files = []
        test_files = []

        # 讲训练、测试集图片文件名写入到数组
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')

        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')

        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        annos = [os.path.join(anno_path, s) for s in os.listdir(anno_path)]
        annos.sort()

        idx = 0

        # 解析xml
        for anno in annos:

            try:
                idx += 1

                et = ET.parse(anno)
                element = et.getroot()

                # 解析基础图片数据
                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_filename = element_filename + '.jpg' if '.jpg' not in element_filename else element_filename
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                annotation_data = {}

                # 如果有检测目标，解析目标数据
                if len(element_objs) > 0:
                    annotation_data = {'filename': element_filename,
                                       'filepath': os.path.join(imgs_path, element_filename),
                                       'width': element_width,
                                       'height': element_height,
                                       'bboxes': []}

                    # 划分训练、测试集
                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                # 加入类别映射
                for element_obj in element_objs:

                    class_name = element_obj.find('name').text

                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        # 类别id从1开始，0保留为背景
                        class_mapping[class_name] = len(class_mapping) + 1

                    obj_bbox = element_obj.find('bndbox')

                    # voc的坐标格式
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    # difficulty = int(element_obj.find('difficult').text) == 1

                    annotation_data['bboxes'].append(
                        {'class_name': class_name,
                         'class_id': class_mapping[class_name],
                         'x1': x1, 'x2': x2,
                         'y1': y1, 'y2': y2,
                         # 'difficult': difficulty
                         })

                all_imgs.append(annotation_data)

            except Exception as e:

                print(e)
                continue

    return all_imgs, classes_count, class_mapping

if __name__ == '__main__':

    voc_path = '/home/speciallan/python/data'
    get_voc_dataset(voc_path, [])
