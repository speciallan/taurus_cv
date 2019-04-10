#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan
"""
通用获取PASCAL VOC公共数据集
"""

import os
import xml.etree.cElementTree as ET

# 获取voc数据
def get_voc_data(input_path, class_mapping):

    all_imgs = []

    classes_count = {}

    data_paths = [os.path.join(input_path, s) for s in ['VOC2007']]

    print('Parsing annotation files')

    for data_path in data_paths:

        # 解析数据集文件夹
        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_files = []
        test_files = []

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

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]

        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filename': element_filename,
                                       'filepath': os.path.join(imgs_path, element_filename),
                                       'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

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
                    difficulty = int(element_obj.find('difficult').text) == 1

                    annotation_data['bboxes'].append(
                        {'class_name': class_name,
                         'class_id': class_mapping[class_name],
                         'x1': x1, 'x2': x2,
                         'y1': y1, 'y2': y2,
                         'difficult': difficulty})

                all_imgs.append(annotation_data)

            except Exception as e:

                print(e)
                continue

    return all_imgs, classes_count, class_mapping

if __name__ == '__main__':

    voc_path = '/home/speciallan/python/data'
    get_voc_data(voc_path)
