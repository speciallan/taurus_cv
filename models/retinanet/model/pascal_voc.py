import os
import random
import threading
import warnings
from xml.dom import minidom

import cv2
import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from six import raise_from

from model.anchors import anchor_targets_bbox, bbox_transform
from model.image import read_image_bgr, TransformParameters, apply_transform, adjust_transform_for_image, resize_image, preprocess_image
from model.transform import transform_aabb

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def save_annotations(filepath, filename, image, boxes):
    cv2.imwrite(os.path.join(os.path.join(filepath, "images"), filename + ".jpg"), image)
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename + ".jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.shape[0])
    ET.SubElement(size, "height").text = str(image.shape[1])
    ET.SubElement(size, "depth").text = str(image.shape[2])
    for box in boxes:
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = box["name"]
        ET.SubElement(object, "pose").text = "Unspecified"
        ET.SubElement(object, "truncated").text = "0"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    myfile = open(os.path.join(os.path.join(filepath, "annotations"), filename + ".xml"), "w")
    myfile.write(xmlstr)


def _find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('Element non trovato \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('Valore non ammissibile per \'{}\': {}'.format(debug_name, e)), None)
    return result


class Generator(object):
    def __init__(
            self,
            image_min_side,
            image_max_side,
            transform_generator=None,
            batch_size=1,
            group_method='ratio',  # può valere 'none', 'random', 'ratio'
            shuffle_groups=True,
            transform_parameters=None,
            debug=False
    ):
        self.debug = debug
        if self.debug:
            batch_size = 1

        self.transform_generator = transform_generator
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()

        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError('Metodo "size" non implementato')

    def num_classes(self):
        raise NotImplementedError('Metodo "num_classes" non implementato')

    def name_to_label(self, name):
        raise NotImplementedError('Metodo "name_to_label" non implementato')

    def label_to_name(self, label):
        raise NotImplementedError('Metodo "label_to_name" non implementato')

    def name_from_index(self, index):
        raise NotImplementedError('Metodo "name_from_index" non implementato')

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError('Metodo "image_aspect_ratio" non implementato')

    def load_image(self, image_index):
        raise NotImplementedError('Metodo "load_image" non implementato')

    def load_annotations(self, image_index):
        raise NotImplementedError('Metodo "load_annotations" non implementato')

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # verifica le annotazioni
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert (isinstance(annotations, np.ndarray)), '\'load_annotations\' dovrebbe ritornare una lista di numpy array, invece ricevo: {}' \
                .format(type(annotations))

            annotations[:, 0] = np.maximum(0, annotations[:, 0])
            annotations[:, 1] = np.maximum(0, annotations[:, 1])
            annotations[:, 2] = np.minimum(image.shape[1], annotations[:, 2])
            annotations[:, 3] = np.minimum(image.shape[0], annotations[:, 3])

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 0] < 0) |
                (annotations[:, 1] < 0) |
                (annotations[:, 2] > image.shape[1]) |
                (annotations[:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Immagine "{}" con id {} (shape {}) contiene dei box invalidi: {}.'.format(
                    self.name_from_index(group[index]),
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations):

        # trasforma in modo random sia le immagini che le annotazioni
        if self.transform_generator:
            # preparo le trasformazioni random
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # applico alla immagine
            image = apply_transform(transform, image, self.transform_parameters)

            # applico alle annotazioni
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations

    def save_img_ann(self, image, annotations, where, index):
        name = str(self.group_index) + '_' + str(index) + '_' + where + '.jpg'

        image_n = image.copy()

        image_n -= np.min(image_n)
        image_n /= np.max(image_n)

        plt.imshow(image_n)
        current_axis = plt.gca()
        for ann in annotations:
            current_axis.add_patch(plt.Rectangle((ann[0], ann[1]), ann[2] - ann[0], ann[3] - ann[1], color=(1.0, 1.0, 1.0), fill=False, linewidth=2))
        plt.savefig(name)
        plt.close()

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group_entry(self, image, annotations, index):

        # preprocesso l'immagine
        image = self.preprocess_image(image)

        if self.debug:
            self.save_img_ann(image, annotations, '0PRE', index)

        # trasformo in modo random sia le immagini che le annotazioni
        image, annotations = self.random_transform_group_entry(image, annotations)

        # ridimensiono l'immagine
        image, image_scale = self.resize_image(image)

        # applico la scala alle annotazioni
        annotations[:, :4] *= image_scale

        if self.debug:
            self.save_img_ann(image, annotations, '1POST', index)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocessa un gruppo di immagini e annotazioni
            image, annotations = self.preprocess_group_entry(image, annotations, index)

            image_group[index] = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determina l'ordine delle immagini
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide il tutto in gruppi (corrispondenti ai batch)
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # ottiene la massima dimensione delle immagini
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # costruisce un vettore batch dove salvare le immagini
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copia le immagini nel vettore batch
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def anchor_targets(self,
                       image_shape,
                       annotations,
                       num_classes,
                       mask_shape=None,
                       negative_overlap=0.4,
                       positive_overlap=0.5,
                       **kwargs):
        return anchor_targets_bbox(image_shape, annotations, num_classes, mask_shape, negative_overlap, positive_overlap, **kwargs)

    def compute_targets(self, image_group, annotations_group):
        # ottiene la massima dimensione delle immagini
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # calcola label e target di regressione
        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # calcola i target di regressione
            labels_group[index], annotations, anchors = self.anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)
            regression_group[index] = bbox_transform(anchors, annotations)

            # aggiunge gli stati degli anchor ai target di regressione
            anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copia tutte le label e i valori di regressione al blob batch
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...] = labels
            regression_batch[index, ...] = regression

        return [regression_batch, labels_batch]

    def compute_input_output(self, group):
        # carica immagini e annotazioni
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # verifica la validità delle annotazioni
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # esegue un preprocesso dull'intero gruppo
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # calcola il layer di input della rete
        inputs = self.compute_inputs(image_group)

        # calcola il layer di output della rete
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # avanza sul prossimo gruppo, usando il lock per essere thread-safe (in caso di multithreading)
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # mischia i gruppi all'inizio di una epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


class PascalVocGenerator(Generator):
    def __init__(
            self,
            annotations_path,
            images_path,
            ids,
            classes,
            **kwargs
    ):
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.image_names = ids
        self.classes = {}

        for i, cl in enumerate(classes):
            self.classes[cl] = i

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def name_from_index(self, index):
        return self.image_names[index]

    def image_aspect_ratio(self, image_index):
        path = os.path.join(self.images_path, self.image_names[image_index] + '.jpg')
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = os.path.join(self.images_path, self.image_names[image_index] + '.jpg')
        return read_image_bgr(path)

    def __parse_annotation(self, element):
        class_name = _find_node(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('解析 \'{}\' 出现错误: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label(class_name)

        bndbox = _find_node(element, 'bndbox')
        box[0, 0] = _find_node(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _find_node(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _find_node(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _find_node(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return box

    def __parse_annotations(self, xml_root):
        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                box = self.__parse_annotation(element)
                boxes = np.append(boxes, box, axis=0)
            except ValueError as e:
                raise_from(ValueError('可能出现问题 #{}: {}'.format(i, e)), None)

        return boxes

    def load_annotations(self, image_index):
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.annotations_path, filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('文件问题: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('文件问题: {}: {}'.format(filename, e)), None)
