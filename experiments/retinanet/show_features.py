import matplotlib as mpl

mpl.use('Agg')

import numpy as np
import os
import cv2

from keras.engine.topology import Container
from keras.layers import Conv2D
from keras import backend as K

from taurus_cv.models.retinanet.model.image import resize_image
from taurus_cv.models.retinanet.model.resnet import resnet_retinanet
from taurus_cv.models.retinanet.config import Config


def crea_dizionario_layer_ricorsivo(layer, dict):
    if isinstance(layer, Container):
        for l in layer.layers:
            crea_dizionario_layer_ricorsivo(l, dict)
    else:
        dict[layer.name] = layer


config = Config('configRetinaNet.json')

wpath = config.base_weights_path
wname = "BASE"
classes = ['person',
           'bicycle',
           'car',
           'motorcycle',
           'airplane',
           'bus',
           'train',
           'truck',
           'boat',
           'traffic light',
           'fire hydrant',
           'stop sign',
           'parking meter',
           'bench',
           'bird',
           'cat',
           'dog',
           'horse',
           'sheep',
           'cow',
           'elephant',
           'bear',
           'zebra',
           'giraffe',
           'backpack',
           'umbrella',
           'handbag',
           'tie',
           'suitcase',
           'frisbee',
           'skis',
           'snowboard',
           'sports ball',
           'kite',
           'baseball bat',
           'baseball glove',
           'skateboard',
           'surfboard',
           'tennis racket',
           'bottle',
           'wine glass',
           'cup',
           'fork',
           'knife',
           'spoon',
           'bowl',
           'banana',
           'apple',
           'sandwich',
           'orange',
           'broccoli',
           'carrot',
           'hot dog',
           'pizza',
           'donut',
           'cake',
           'chair',
           'couch',
           'potted plant',
           'bed',
           'dining table',
           'toilet',
           'tv',
           'laptop',
           'mouse',
           'remote',
           'keyboard',
           'cell phone',
           'microwave',
           'oven',
           'toaster',
           'sink',
           'refrigerator',
           'book',
           'clock',
           'vase',
           'scissors',
           'teddy bear',
           'hair drier',
           'toothbrush']

if os.path.isfile(config.trained_weights_path):
    wpath = config.trained_weights_path
    classes = config.classes
    wname = "DEFINITIVI"
elif os.path.isfile(config.pretrained_weights_path):
    wpath = config.pretrained_weights_path
    classes = config.classes
    wname = "PRETRAINED"

if config.type.startswith('resnet'):
    model, bodyLayers = resnet_retinanet(len(classes), backbone=config.type, weights='imagenet', nms=True)
else:
    model = None
    bodyLayers = None
    print("Tipo modello non riconosciuto ({})".format(config.type))
    exit(1)

model.load_weights(wpath, by_name=True, skip_mismatch=True)
print("Caricati pesi " + wname)

layer_dict = dict()
crea_dizionario_layer_ricorsivo(model, layer_dict)

for imgf in os.listdir(config.test_images_path):
    imgfp = os.path.join(config.test_images_path, imgf)
    if os.path.isfile(imgfp):
        orig_image, _ = resize_image(cv2.imread(imgfp), min_side=config.img_min_size, max_side=config.img_max_size)
        cv2.imshow('Immagine', orig_image)

        skip_img = False

        for layer in layer_dict:
            if isinstance(layer_dict[layer], Conv2D):
                get_layer_output = K.function([model.input],
                                              [layer_dict[layer].output])

                intermediate_output = (get_layer_output([[orig_image]])[0])
                intermediate_output = intermediate_output - intermediate_output.min()
                intermediate_output = intermediate_output / intermediate_output.max()

                filter_idx = 0
                while filter_idx < layer_dict[layer].output.shape[3]:

                    filter_img = intermediate_output[0, :, :, filter_idx]
                    n_filter_img = np.array(filter_img * 255, dtype=np.uint8)
                    cm_filter_img = cv2.applyColorMap(n_filter_img, cv2.COLORMAP_JET)
                    cv2.imshow('Filtro', cm_filter_img)

                    # attendo la pressione di un tasto
                    # 'q' esce completamenet
                    # 'm' si sposta sulla prossima immagine
                    # 'n' si sposta sul prossimo layer
                    # '+' avanza di un filtro
                    # '-' indietreggia di un filtro
                    k = cv2.waitKey()
                    if k == ord('q'):
                        exit(0)
                    if k == ord('n'):
                        break
                    if k == ord('m'):
                        skip_img = True
                        break
                    if k == ord('+'):
                        filter_idx += 1
                    if k == ord('-'):
                        filter_idx = max(0, filter_idx - 1)

            if skip_img:
                break
