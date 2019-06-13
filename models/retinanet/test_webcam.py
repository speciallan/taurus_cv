import os

import cv2
import numpy as np

from config import Config
from model.image import preprocess_image, resize_image
from model.resnet import resnet_retinanet

# leggo la configurazione
config = Config('configRetinaNet.json')

# se non ci sono pesi specifici, uso i pesi base e le classi base
wname = 'BASE'
wpath = config.base_weights_path
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

# se invece ci sono pesi specifici, uso questi pesi e le classi per cui sono stati trovati
if os.path.isfile(config.trained_weights_path):
    wname = "DEFINITIVI"
    wpath = config.trained_weights_path
    classes = config.classes
if os.path.isfile(config.pretrained_weights_path):
    wname = 'PRETRAINED'
    wpath = config.pretrained_weights_path
    classes = config.classes

# creo il modello
if config.type.startswith('resnet'):
    model, _ = resnet_retinanet(len(classes), backbone=config.type, weights='imagenet', nms=True)
else:
    model = None
    print("Tipo modello non riconosciuto ({})".format(config.type))
    exit(1)

# carico i pesi
print("Modello backend: ", config.type)
model.load_weights(wpath, by_name=True, skip_mismatch=True)
print("Caricati pesi " + wname)

# carico le immagini originali e quelle ridimensionate in due array
# ne prendo una alla volta per minimizzare la memoria GPU necessaria
cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    orig_image = img.copy()

    img = preprocess_image(img.copy())
    img, scale = resize_image(img, min_side=config.img_min_size, max_side=config.img_max_size)

    # eseguo la perdizione sulle immagini ridimensionate
    _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))

    # esegue il clip dei box predetti alle dimensioni dell'immagine
    detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
    detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
    detections[:, :, 2] = np.minimum(img.shape[1], detections[:, :, 2])
    detections[:, :, 3] = np.minimum(img.shape[0], detections[:, :, 3])

    # corregge i box alla scala dell'immagine
    detections[0, :, :4] /= scale

    # seleziona i punteggi della classificazione
    scores = detections[0, :, 4:]

    # filtra quelli con punteggio > 0.9
    indices = np.where(detections[0, :, 4:] >= 0.9)

    # seleziona i punteggi che passano il filtro
    scores = scores[indices]

    # ordina i punteggi
    scores_sort = np.argsort(-scores)[:100]

    # seleziona le varie informazioni di ogni punteggio selezionato
    image_boxes = detections[0, indices[0][scores_sort], :4]
    image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
    image_detections = np.append(image_boxes, image_scores, axis=1)
    image_predicted_labels = indices[1][scores_sort]

    if len(image_boxes) > 0:
        for i, box in enumerate(image_boxes):
            # trasformo le coordinate normalizzate in coordinate assolute
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            label = '{}: {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0])
            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(orig_image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("webcam", orig_image)

    # premere 'esc' per uscire
    if cv2.waitKey(1) == 27:
        break
