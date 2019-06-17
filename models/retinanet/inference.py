import matplotlib as mpl

mpl.use('Agg')

import cv2

from model.pascal_voc import save_annotations

from model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb

import numpy as np

from model.resnet import resnet_retinanet
from matplotlib import pyplot as plt

from config import Config
import os

# leggo la configurazione
config = Config('configRetinaNet.json')

# se non ci sono pesi specifici, uso i pesi base e le classi base
wname = 'BASE'
wpath = config.base_weights_path
classes = ['0', '1', '2', '3', '4', '5', '6']

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
if os.path.isfile(wpath):
    model.load_weights(wpath, by_name=True, skip_mismatch=True)
    print("Caricati pesi " + wname)
else:
    print("Senza pesi")

start_index = config.test_start_index
for nimage, imgf in enumerate(sorted(os.listdir(config.test_images_path))):
    imgfp = os.path.join(config.test_images_path, imgf)
    if os.path.isfile(imgfp):
        try:
            img = read_image_bgr(imgfp)
        except:
            continue
        img = preprocess_image(img.copy())
        img, scale = resize_image(img, min_side=config.img_min_size, max_side=config.img_max_size)

        orig_image = read_image_rgb(imgfp)

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

        # filtra quelli con punteggio > 0.2
        indices = np.where(detections[0, :, 4:] >= 0.3)

        # seleziona i punteggi che passano il filtro
        scores = scores[indices]

        # ordina i punteggi
        scores_sort = np.argsort(-scores)[:100]

        # seleziona le varie informazioni di ogni punteggio selezionato
        image_boxes = detections[0, indices[0][scores_sort], :4]
        image_scores = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]

        if config.test_save_annotations:
            orig_image = cv2.imread(imgfp)

            boxes = []
            if len(image_boxes) > 0:
                for i, box in enumerate(image_boxes):
                    box_json = {
                        "name": classes[image_predicted_labels[i]],
                        "xmin": int(box[0]),
                        "ymin": int(box[1]),
                        "xmax": int(box[2]),
                        "ymax": int(box[3])
                    }
                    boxes.append(box_json)
            save_annotations(config.test_result_path,
                             "{0:08d}".format(start_index + nimage),
                             orig_image,
                             boxes)
        else:
            # prepara una tavolozza di colori
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
            # disegno i box
            plt.imshow(orig_image)
            current_axis = plt.gca()

            if len(image_boxes) > 0:
                for i, box in enumerate(image_boxes):
                    # trasformo le coordinate normalizzate in coordinate assolute
                    xmin = box[0]
                    ymin = box[1]
                    xmax = box[2]
                    ymax = box[3]
                    color = colors[i % len(colors)]
                    label = '{}: {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0])
                    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
            plt.savefig(os.path.join(config.test_result_path, imgf))
            plt.close()

        print("Elaborata immagine '" + imgf + "'")
