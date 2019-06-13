import cv2
import numpy as np
from model.generator import get_generators
from model.visualization import draw_annotations, draw_boxes

from config import Config

# leggo la configurazione
config = Config('configRetinaNet.json')

generator, _, _, _ = get_generators(config.images_path, config.annotations_path, 1., 1, config.classes, 512,512, shuffle=False, transform=False)

# creo la finestra di visualizzazione delle immagini
cv2.namedWindow('Immagine', cv2.WINDOW_NORMAL)

# wisualizza una immagine alla volta
i = 0
while i < generator.size():
    # carica l'immagine
    image = generator.load_image(i)
    annotations = generator.load_annotations(i)

    # applica trasformazioni di augmentation
    image, annotations = generator.random_transform_group_entry(image, annotations)

    # ridimensiona immagine e annotazioni
    image, image_scale = generator.resize_image(image)
    annotations[:, :4] *= image_scale

    # trasformo le immagini in sfumature di grigio per evidenziare meglio i colori dei box
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # ottiene gli anchor box target
    labels, boxes, anchors = generator.anchor_targets(image.shape, annotations, generator.num_classes())

    # disegna gli anchor box sull'immagine (in azzurro)
    draw_boxes(image, anchors[np.max(labels, axis=1) == 1], (255, 255, 0), thickness=1)

    # disegna i box delle annotazioni (in rosso)
    draw_annotations(image, annotations, color=(0, 0, 255), generator=generator)

    # disegna i bounding box (in verde)
    # il risultato è che le annotazioni senza bounding box resteranno in rosso
    draw_boxes(image, boxes[np.max(labels, axis=1) == 1], (0, 255, 0))

    # scrive in alto a sinistra il nome del file
    cv2.putText(image, generator.name_from_index(i), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow('Immagine', image)

    # attende la pressione di un tasto
    # con "q" si esce
    # con "+" si avanza nell'elenco delle immagini
    # con "-" si arretra nell'elenco delle immagini
    k = cv2.waitKey()
    if k == ord('q'):
        break
    if k == ord('+'):
        i = min(generator.size() - 1, i + 1)
    if k == ord('-'):
        i = max(0, i - 1)
