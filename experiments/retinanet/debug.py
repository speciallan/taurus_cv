import cv2
import numpy as np

from taurus_cv.models.retinanet.model.generator import get_generators
from taurus_cv.models.retinanet.model.visualization import draw_annotations, draw_boxes
from taurus_cv.models.retinanet.config import Config

config = Config('configRetinaNet.json')

generator, _, _, _ = get_generators(config.images_path, config.annotations_path, 1., 1, config.classes, 512,512, shuffle=False, transform=False)

cv2.namedWindow('Immagine', cv2.WINDOW_NORMAL)

i = 0
while i < generator.size():
    image = generator.load_image(i)
    annotations = generator.load_annotations(i)

    image, annotations = generator.random_transform_group_entry(image, annotations)

    image, image_scale = generator.resize_image(image)
    annotations[:, :4] *= image_scale

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    labels, boxes, anchors = generator.anchor_targets(image.shape, annotations, generator.num_classes())

    draw_boxes(image, anchors[np.max(labels, axis=1) == 1], (255, 255, 0), thickness=1)

    draw_annotations(image, annotations, color=(0, 0, 255), generator=generator)

    draw_boxes(image, boxes[np.max(labels, axis=1) == 1], (0, 255, 0))

    cv2.putText(image, generator.name_from_index(i), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow('Image', image)

    k = cv2.waitKey()
    if k == ord('q'):
        break
    if k == ord('+'):
        i = min(generator.size() - 1, i + 1)
    if k == ord('-'):
        i = max(0, i - 1)
