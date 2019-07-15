import sys
import argparse
sys.path.append('../../..')

import matplotlib as mpl

mpl.use('Agg')

import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from taurus_cv.models.retinanet.model.pascal_voc import save_annotations
from taurus_cv.models.retinanet.model.image import read_image_bgr, preprocess_image, resize_image, read_image_rgb
from taurus_cv.models.retinanet.config import Config
from taurus_cv.models.fsaf.networks.retinanet import retinanet as retinanet
from taurus_cv.models.fsaf.config import current_config as config2

start_time = time.time()

config = Config('configRetinaNet.json')

wname = 'BASE'
wpath = config.base_weights_path
classes = ['0', '1', '2', '3', '4', '5', '6', '7']

if os.path.isfile(config.trained_weights_path):
    wname = "definitivi"
    wpath = config.trained_weights_path
    classes = config.classes
if os.path.isfile(config.pretrained_weights_path):
    wname = 'pretrained'
    wpath = config.pretrained_weights_path
    classes = config.classes

model = retinanet(config2)
model.load_weights(config2.retinanet_weights)

print("backend: ", config.type)

if os.path.isfile(wpath):
    model.load_weights(wpath, by_name=True, skip_mismatch=True)
    print("权重" + wname)
else:
    print("权重None")

start_index = config.test_start_index
font = cv2.FONT_HERSHEY_SIMPLEX

def random_color():
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r,g,b)

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

        _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))
        # print(detections[0][0][:4], detections[0][0][4:])
        # exit()

        # bbox要取到边界内
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(img.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(img.shape[0], detections[:, :, 3])

        detections[0, :, :4] /= scale

        scores = detections[0, :, 4:]

        # 推测置信度
        indices = np.where(detections[0, :, 4:] >= 0.3)

        scores = scores[indices]

        scores_sort = np.argsort(-scores)[:100]

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
            save_annotations(config.test_result_path, "{0:08d}".format(start_index + nimage), orig_image, boxes)

        else:
            # colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist()
            # plt.imshow(orig_image)
            # current_axis = plt.gca()

            orig_image = cv2.imread(imgfp)
            show_img = orig_image.copy()

            # color = random_color()
            color = (0, 255, 255)

            # plt.gca().add_patch(plt.Rectangle(xy=(cat_dict['bbox'][i][1], cat_dict['bbox'][i][0]),
            #                                   width=cat_dict['bbox'][i][3] - cat_dict['bbox'][i][1],
            #                                   height=cat_dict['bbox'][i][2] - cat_dict['bbox'][i][0],
            #                                   edgecolor=[c / 255 for c in label_colors[cat_idx]],
            #                                   fill=False, linewidth=2))

            if len(image_boxes) > 0:
                for i, box in enumerate(image_boxes):
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])

                    # color = colors[i % len(colors)]
                    # label = '{}: {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0])
                    # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=1)) #use
                    # current_axis.text(xmin, ymin, label, size='1', color='white', bbox={'facecolor': color, 'alpha': 1.0})
                    # current_axis.text(xmin, ymin, classes[image_predicted_labels[i]], size='1', color='white')
                    # current_axis.text(xmin, ymin-1, '{} {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0]), size='10', color=color) # use

                    show_img = cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), color, 1)
                    show_img = cv2.putText(show_img, '{} {:.2f}'.format(classes[image_predicted_labels[i]], image_scores[i][0]), (xmin, ymin-2), font, 0.5, color, 1)

            cv2.imwrite(os.path.join(config.test_result_path, imgf), show_img)
            # plt.savefig(os.path.join(config.test_result_path, imgf))
            # plt.close()

        print("生成图片 '" + imgf + "'" + ' time:{}'.format(time.time() - start_time))
        start_time = time.time()

