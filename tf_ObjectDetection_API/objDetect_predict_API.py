#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/12 11:56
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time

import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

sys.path.append("..")

# Grab path to current working directory
# CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
# PATH_TO_CKPT = os.path.join('/eDisk/Zack_SSD_mv2/003_ssdmv2','frozen_inference_graph.pb')
# Path to label map file
# PATH_TO_LABELS = os.path.join(CWD_PATH,'my_training_mnv2_ssd','my_data_label_map.pbtxt')


# Number of classes the object detector can identify
NUM_CLASSES = 2
MY_DATA_LABELS = {
    0: 'non',
    1: 'Car',
    2: 'Bus'
}


def plt_bboxes(img, rshape, classes, scores, bboxes, figsize=(50, 50), linewidth=5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    img_ = img[:, :, [2, 1, 0]]
    plt.imshow(img_)
    height = rshape[0]
    width = rshape[1]
    colors = dict()
    for i in range(len(classes)):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i][0] * height)
            xmin = int(bboxes[i][1] * width)
            ymax = int(bboxes[i][2] * height)
            xmax = int(bboxes[i][3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            #             class_name = str(MY_DATA_LABELS[cls_id])
            class_name = str(MY_DATA_LABELS[cls_id])

            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=30, color='white')
    plt.show()


def generate_txt(image_name, bboxes, boxes_infor, shape):
    print(image_name)
    txt_name = image_name[:-4] + '.txt'
    if not os.path.exists(Result_PATH):
        os.makedirs(Result_PATH)
    img_h, img_w, img_c = shape
    with open(os.path.join(Result_PATH, txt_name), 'w') as f:
        if len(bboxes) != 0:
            for idx, box in enumerate(bboxes):
                x1 = str(box[1] * img_w) + ' '
                y1 = str(box[0] * img_h) + ' '
                x2 = str(box[3] * img_w) + ' '
                y2 = str(box[2] * img_h)
                classes = MY_DATA_LABELS[int(boxes_infor[0][idx])] + ' '
                prob = str(boxes_infor[1][idx]) + ' '
                obj_w = [classes, prob, x1, y1, x2, y2]
                f.writelines(obj_w)
                f.write('\n')
        else:
            obj_w = ['-1 ', '0 ', '0 ', '0 ', '0 ', '0 ']
            f.writelines(obj_w)
            f.write('\n')


if __name__ == '__main__':
    PATH_TO_CKPT = os.path.join('/eDisk/Zack_SSD_mv2/007_ssdmv2','frozen_inference_graph.pb')    # for object detection.
    detection_graph = tf.Graph()
    Result_PATH = '/home/zack/studio/Github/ssd_kerasV2/mAP/input/detection-results'
    # IMAGE_PATH = '/eDisk/Merge_Train_Dataset/Val_Dataset'
    IMAGE_PATH = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/val/'
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()

        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    all_total_time = []
    select_threshold = 0.85

    for parent, dirnames, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            full_filename = os.path.join(parent, filename)
            img = cv2.imread(full_filename)
            image_expanded = np.expand_dims(img, axis=0)

            start = time.time()
            print('start time:', start)
            boxes, scores, classes, num = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            elapsed = (time.time() - start)
            all_total_time.append(elapsed)

            print('one img total time:', elapsed)

            rshape = img.shape
            rlabels_ = np.squeeze(classes).astype(np.int32)
            rscores_ = np.squeeze(scores)
            rbboxes_ = np.squeeze(boxes)
            pass_num = [i for i, x in enumerate(rscores_) if x >= select_threshold]

            rlabels = []
            rscores = []
            rbboxes = []
            for j in pass_num:
                rlabels.append(rlabels_[j])
                rscores.append(rscores_[j])
                rbboxes.append(rbboxes_[j])

            print('--------------------', filename, type(rlabels), type(rscores), type(rbboxes))
            print(rlabels, rscores, rbboxes)
            generate_txt(filename, rbboxes, [rlabels, rscores], rshape)
            # plt_bboxes(img, rshape, rlabels, rscores, rbboxes)

    print('-----------end')
