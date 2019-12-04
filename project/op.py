#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/11/19 15:13
import tensorflow as tf
import os
import math
import time
from keras.applications.imagenet_utils import preprocess_input
import cv2
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import imread
import tensorflow as tf

# from model.ssd300VGG16 import SSD
from model.ssd300MobileNetV2Lite import SSD
# from model.ssd300Mv2Lite_Pro import SSD
from ssd_utils import BBoxUtility


def generate_txt(image_name, boxes, boxes_infor, shape):
    print(image_name)
    txt_name = image_name[:-4] + '.txt'
    if not os.path.exists(Result_PATH):
        os.makedirs(Result_PATH)
    img_h, img_w, img_c = shape
    with open(os.path.join(Result_PATH, txt_name), 'w') as f:
        if len(boxes) != 0:
            for idx in range(len(boxes[0])):
                x1 = str(boxes[0][idx] * img_w) + ' '
                y1 = str(boxes[1][idx] * img_h) + ' '
                x2 = str(boxes[2][idx] * img_w) + ' '
                y2 = str(boxes[3][idx] * img_h)
                classes = CLASS_2_ID[int(boxes_infor[0][idx]) - 1] + ' '
                prob = str(boxes_infor[1][idx]) + ' '
                obj_w = [classes, prob, x1, y1, x2, y2]
                f.writelines(obj_w)
                f.write('\n')
        else:
            f.write('\n')


def sigmoid_self(x):
    min_value = -40.0
    max_value = 40.0
    if x < min_value:
        x = min_value
    if x > max_value:
        x = max_value
    return 1.0 / (1 + math.exp(-1.0 * x))


def sigmoid_list_self(x):
    sigmoid_list = list()
    for each_elem in x:
        sigmoid_list.append(sigmoid_self(each_elem))
    return sigmoid_list


def softmax_self(y: list()):
    min_value = -40.0
    max_value = 40.0
    y_softmax = list()
    sum_exp = 0
    for each_elem in y:
        t = each_elem
        if t < min_value:
            t = min_value
        if t > max_value:
            t = max_value
        sum_exp += math.exp(t)
    for each_elem in y:
        t = each_elem
        if t < min_value:
            t = min_value
        if t > max_value:
            t = max_value
        y_softmax.append(math.exp(t) / sum_exp)
    return y_softmax


def get_max(box_confidence, box_classes_prob):
    confidence = list()
    max_value = 0
    max_index = 0
    for index, prob in enumerate(box_classes_prob):
        t = box_confidence * prob
        if t > max_value:
            max_value = t
            max_index = index
    return max_value, max_index


def get_coord_confidence(x, anchors, size=(13, 13), score_threshold=0.85, anchors_num=3, classes_num=7):
    box_dimen = anchors_num * (5 + classes_num)
    row_num = size[0]
    column_num = size[1]
    index_start = 0
    index_interval = 5 + classes_num
    list_filters = list()
    for each_row in range(row_num):
        for each_cols in range(column_num):
            for each_anchor in range(anchors_num):
                each_box_info = x[index_start:(index_start + index_interval)]
                box_x = each_box_info[0]
                box_x = sigmoid_self(box_x)
                box_x += each_cols
                box_x /= column_num
                box_y = each_box_info[1]
                box_y = sigmoid_self(box_y)
                box_y += each_row
                box_y /= row_num
                box_w = each_box_info[2]
                box_w = math.exp(box_w) * anchors[each_anchor][0] / 352
                box_h = each_box_info[3]
                box_h = math.exp(box_h) * anchors[each_anchor][1] / 352
                box_confidence = each_box_info[4]
                # print('zz', each_row, each_cols, box_confidence)
                box_confidence = sigmoid_self(box_confidence)
                box_class_prob = each_box_info[5:]
                box_class_prob = sigmoid_list_self(box_class_prob.tolist())
                max_value, max_index = get_max(box_confidence, box_class_prob)
                box_tx = box_x - box_w / 2
                box_ty = box_y - box_h / 2
                box_bx = box_x + box_w / 2
                box_by = box_y + box_h / 2
                temp_list = [box_tx, box_ty, box_bx, box_by, max_value, max_index]
                if max_value > score_threshold:
                    list_filters.append(temp_list)

                index_start += index_interval
    print("finish")
    return list_filters


def del_element(x, index):
    offset = 0
    for i in index:
        del x[i - offset]
        offset += 1
    return x


def nms(x_sets, iou_threshold=0.5):
    out_list = list()
    while True:
        if len(x_sets) == 0:
            break
        else:
            max_confidence = x_sets[0]
            out_list.append(max_confidence)
            max_c_tx = max_confidence[0]
            max_c_ty = max_confidence[1]
            max_c_bx = max_confidence[2]
            max_c_by = max_confidence[3]
            max_area = (max_c_bx - max_c_tx) * (max_c_by - max_c_ty)
            del x_sets[0]
            boxes_num = len(x_sets)
            del_index_list = list()
            for i in range(boxes_num):
                tx = x_sets[i][0]
                ty = x_sets[i][1]
                bx = x_sets[i][2]
                by = x_sets[i][3]
                dist_area = (bx - tx) * (by - ty)
                max_tx = 0
                max_ty = 0
                min_bx = 0
                min_by = 0
                if max_c_tx >= tx:
                    max_tx = max_c_tx
                else:
                    max_tx = tx
                if max_c_ty >= ty:
                    max_ty = max_c_ty
                else:
                    max_ty = ty

                if max_c_bx >= bx:
                    min_bx = bx
                else:
                    min_bx = max_c_bx
                if max_c_by >= by:
                    min_by = by
                else:
                    min_by = max_c_by
                if max_tx < min_bx and max_ty < min_by:
                    intersect_area = (min_by - max_ty) * (min_bx - max_tx)
                else:
                    intersect_area = 0
                # intersect_area = (min_by-max_ty)*(min_bx-max_tx)
                # if intersect_area < 0:
                #     intersect_area = 0
                iou = intersect_area / (max_area + dist_area - intersect_area)
                if iou > iou_threshold:
                    del_index_list.append(i)
                print(i)
            x_sets = del_element(x_sets, del_index_list)
    return out_list


def get_output_by_nms(boxes, iou_threshold=0.5):
    out_list = list()
    classes_index = []
    for each_box in boxes:
        index_classes = each_box[5]
        if index_classes not in classes_index:
            classes_index.append(index_classes)
    for index in classes_index:
        temp_list = list()
        for each_box in boxes:
            index_classes = each_box[5]
            if index_classes == index:
                temp_list.append(each_box)
        temp_list = sorted(temp_list, key=lambda x: x[4], reverse=True)
        out_list.extend(nms(temp_list, iou_threshold=iou_threshold))
    return out_list


if __name__ == '__main__':
    voc_classes = ['car', 'motor']
    CLASS_2_ID = dict(zip(range(len(voc_classes)), voc_classes))
    NUM_CLASSES = 1 + 2
    MODEL_PATH = "../weights/checkpoints/l_test/weights.200-2.35_2.50.hdf5"

    Result_PATH = '../mAP/input/detection-results'

    input_shape = (300, 300, 3)
    model = SSD(input_shape, num_classes=NUM_CLASSES)
    model.load_weights(filepath=MODEL_PATH, by_name=True)

    bbox_util = BBoxUtility(NUM_CLASSES)

    val_img_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/val/'
    for parent, dirnames, filenames in os.walk(val_img_path):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            # file_name = 'c59500f0-602e9446.jpg'
            img_path = os.path.join(parent, file_name)  # 获取文件全路径
            print(img_path)
            if file_name[-4:] in ['.jpg', '.png']:
                img_origin = cv2.imread(img_path)
                img = cv2.resize(img_origin, input_shape[:2]).astype('float32')
            else:
                continue

            start_time = time.clock()
            inputs = preprocess_input(np.array([img]))
            preds = model.predict(inputs, batch_size=1, verbose=1)
            end_time = time.clock()
            print('Take time:', end_time - start_time)
            results = bbox_util.detection_out(preds)
            result = results[0]  # It is only one image

            # Parse the outputs.
            det_label = result[:, 0]
            det_conf = result[:, 1]

            det_xmin = result[:, 2]
            det_ymin = result[:, 3]
            det_xmax = result[:, 4]
            det_ymax = result[:, 5]

            # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if 0.5 <= conf]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            # boxes_loc = np.concatenate((top_xmin, top_ymin, top_xmax, top_ymax))
            boxes_loc = [top_xmin, top_ymin, top_xmax, top_ymax]
            boxes_infor = [top_label_indices, top_conf]
            generate_txt(file_name, boxes_loc, boxes_infor, img_origin.shape)

            # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            # plt.imshow(img / 255.)
            # currentAxis = plt.gca()
            #
            # for i in range(top_conf.shape[0]):
            #     xmin = int(round(top_xmin[i] * img.shape[1]))
            #     ymin = int(round(top_ymin[i] * img.shape[0]))
            #     xmax = int(round(top_xmax[i] * img.shape[1]))
            #     ymax = int(round(top_ymax[i] * img.shape[0]))
            #     score = top_conf[i]
            #     label = int(top_label_indices[i])
            #     label_name = voc_classes[label - 1]
            #     display_txt = '{:0.2f}, {}'.format(score, label_name)
            #     coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            #     color = colors[label]
            #     currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            #     currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
