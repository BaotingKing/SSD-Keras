#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/12/04 11:17
import os
import cv2
import json

CLASS_NEED = {
    'car': 0,
    'Car': 0,
    'bus': 1,
    'truck': 1,
    'Truck': 1,
    'Van': 1,
}


def search_file(rootdir, target_file):
    target_file_path = None
    for parent, dirnames, filenames in os.walk(rootdir):
        if target_file in filenames:
            target_file_path = os.path.join(parent, target_file)
            break
    return target_file_path


def single_img_GT_show(image_name, data_type=''):
    """
        根据图片名找到对应标签信息并显示到图片上
    """
    whole_data_info = []
    if data_type == 'BDD':
        label_path = LABLE_PATH
        if label_path[-5:] == ".json" and os.path.exists(label_path):
            with open(label_path) as f:
                labels_data_info = json.load(f)
            for item_dict in labels_data_info:
                if image_name == item_dict['name']:
                    labels = item_dict['labels']
                    for k in labels:
                        if k['category'] in CLASS_NEED.keys():
                            bbox = k['box2d']
                            single_obj_value = [k['category'],
                                                min(bbox['x1'], bbox['x2']),
                                                min(bbox['y1'], bbox['y2']),
                                                max(bbox['x1'], bbox['x2']),
                                                max(bbox['y1'], bbox['y2'])]
                            whole_data_info.append(single_obj_value)
                    break
    elif data_type == 'Cityscape':
        label_name = image_name.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        label_path = search_file(LABLE_PATH, label_name)
        if os.path.exists(label_path):
            with open(label_path) as f:
                labels_data_json = json.load(f)

            for obj in labels_data_json['objects']:
                label_class = obj['label']
                if label_class in CLASS_NEED.keys():
                    polygon_set = obj['polygon']
                    x_, y_ = [], []
                    for point in polygon_set:
                        x_.append(point[0])
                        y_.append(point[1])

                    single_obj_value = [label_class,
                                        min(x_),
                                        min(y_),
                                        max(x_),
                                        max(y_)]
                    whole_data_info.append(single_obj_value)
    elif data_type == 'KITTY':
        label_name = image_name.replace('.png', '.txt')
        label_path = search_file(LABLE_PATH, label_name)    # KITTY dataset's labels is independent
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                split_lines = f.readlines()

                for split_line in split_lines:
                    line = split_line.strip().split()
                    label_class = line[0]
                    if label_class in CLASS_NEED.keys():
                        box = {
                            'x1': float(line[4]),
                            'y1': float(line[5]),
                            'x2': float(line[6]),
                            'y2': float(line[7])
                        }
                        single_obj_value = [label_class,
                                            min(box['x1'], box['x2']),
                                            min(box['y1'], box['y2']),
                                            max(box['x1'], box['x2']),
                                            max(box['y1'], box['y2'])]
                        whole_data_info.append(single_obj_value)
    elif data_type == 'Udacity':
        label_name = image_name.replace('.jpg', '.txt')
        label_path = search_file(LABLE_PATH, label_name)    # Udacity dataset's labels is independent
        with open(label_path, 'r') as f:
            split_lines = f.readlines()

            for split_line in split_lines:
                line = split_line.strip().split()
                label_class = line[0]
                if label_class in CLASS_NEED.keys():
                    box = {
                        "x1": float(line[1]),
                        "y1": float(line[2]),
                        "x2": float(line[3]),
                        "y2": float(line[4])
                    }
                    single_obj_value = [label_class,
                                        min(box['x1'], box['x2']),
                                        min(box['y1'], box['y2']),
                                        max(box['x1'], box['x2']),
                                        max(box['y1'], box['y2'])]
                    whole_data_info.append(single_obj_value)
    else:
        label_name = image_name.replace('.png', '.txt')
        label_path = os.path.join(LABLE_PATH, label_name)
        print('[Warning]: No dataset!')

    img_path = os.path.join(TRAIN_IMG_PATH, image_name)
    if not os.path.exists(img_path):
        print('[Warning]: Can not find the image {0}'.format(image_name))
    img_origin = cv2.imread(img_path)

    for obj in whole_data_info:
        cv2.rectangle(img_origin, (int(obj[1]), int(obj[2])),
                      (int(obj[3]), int(obj[4])),
                      color=(225, 220, 100), thickness=1)
        cv2.putText(img_origin, 'C: ' + str(obj[0]), (int(obj[1]), int(obj[2]) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print('---===: ', image_name)
    cv2.imshow('display', img_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    TRAIN_IMG_PATH = '/eDisk/Merge_Train_Dataset/Train_Dataset'
    LABLE_PATH = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/bdd100k_labels_images_train.json'
    LABLE_PATH = '/eDisk/FCWS_dataset/Cityscape/cityscaps_label/gtFine/train'
    LABLE_PATH = '/eDisk/FCWS_dataset/KITTI/training_label/label_2/'    # KITTY
    LABLE_PATH = '/eDisk/FCWS_dataset/Udacity/object_dataset/label_txt'    # Udacity
    img_name = '1478897450524944901.jpg'
    single_img_GT_show(img_name, data_type='Udacity')

