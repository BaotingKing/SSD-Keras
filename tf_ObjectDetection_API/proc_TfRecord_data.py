#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/10/12 11:56
import cv2
import os
import io
import numpy as np
import pandas as pd
import json

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

CLASS_NEED = {
    'car': 0,
    'Car': 0,
    'bus': 1,
    'truck': 1,
    'Truck': 1,
    'Van': 1,
}


def class_text_to_int(row_label):
    if row_label == 'car' or row_label == 'Car':
        return 1
    elif row_label == 'bus' or row_label == 'Bus':
        return 2
    elif row_label == 'truck' or row_label == 'Truck':
        return 2
    elif row_label == 'van' or row_label == 'Van':
        return 2
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')

    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes), }))
    return tf_example


class TFRecord_preprocessor(object):
    def __init__(self, json_path, image_path, data_type='train', lable_type='sigle'):
        self.json_path = json_path
        self.data_type = data_type

        self.path_prefix = image_path
        # self.num_classes = CLASS_NUM
        self.data = dict()
        if lable_type == 'BDD':
            self._preprocess_BDD100K_json()
        elif lable_type == 'Cityscape':
            self._preprocess_Cityscapes_jsons()
        elif lable_type == 'KITTY':
            self._preprocess_Kitty_txt()
        elif lable_type == 'Udacity':
            self._preprocess_Udacity_txt()
        else:
            print('[Warning]: No dataset!')

    def _preprocess_BDD100K_json(self):
        """
            For BDD100K dataset
            All images's labels in one json file.
        """
        filenames = os.listdir(self.json_path)
        for filename in filenames:
            if filename[-5:] == ".json" and self.data_type in filename:
                filename_path = os.path.join(self.json_path, filename)
                with open(filename_path) as f:
                    labels_data_json = json.load(f)

                    whole_data_info = []
                    for item_dict in labels_data_json:
                        name = item_dict['name']
                        img_path = os.path.join(TRAIN_IMG_PATH, name)
                        img = cv2.imread(img_path)

                        BDD_width = img.shape[1]
                        BDD_hight = img.shape[0]

                        labels = item_dict['labels']
                        for k in labels:
                            if k['category'] in CLASS_NEED.keys():
                                bbox = k['box2d']
                                single_obj_value = [name,
                                                    BDD_width,
                                                    BDD_hight,
                                                    k['category'],
                                                    min(bbox['x1'], bbox['x2']),
                                                    min(bbox['y1'], bbox['y2']),
                                                    max(bbox['x1'], bbox['x2']),
                                                    max(bbox['y1'], bbox['y2'])]
                                whole_data_info.append(single_obj_value)

                column_attribute = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                self.data = pd.DataFrame(whole_data_info, columns=column_attribute)

    def _preprocess_Cityscapes_jsons(self):
        """
            For Cityscape dataset
            All images's labels in their respective json files
        """
        label_file_path = os.path.join(self.json_path, self.data_type)
        whole_data_info = []
        for parent, dirnames, filenames in os.walk(label_file_path):  # 分别得到根目录，子目录和根目录下文件
            for filename in filenames:
                if filename[-5:] == ".json":
                    json_file_path = os.path.join(parent, filename)  # 获取文件全路径
                    with open(json_file_path) as f:
                        labels_data_json = json.load(f)

                        img_width = labels_data_json['imgWidth']
                        img_heigh = labels_data_json['imgHeight']
                        bounding_boxes = []
                        bounding_boxes_cls = []
                        name_id = filename.replace('_gtFine_polygons.json', '')
                        image_name = name_id + '_leftImg8bit.png'
                        for obj in labels_data_json['objects']:
                            label_class = obj['label']
                            if label_class in CLASS_NEED.keys():
                                polygon_set = obj['polygon']
                                x_, y_ = [], []
                                for point in polygon_set:
                                    x_.append(point[0])
                                    y_.append(point[1])

                                single_obj_value = [image_name,
                                                    img_width,
                                                    img_heigh,
                                                    label_class,
                                                    min(x_),
                                                    min(y_),
                                                    max(x_),
                                                    max(y_)]
                                whole_data_info.append(single_obj_value)

            column_attribute = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax',
                                'ymax']
            self.data = pd.DataFrame(whole_data_info, columns=column_attribute)

    def _preprocess_Kitty_txt(self):
        whole_data_info = []
        for parent, dirnames, filenames in os.walk(self.json_path):  # 分别得到根目录，子目录和根目录下文件
            for file_name in filenames:
                if file_name[-4:] == '.txt':  # KITTY dataset's labels
                    label_path = os.path.join(parent, file_name)  # 获取文件全路径
                    name = file_name[:-4] + '.png'  # KITTY's type
                    img_path = os.path.join(self.path_prefix, name)

                    if os.path.exists(label_path) and os.path.exists(img_path):
                        img_size = cv2.imread(img_path).shape
                        img_heigh = img_size[0]
                        img_width = img_size[1]
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
                                    single_obj_value = [name,
                                                        img_width,
                                                        img_heigh,
                                                        label_class,
                                                        min(box['x1'], box['x2']),
                                                        min(box['y1'], box['y2']),
                                                        max(box['x1'], box['x2']),
                                                        max(box['y1'], box['y2'])]
                                    whole_data_info.append(single_obj_value)

        column_attribute = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax',
                            'ymax']
        self.data = pd.DataFrame(whole_data_info, columns=column_attribute)

    def _preprocess_Udacity_txt(self):
        whole_data_info = []
        for parent, dirnames, filenames in os.walk(self.json_path):  # Udacity's label in txt file
            for file_name in filenames:
                if file_name[-4:] == '.txt':  # Udacity dataset's labels
                    label_path = os.path.join(parent, file_name)
                    name = file_name[:-4] + '.jpg'  # Udacity's type
                    img_path = os.path.join(self.path_prefix, name)

                    if os.path.exists(label_path) and os.path.exists(img_path):
                        img_size = cv2.imread(img_path).shape
                        img_heigh = img_size[0]
                        img_width = img_size[1]
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
                                    single_obj_value = [name,
                                                        img_width,
                                                        img_heigh,
                                                        label_class,
                                                        min(box['x1'], box['x2']),
                                                        min(box['y1'], box['y2']),
                                                        max(box['x1'], box['x2']),
                                                        max(box['y1'], box['y2'])]
                                    whole_data_info.append(single_obj_value)

        column_attribute = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax',
                            'ymax']
        self.data = pd.DataFrame(whole_data_info, columns=column_attribute)


def test_tfRecord():
    for g in bdd_data_df.groupby('filename'):
        img_name = g[0]
        img_labels = g[1]  # It is DataFrame type

        img_path = os.path.join(TRAIN_IMG_PATH, img_name)
        if not os.path.exists(img_path):
            continue
        img_origin = cv2.imread(img_path)
        img_shape = img_origin.shape

        for idx, row in img_labels.iterrows():
            if row['width'] != img_shape[1] and row['height'] != img_shape[0]:
                print('[Error]: The image {0} width or height is error!!!'.format(img_name))
            cv2.rectangle(img_origin, (int(row['xmin']), int(row['ymin'])),
                          (int(row['xmax']), int(row['ymax'])),
                          color=(225, 220, 100), thickness=1)
            cv2.putText(img_origin, 'C: ' + str(row['class']), (int(row['xmin']), int(row['ymin']) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('display', img_origin)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)


if __name__ == '__main__':
    run_flag = 2
    TRAIN_IMG_PATH = '/eDisk/Merge_Train_Dataset/Train_Dataset'
    if run_flag == 0:
        json_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/'
        TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/train_merge/'  # merge train and val images
        for idx in ['train', 'val']:
            if idx == 'train':
                output_path = 'BDD_train.record'
            elif idx == 'val':
                output_path = 'BDD_val.record'
            data_df = TFRecord_preprocessor(json_path, TRAIN_IMG_PATH, idx, lable_type='BDD').data

            print(data_df)
            grouped = split(data_df, 'filename')

            writer = tf.python_io.TFRecordWriter(output_path)

            for group in grouped:
                tf_example = create_tf_example(group, TRAIN_IMG_PATH)
                writer.write(tf_example.SerializeToString())

            writer.close()

            output_path = os.path.join(os.getcwd(), output_path)
            print('Successfully created the TFRecords: {}'.format(output_path))
    elif run_flag == 1:
        output_path = './merge_data_train.record'
        json_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/'
        # TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/train_merge/'  # merge train and val images
        bdd_data_df = TFRecord_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='BDD').data

        json_path = '/eDisk/FCWS_dataset/Cityscape/cityscaps_label/gtFine/'
        # TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/Cityscape/cityscaps/leftImg8bit'
        cityscape_data_df = TFRecord_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='Cityscape').data

        json_path = '/eDisk/FCWS_dataset/Udacity/object_dataset/label_txt'
        # TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/Cityscape/cityscaps/leftImg8bit'
        Udacity_data_df = TFRecord_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='Udacity').data

        print(bdd_data_df, cityscape_data_df)
        merge_data = pd.concat([bdd_data_df, cityscape_data_df, Udacity_data_df])
        print(merge_data)
        grouped = split(merge_data, 'filename')

        writer = tf.python_io.TFRecordWriter(output_path)

        for group in grouped:
            tf_example = create_tf_example(group, TRAIN_IMG_PATH)
            writer.write(tf_example.SerializeToString())

        writer.close()

        output_path = os.path.join(os.getcwd(), output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))
    elif run_flag == 2:
        """generator and test different dataset"""
        json_path = '/eDisk/FCWS_dataset/Cityscape/cityscaps_label/gtFine/'
        # TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/Cityscape/cityscaps/leftImg8bit'
        data_df = TFRecord_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='Cityscape').data

        for g in data_df.groupby('filename'):
            img_name = g[0]
            img_labels = g[1]  # It is DataFrame type

            img_path = os.path.join(TRAIN_IMG_PATH, img_name)
            if not os.path.exists(img_path):
                continue
            img_origin = cv2.imread(img_path)
            img_shape = img_origin.shape

            for idx, row in img_labels.iterrows():
                if row['width'] != img_shape[1] and row['height'] != img_shape[0]:
                    print('[Error]: The image {0} width or height is error!!!'.format(img_name))
                cv2.rectangle(img_origin, (int(row['xmin']), int(row['ymin'])),
                              (int(row['xmax']), int(row['ymax'])),
                              color=(225, 220, 100), thickness=1)
                cv2.putText(img_origin, 'C: ' + str(row['class']), (int(row['xmin']), int(row['ymin']) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print('---===: ', img_name)
            cv2.imshow('display', img_origin)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.waitKey(0)
            else:
                break

