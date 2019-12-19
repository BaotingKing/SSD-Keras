#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/12/18 17:01
"""
    This is the script for pascal VOC
    Pascal VOC2012:(Contains the following files)
        Annotations: 17125's xml files, format is: *.xml        (2007~2012)
        ImageSets:  ????
        JPEGImages: 17125's origin images, format is: *.jpg     (2007~2012)
        SegmentationClass: 2913's Semantic segmentation         (2007~2011)
        SegmentationObject: 2913's Instance segmentation        (2007~2011)
        val_temp: 2010~2012 yeas's 6000 images
"""
import cv2
import os
import json
import pickle
import pandas as pd
import numpy as np
from collections import namedtuple
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import tensorflow as tf

original_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor', ]    # VOC has 20 class_num
class_need = ['car', 'bicycle', 'person', 'motorbike', 'bus', 'tvmonitor']
std_class_ind = {
    'car': '0',
    'bus': '1',
    'tvmonitor': '1',
    'bicycle': 'bike',
}

def read_voc_xml_to_txt():
    for parent, dirnames, filenames in os.walk(Original_xml_DIR):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            tree = ET.parse(full_path)
            root = tree.getroot()

            name = file_name[:-4]
            img_name = name + '.jpg'
            # img_path = os.path.join(img_dir, img_name)
            # img_size = cv2.imread(img_path).shape
            print('img_name', img_name)
            txt_path = TXT_DIR + name + '.txt'

            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                txt = open(txt_path, 'a')

                new_line = str(label) + ' ' + str(int(bbox.find('xmin').text)) + ' ' + str(
                    int(bbox.find('ymin').text)) + ' ' + str(int(bbox.find('xmax').text)) + ' ' + str(
                    int(bbox.find('ymax').text))
                txt.writelines(new_line)
                txt.write('\n')
                txt.close()
    print('to_txt_done')


def generate_xml(name, split_lines, img_size):
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('VOC')
    title.appendChild(title_text)
    annotation.appendChild(title)
    img_name = name + '.jpg'
    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
    source = doc.createElement('source')
    annotation.appendChild(source)
    title = doc.createElement('database')
    title_text = doc.createTextNode('The VOC Database')
    title.appendChild(title_text)
    source.appendChild(title)
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('VOC')
    title.appendChild(title_text)
    source.appendChild(title)
    size = doc.createElement('size')
    annotation.appendChild(size)
    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_need:
            object = doc.createElement('object')
            annotation.appendChild(object)
            title = doc.createElement('name')
            title_text = doc.createTextNode(std_class_ind[line[0]])
            title.appendChild(title_text)
            object.appendChild(title)
            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[1]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[2]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[3]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
    # 将DOM对象doc写入文件
    f = open(xml_dir + name + '.xml', 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def generate_json(name, split_lines, img_size):
    img_name = name + '.jpg'
    object_list = []
    for split_line in split_lines:
        line = split_line.strip().split()
        if line[0] in class_need:
            object_ = {
                "class": str(std_class_ind[line[0]]),
                "x1": float(line[1]),
                "y1": float(line[2]),
                "x2": float(line[3]),
                "y2": float(line[4])
            }
            object_list.append(object_)

    json_str_from_txt = {
        "img_name": img_name,
        "height": img_size[0],
        "width": img_size[1],
        "depth": img_size[2],

        "object": object_list
    }

    json_str = json.dumps(json_str_from_txt)

    with open(json_dir + name + '.json', 'w') as f:
        f.write(json_str)


def generate_voc_pkl():
    data_dict = {}
    for parent, dirnames, filenames in os.walk(Original_xml_DIR):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            tree = ET.parse(full_path)
            root = tree.getroot()

            img_name = root.find('filename').text
            img_size = root.find('size')
            width = float(img_size.find('width').text)
            height = float(img_size.find('height').text)

            bounding_boxes = []
            bounding_boxes_cls = []
            for obj in root.findall('object'):
                label_class = obj.find('name').text
                if label_class not in original_class:
                    continue
                bbox = obj.find('bndbox')
                box = [
                    float(bbox.find('xmin').text) / width,
                    float(bbox.find('ymin').text) / height,
                    float(bbox.find('xmax').text) / width,
                    float(bbox.find('ymax').text) / height
                ]
                bounding_boxes.append(box)
                one_hot_vector = [0] * len(original_class)
                idx = original_class.index(label_class)
                one_hot_vector[idx] = 1
                bounding_boxes_cls.append(one_hot_vector)

            if len(bounding_boxes) == 0 or len(bounding_boxes_cls) == 0:  # avoid no target
                continue
            obj_bboxes = np.asarray(bounding_boxes)
            obj_classes_id = np.asarray(bounding_boxes_cls)
            image_data = np.hstack((obj_bboxes, obj_classes_id))
            data_dict[img_name] = image_data
    print('[Info]: ----------', data_dict)
    pickle.dump(data_dict, open(Save_Path, 'wb'))
    print('Create VOC pkl is Ok!')


def generate_voc_tfrecord():
    whole_data_info = []
    for parent, dirnames, filenames in os.walk(Original_xml_DIR):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            tree = ET.parse(full_path)
            root = tree.getroot()

            img_name = root.find('filename').text
            img_size = root.find('size')
            width = float(img_size.find('width').text)
            height = float(img_size.find('height').text)

            for obj in root.findall('object'):
                label_class = obj.find('name').text
                if label_class not in original_class:
                    continue
                bbox = obj.find('bndbox')
                box = {
                    'x1': float(bbox.find('xmin').text),
                    'y1': float(bbox.find('ymin').text),
                    'x2': float(bbox.find('xmax').text),
                    'y2': float(bbox.find('ymax').text)
                }
                single_obj_value = [img_name,
                                    width,
                                    height,
                                    label_class,
                                    min(box['x1'], box['x2']),
                                    min(box['y1'], box['y2']),
                                    max(box['x1'], box['x2']),
                                    max(box['y1'], box['y2'])]
                whole_data_info.append(single_obj_value)

            column_attribute = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax',
                                'ymax']
            data_df = pd.DataFrame(whole_data_info, columns=column_attribute)
    print('[Info]: ----------', data_df)
    if False:
        grouped = split(data_df, 'filename')
        writer = tf.python_io.TFRecordWriter(output_path)
        for group in grouped:
            tf_example = create_tf_example(group, TRAIN_IMG_PATH)
            writer.write(tf_example.SerializeToString())
        writer.close()
        output_path = os.path.join(os.getcwd(), output_path)

    print('Successfully created the TFRecords: {}'.format(output_path))
    print('Create VOC TfRecord is Ok!')


def test_pkl(train_img_path, pkl_path):
    gt = pickle.load(open(pkl_path, 'rb'))
    keys = sorted(gt.keys())

    class_name_np = np.array(original_class)
    for key in keys:
        img_path = os.path.join(train_img_path, key)
        # img = imread(img_path).astype('float32')
        img = cv2.imread(img_path)

        img_size = img.copy()
        img_shape = img.shape[:2]
        WIDTH = img.shape[1]
        HEIGHT = img.shape[0]
        resize_shape = (300, 300)
        img_size = cv2.resize(img_size, resize_shape)

        for bbox in gt[key]:
            class_mask = bbox[4:] == 1
            class_name = class_name_np[class_mask][0]
            x1 = int(bbox[0] * WIDTH)
            y1 = int(bbox[1] * HEIGHT)
            x2 = int(bbox[2] * WIDTH)
            y2 = int(bbox[3] * HEIGHT)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(img, 'C: ' + str(class_name), (int(x1), int(y1) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # if show_bdd_gt:
            #     for obj in img_dict[key]:
            #         if obj['category'] in CLASS_NEED.keys():
            #             left_top_point = (int(obj['box2d']['x1']) + 2, int(obj['box2d']['y1']))
            #             right_down_point = (int(obj['box2d']['x2']) + 2, int(obj['box2d']['y2']))
            #             if CLASS_NEED[obj['category']] == 0:
            #                 cv2.rectangle(img, left_top_point, right_down_point, (0, 255, 0), thickness=2)
            #             else:
            #                 cv2.rectangle(img, left_top_point, right_down_point, (200, 255, 255), thickness=2)

            x1 = int(bbox[0] * resize_shape[0])
            y1 = int(bbox[1] * resize_shape[1])
            x2 = int(bbox[2] * resize_shape[0])
            y2 = int(bbox[3] * resize_shape[1])
            cv2.rectangle(img_size, (x1, y1), (x2, y2), (255, 255, 0), thickness=2)
            cv2.putText(img_size, 'C: ' + str(class_name), (int(x1), int(y1) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        jion_img_size = (max(img_shape[0], resize_shape[0]), img_shape[1] + resize_shape[1], 3)
        jion_img = np.zeros(jion_img_size).astype('uint8')
        jion_img[:img_shape[0], :img_shape[1], :] = img
        jion_img[:resize_shape[0]:, img_shape[1]:, :] = img_size[:, :, :]
        print('-----------------', key)
        cv2.imshow('orgin', jion_img)
        # cv2.imshow('img', img)
        # cv2.imshow('img_size', img_size)
        cv2.waitKey(0)


if __name__ == '__main__':
    print('This is a special script for processing Pascal VOC data......')
    Img_DIR = '/eDisk/FCWS_dataset/Pascal_VOC/VOC2012/JPEGImages/'
    Original_xml_DIR = '/eDisk/FCWS_dataset/Pascal_VOC/VOC2012/Annotations'
    Save_Path = 'VOC_2012.pkl'
    trans_format = 'pkl'
    if trans_format == 'pkl':
        generate_voc_pkl()
        # test_pkl(Img_DIR, Save_Path)
    elif trans_format == 'tfrecord':
        # 待调试 2019-12-19
        generate_voc_tfrecord()
    elif trans_format == 'json' or trans_format == 'xml':
        TXT_DIR = '/home/ulsee_lee/ULsee/Annotations/txt/'  # 保存txt格式的文件夹地址
        xml_dir = '/home/ulsee_lee/ULsee/Annotations/xml/'  # 保存xml格式文件的文件夹地址
        json_dir = '/home/ulsee_lee/ULsee/Annotations/json/'  # 保存json格式文件的文件夹地址
        read_voc_xml_to_txt()    # generator txt of xml
        need_save_json = True
        need_save_xml = True
        for parent, dirnames, filenames in os.walk(TXT_DIR):  # 分别得到根目录，子目录和根目录下文件
            for file_name in filenames:
                full_path = os.path.join(parent, file_name)  # 获取文件全路径
                f = open(full_path)
                split_lines = f.readlines()
                #         print(split_lines)
                name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
                img_name = name + '.jpg'
                img_path = os.path.join(Img_DIR, img_name)
                img_size = cv2.imread(img_path).shape

                if need_save_json:
                    generate_json(name, split_lines, img_size)
                if need_save_xml:
                    generate_xml(name, split_lines, img_size)

    print('The VOC data processes is Ok!')
