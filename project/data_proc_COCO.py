#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: ZK
# Time: 2019/12/19 16:16
"""
    This is the script for COCO-2017 dataset
    COCO-2017:(Contains the following files)
        train&val:
            annotations:  ***.json
            train2017: 118,287's images, format: *.jpg
            val2017:     5,000's images, format: *.jpg
        test:
            annotations:  ***.json
            test2017:   40,670's images, format: *.jpg
"""
import cv2
import os
import json
import pickle
import numpy as np
from xml.dom.minidom import Document

classes = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
           7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
           13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
           18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
           24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
           32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
           37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
           41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
           46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
           52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
           57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
           63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
           72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
           78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
           84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
           89: 'hair drier', 90: 'toothbrush'}

class_need = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
std_class_ind = {'car': 'car',
                 'truck': 'truck',
                 'bicycle': 'bike',
                 'person': 'person',
                 'motorcycle': 'bike',
                 'bus': 'bus',
                 }


def read_json_to_txt(json_label_path, txt_dir):
    with open(json_label_path, 'r') as f:
        data = json.load(f)

    img_info_dic = {}
    for img_info in data['images']:
        img_info_dic[img_info['id']] = img_info['file_name']

    all_box_list = []
    for bbox_info in data['annotations']:
        bbox = bbox_info['bbox']
        box_list = [
            bbox_info['image_id'],
            classes[bbox_info['category_id']],
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2] + bbox[0]),
            int(bbox[3] + bbox[1]), ]
        all_box_list.append(box_list)

    # to txt
    for box_list in all_box_list:
        txt_path = txt_dir + img_info_dic[box_list[0]][:-4] + '.txt'  # 生成的txt标注文件地址
        txt = open(txt_path, 'a')

        new_line = box_list[1] + ' ' + str(int(box_list[2])) + ' ' + str(int(box_list[3])) + ' ' + str(
            int(box_list[4])) + ' ' + str(int(box_list[5]))
        txt.writelines(new_line)
        txt.write('\n')
        txt.close()
    print('to_txt_done')


def generate_xml(name, lines, img_size):
    doc = Document()  # 创建DOM文档对象
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('coco')
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
    title_text = doc.createTextNode('The coco Database')
    title.appendChild(title_text)
    source.appendChild(title)
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('coco')
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

    for split_line in lines:
        line = split_line.strip().split()

        if line[0] in class_need:
            object = doc.createElement('object')
            annotation.appendChild(object)
            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)
            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(line[1])))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(line[2])))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(line[3])))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(line[4])))
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
                "class": str(line[0]),
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
    json_str = json.dumps(json_str_from_txt, indent=2)

    with open(json_dir + name + '.json', 'w') as file_h:
        file_h.write(json_str)


def generate_coco_pkl(Original_Json, Save_Path):
    data_dict = {}
    with open(Original_Json, 'r') as file_handle:
        json_data = json.load(file_handle)
    img_info_dic = {}
    for img_info in json_data['images']:        # 在"images"关键字下面存放的是图片名字,width,high和图片id信息,格式="0000id.jpg"
        img_info_dic[img_info['id']] = {
            "image_name": img_info['file_name'],
            "width": img_info['width'],
            "height": img_info['height'],
            "bboxes": [],
            "bboxes_cls": [],
        }

    for bbox_info in json_data['annotations']:   # "annotations"存放的是segmentation,image_id,bbox和category_id.
        class_name = classes[bbox_info['category_id']]
        if class_name not in class_need:
            continue
        bbox_img_info = img_info_dic[bbox_info['image_id']]
        width = bbox_img_info['width']
        height = bbox_img_info['height']

        bbox = bbox_info['bbox']    # 左上角坐标和width,high
        box = [
            float(bbox[0]) / width,
            float(bbox[1]) / height,
            float(bbox[2] + bbox[0]) / width,
            float(bbox[3] + bbox[1]) / height,
        ]
        bbox_img_info['bboxes'].append(box)
        one_hot_vector = [0] * len(class_need)
        idx = class_need.index(class_name)
        one_hot_vector[idx] = 1
        bbox_img_info['bboxes_cls'].append(one_hot_vector)

    cnt = 0
    for img_id, img_info in img_info_dic.items():
        bounding_boxes = img_info['bboxes']
        bounding_boxes_cls = img_info['bboxes_cls']
        if len(bounding_boxes) == 0 or len(bounding_boxes_cls) == 0:  # avoid no target
            continue
        img_name = img_info['image_name']
        obj_bboxes = np.asarray(bounding_boxes)
        obj_classes_id = np.asarray(bounding_boxes_cls)
        image_data = np.hstack((obj_bboxes, obj_classes_id))
        data_dict[img_name] = image_data
        cnt += 1

    print('[Info]: ------{0} \n total is {1}'.format(data_dict, cnt))
    pickle.dump(data_dict, open(Save_Path, 'wb'))
    print('Create COCO pkl is Ok!')


def generate_coco_tfrecord():
    pass


def test_pkl(train_img_path, pkl_path):
    gt = pickle.load(open(pkl_path, 'rb'))
    keys = sorted(gt.keys())

    class_name_np = np.array(class_need)
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
    print('This is a special script for processing COCO 2017 data......')
    Img_DIR = '/eDisk/FCWS_dataset/coco2017/train&val/val2017/'
    Original_Json = '/eDisk/FCWS_dataset/coco2017/train&val/annotations/instances_val2017.json'
    Save_Path = 'COCO_2017.pkl'
    trans_format = 'pkl'
    if trans_format == 'pkl':
        # generate_coco_pkl(Original_Json, Save_Path)
        test_pkl(Img_DIR, Save_Path)

    elif trans_format == 'tfrecord':
        generate_coco_tfrecord()
    elif trans_format == 'json' or trans_format == 'xml':
        TXT_DIR = '.'  # 保存txt格式的文件夹地址
        xml_dir = './xml/'  # 保存xml格式文件的文件夹地址
        json_dir = './json/'  # 保存json格式文件的文件夹地址
        read_json_to_txt(Original_Json, TXT_DIR)    # generator txt of xml
        need_save_json = True
        need_save_xml = True
        for parent, dirnames, filenames in os.walk(TXT_DIR):  # 分别得到根目录，子目录和根目录下文件
            for file_name in filenames:
                full_path = os.path.join(parent, file_name)  # 获取文件全路径
                f = open(full_path)
                split_lines = f.readlines()

                name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
                img_name = name + '.jpg'
                img_path = os.path.join(Img_DIR, img_name)
                img_size = cv2.imread(img_path).shape

                if need_save_xml:
                    generate_xml(name, split_lines, img_size)

    print('The COCO-2017 data processes is Ok!')
