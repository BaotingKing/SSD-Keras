import numpy as np
import os
import cv2
import scipy
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imshow
import json
import pickle
import shutil

if True:
    CLASS_NEED = {
        'car': 0,
        'Car': 0,
        'bus': 1,
        'truck': 1,
        'Truck': 1,
        'Van': 1,

    }
else:
    CLASS_NEED = {
        'background': 0,
        'bus': 1,
        'truck': 1,
        'motor': 2,
        'car': 1,
        'motor': 1,
        'motorcycle': 1,

    }

CLASS_NUM = 2


def search_file(rootdir, target_file):
    target_file_path = None
    for parent, dirnames, filenames in os.walk(rootdir):
        if target_file in filenames:
            target_file_path = os.path.join(parent, target_file)
            break
    return target_file_path


class JSON_preprocessor(object):
    def __init__(self, json_path, image_path, data_type='train', lable_type='sigle'):
        self.json_path = json_path
        self.data_type = data_type

        self.path_prefix = image_path
        self.num_classes = CLASS_NUM
        self.data = dict()
        if lable_type == 'BDD':
            self._preprocess_BDD100K_json()
        elif lable_type == 'Cityscape':
            self._preprocess_Cityscapes_jsons()
        elif lable_type == 'KITTY':
            self._preprocess_Kitty_txt()
        else:
            print('[Warning]: No dataset!')

    def _preprocess_BDD100K_json(self):
        """
            For BDD100K dataset
            All images's labels in one json file.
        """
        filenames = os.listdir(self.json_path)

        # with open('/home/zack/studio/small_data/filename.txt', 'r') as tfile:    # This is for small val data
        #     small_file = tfile.readlines()
        # small_file_name = [small_file[i].strip() for i in range(len(small_file))]

        for filename in filenames:
            if filename[-5:] == ".json" and self.data_type in filename:
                filename_path = os.path.join(self.json_path, filename)
                with open(filename_path) as f:
                    labels_data_json = json.load(f)

                    for item_dict in labels_data_json:
                        name = item_dict['name']
                        # if name not in small_file_name:
                        #     continue
                        img_path = os.path.join(TRAIN_IMG_PATH, name)
                        img = cv2.imread(img_path)

                        BDD_WIDTH = img.shape[1]
                        BDD_HEIGHT = img.shape[0]

                        bounding_boxes = []
                        bounding_boxes_cls = []
                        labels = item_dict['labels']
                        for k in labels:
                            if k['category'] in CLASS_NEED.keys():
                                bbox = self.coordinate_normal(k['box2d'], BDD_WIDTH, BDD_HEIGHT)
                                bounding_boxes.append(bbox)
                                class_one_hot = self._to_one_hot(k['category'])
                                bounding_boxes_cls.append(class_one_hot)
                        if len(bounding_boxes) == 0 or len(class_one_hot) == 0:    # avoid no target
                            continue
                        obj_bboxes = np.asarray(bounding_boxes)
                        obj_classes_id = np.asarray(bounding_boxes_cls)
                        image_data = np.hstack((obj_bboxes, obj_classes_id))
                        self.data[name] = image_data

    def _preprocess_Cityscapes_jsons(self):
        """
            For Cityscape dataset
            All images's labels in their respective json files
        """
        label_file_path = os.path.join(self.json_path, self.data_type)
        merge_img_path = './cityscape_img'
        if not os.path.exists(merge_img_path):
            os.makedirs(merge_img_path)

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

                                box = {
                                    'x1': min(x_),
                                    'y1': min(y_),
                                    'x2': max(x_),
                                    'y2': max(y_)
                                }
                                bbox = self.coordinate_normal(box, img_width, img_heigh)
                                bounding_boxes.append(bbox)
                                class_one_hot = self._to_one_hot(label_class)
                                bounding_boxes_cls.append(class_one_hot)
                            if len(bounding_boxes) == 0 or len(class_one_hot) == 0:  # avoid no target
                                continue
                            else:
                                """move all image into one file"""
                                if True:
                                    image_path = search_file(self.path_prefix, image_name)
                                    if image_path:
                                        shutil.copy(image_path, merge_img_path)
                                    else:
                                        continue    # avoid no image
                                else:
                                    print('Pleas put Cityscape data into one file by yourself.....')

                            obj_bboxes = np.asarray(bounding_boxes)
                            obj_classes_id = np.asarray(bounding_boxes_cls)
                            image_data = np.hstack((obj_bboxes, obj_classes_id))
                            self.data[image_name] = image_data

    def _preprocess_Kitty_txt(self):
        for parent, dirnames, filenames in os.walk(self.json_path):  # 分别得到根目录，子目录和根目录下文件
            for file_name in filenames:
                if file_name[-4:] == '.txt':     # KITTY dataset's labels
                    label_path = os.path.join(parent, file_name)  # 获取文件全路径
                    name = file_name[:-4] + '.png'  # KITTY's type
                    img_path = os.path.join(self.path_prefix, name)

                    if os.path.exists(label_path) and os.path.exists(img_path):
                        img_size = cv2.imread(img_path).shape
                        img_heigh = img_size[0]
                        img_width = img_size[1]
                        with open(label_path, 'r') as f:
                            split_lines = f.readlines()

                            bounding_boxes = []
                            bounding_boxes_cls = []
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
                                    bbox = self.coordinate_normal(box, img_width, img_heigh)
                                    bounding_boxes.append(bbox)
                                    class_one_hot = self._to_one_hot(label_class)
                                    bounding_boxes_cls.append(class_one_hot)

                            if len(bounding_boxes) == 0 or len(class_one_hot) == 0:  # avoid no target
                                continue
                            obj_bboxes = np.asarray(bounding_boxes)
                            obj_classes_id = np.asarray(bounding_boxes_cls)
                            image_data = np.hstack((obj_bboxes, obj_classes_id))
                            self.data[name] = image_data

    def _to_one_hot(self, name):

        one_hot_vector = [0] * self.num_classes

        if 0 <= CLASS_NEED[name] < len(one_hot_vector):
            one_hot_vector[CLASS_NEED[name]] = 1
        else:
            print('unknown label: %s' % name)

        return one_hot_vector

    def coordinate_normal(self, bbox, width, height, flag=True):
        xmin = min(bbox['x1'], bbox['x2'])
        ymin = min(bbox['y1'], bbox['y2'])
        xmax = max(bbox['x1'], bbox['x2'])
        ymax = max(bbox['y1'], bbox['y2'])
        if flag:
            xmin = float(xmin / width)
            ymin = float(ymin / height)
            xmax = float(xmax / width)
            ymax = float(ymax / height)
            bounding_box = [xmin, ymin, xmax, ymax]
        else:
            bounding_box = [xmin, ymin, xmax, ymax]
        return bounding_box


def test_pkl(train_img_path, pkl_path):
    gt = pickle.load(open(pkl_path, 'rb'))
    keys = sorted(gt.keys())

    if 'BDD' in train_img_path:
        show_bdd_gt = True
        img_dict = show_bdd_gt_info()
    else:
        show_bdd_gt = False

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
            x1 = int(bbox[0] * WIDTH)
            y1 = int(bbox[1] * HEIGHT)
            x2 = int(bbox[2] * WIDTH)
            y2 = int(bbox[3] * HEIGHT)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            if show_bdd_gt:
                for obj in img_dict[key]:
                    if obj['category'] in CLASS_NEED.keys():
                        left_top_point = (int(obj['box2d']['x1']) + 2, int(obj['box2d']['y1']))
                        right_down_point = (int(obj['box2d']['x2']) + 2, int(obj['box2d']['y2']))
                        if CLASS_NEED[obj['category']] == 0:
                            cv2.rectangle(img, left_top_point, right_down_point, (0, 255, 0), thickness=2)
                        else:
                            cv2.rectangle(img, left_top_point, right_down_point, (200, 255, 255), thickness=2)

            x1 = int(bbox[0] * resize_shape[0])
            y1 = int(bbox[1] * resize_shape[1])
            x2 = int(bbox[2] * resize_shape[0])
            y2 = int(bbox[3] * resize_shape[1])
            cv2.rectangle(img_size, (x1, y1), (x2, y2), (255, 255, 0), thickness=2)

        jion_img_size = (max(img_shape[0], resize_shape[0]), img_shape[1] + resize_shape[1], 3)
        jion_img = np.zeros(jion_img_size).astype('uint8')
        jion_img[:img_shape[0], :img_shape[1], :] = img
        jion_img[:resize_shape[0]:, img_shape[1]:, :] = img_size[:, :, :]
        print('-----------------', key)
        cv2.imshow('orgin', jion_img)
        # cv2.imshow('img', img)
        # cv2.imshow('img_size', img_size)
        cv2.waitKey(0)


def show_bdd_gt_info():
    """This is test for BDD100K"""
    json_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/bdd100k_labels_images_train.json'
    val_img_dict = {}
    with open(json_path) as f:
        labels_data_json = json.load(f)

        for item_dict in labels_data_json:
            name = item_dict['name']
            labels = item_dict['labels']
            val_img_dict[name] = labels
    return val_img_dict


if __name__ == '__main__':
    print('This is get_data_from_json fun for BDD100K dataset processing......')

    json_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/'
    TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/train_merge/'    # merge train and val images

    # data = JSON_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='BDD').data
    # pickle.dump(data, open('BDD_train.pkl', 'wb'))
    # data = JSON_preprocessor(json_path, TRAIN_IMG_PATH, 'val', lable_type='BDD').data
    # pickle.dump(data, open('BDD_val.pkl', 'wb'))
    # data = JSON_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='BDD').data
    # pickle.dump(data, open('test_BDD.pkl', 'wb'))

    json_path = '/eDisk/FCWS_dataset/Cityscape/cityscaps_label/gtFine/'
    TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/Cityscape/cityscaps/leftImg8bit'
    data = JSON_preprocessor(json_path, TRAIN_IMG_PATH, 'train', lable_type='Cityscape').data
    pickle.dump(data, open('Cityscape_train.pkl', 'wb'))

    # txt_path = '/eDisk/FCWS_dataset/KITTI/training_label/label_2/'
    # TRAIN_IMG_PATH = '/eDisk/FCWS_dataset/KITTI/training/image_2/'
    # data = JSON_preprocessor(txt_path, TRAIN_IMG_PATH, 'train', lable_type='KITTY').data
    # pickle.dump(data, open('Kitty_train.pkl', 'wb'))

    # pkl_path = 'BDD_train.pkl'
    pkl_path = 'small_BDD.pkl'
    # pkl_path = 'Cityscape_train.pkl'
    # pkl_path = 'Kitty_train.pkl'
    # TRAIN_IMG_PATH = './cityscape_img'
    test_pkl(TRAIN_IMG_PATH, pkl_path)
    # gt = pickle.load(open('../project/BDD_train.pkl', 'rb'))
    # key = '37947409-566299fe.jpg'
    # a = gt[key]
    # print(gt)
