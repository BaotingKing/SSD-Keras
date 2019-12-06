# -*- coding: utf-8 -*-
import os
import json

CLASS_NEED = {
    'car': 'Car',
    'Car': 'Car',
    'bus': 'Bus',
    'Bus': 'Bus',
    'truck': 'Bus',
    'Truck': 'Bus',
    'Van': 'Bus',
}


def generate_txt(text_out_path, bboxes_infor):
    with open(text_out_path, 'w') as f:
        for idx, obj in enumerate(bboxes_infor):
            classes = CLASS_NEED[obj[0]] + ' '
            x1 = str(obj[1]) + ' '
            y1 = str(obj[2]) + ' '
            x2 = str(obj[3]) + ' '
            y2 = str(obj[4])
            obj_w = [classes, x1, y1, x2, y2]
            f.writelines(obj_w)
            f.write('\n')


class GT_Txt_preprocessor(object):
    def __init__(self, label_path, result_out_path, lable_type=''):
        self.label_path = label_path
        self.result_out_path = result_out_path
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
        if not os.path.exists(self.result_out_path):
            os.makedirs(self.result_out_path)
        filename = self.label_path
        if filename[-5:] == ".json" and 'val' in filename and os.path.exists(filename):    # assign file is val
            with open(filename) as f:
                labels_data_json = json.load(f)

                for item_dict in labels_data_json:
                    img_name = item_dict['name']

                    objs_info = []
                    labels = item_dict['labels']
                    for k in labels:
                        if k['category'] in CLASS_NEED.keys():
                            bbox = k['box2d']
                            single_obj_value = [k['category'],
                                                min(bbox['x1'], bbox['x2']),
                                                min(bbox['y1'], bbox['y2']),
                                                max(bbox['x1'], bbox['x2']),
                                                max(bbox['y1'], bbox['y2'])]
                            objs_info.append(single_obj_value)
                    if objs_info != 0:
                        txt_name = img_name[:-4] + '.txt'
                        generate_txt(os.path.join(self.result_out_path, txt_name), objs_info)
        print('---------------------: BDD100K is OK')

    def _preprocess_Cityscapes_jsons(self):
        """
            For Cityscape dataset
            All images's labels in their respective json files
        """
        if not os.path.exists(self.result_out_path):
            os.makedirs(self.result_out_path)
        for parent, dirnames, filenames in os.walk(self.label_path):  # 分别得到根目录，子目录和根目录下文件
            for filename in filenames:
                if filename[-5:] == ".json":
                    json_file_path = os.path.join(parent, filename)  # 获取文件全路径
                    with open(json_file_path) as f:
                        labels_data_json = json.load(f)

                        objs_info = []
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
                                objs_info.append(single_obj_value)
                    if objs_info != 0:
                        name_id = filename.replace('_gtFine_polygons.json', '')
                        image_name = name_id + '_leftImg8bit.png'
                        txt_name = image_name[:-4] + '.txt'
                        generate_txt(os.path.join(self.result_out_path, txt_name), objs_info)
        print('---------------------: Cityscape is OK')

    def _preprocess_Kitty_txt(self):
        if not os.path.exists(self.result_out_path):
            os.makedirs(self.result_out_path)
        for parent, dirnames, filenames in os.walk(self.json_path):  # 分别得到根目录，子目录和根目录下文件
            for file_name in filenames:
                if file_name[-4:] == '.txt':  # KITTY dataset's labels
                    label_path = os.path.join(parent, file_name)  # 获取文件全路径
                    name = file_name[:-4] + '.png'  # KITTY's type
                    img_path = os.path.join(self.path_prefix, name)

                    if os.path.exists(label_path):
                        objs_info = []
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
                                    objs_info.append(single_obj_value)
                                if objs_info != 0:
                                    txt_name = file_name[:-4] + '.txt'
                                    generate_txt(os.path.join(self.result_out_path, txt_name), objs_info)
        print('---------------------: Kitty is OK')

    def _preprocess_Udacity_txt(self):
        if not os.path.exists(self.result_out_path):
            os.makedirs(self.result_out_path)
        for parent, dirnames, filenames in os.walk(self.label_path):  # Udacity's label in txt file
            for file_name in filenames:
                if file_name[-4:] == '.txt':  # Udacity dataset's labels
                    label_path = os.path.join(parent, file_name)

                    if os.path.exists(label_path):
                        objs_info = []
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
                                    objs_info.append(single_obj_value)
                                if objs_info != 0:
                                    txt_name = file_name[:-4] + '.txt'
                                    generate_txt(os.path.join(self.result_out_path, txt_name), objs_info)
        print('---------------------: Udacity is OK')


if __name__ == '__main__':
    print('This is used to generate GT files!')

    # json_path = '/eDisk/FCWS_dataset/BDD100k/bdd100k/labels/bdd100k_labels_images_val.json'
    # output_path = './mAP/BDD_GT'
    # # bdd_data_df = GT_Txt_preprocessor(json_path, output_path, lable_type='BDD')
    #
    # json_path = '/eDisk/FCWS_dataset/Cityscape/cityscaps_label/gtFine/val'
    # output_path = './mAP/Cityscape_GT'
    # cityscape_data_df = GT_Txt_preprocessor(json_path, output_path, lable_type='Cityscape')

    txt_path = '/eDisk/FCWS_dataset/Udacity/object_detection_crowdai/label_txt'
    output_path = './mAP/Udacity_GT'
    Udacity_data_df = GT_Txt_preprocessor(txt_path, output_path, lable_type='Udacity')

