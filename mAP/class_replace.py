#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time: 2019/11/20 18:23
import os
import re

transform_class = ['truck', 'bus']
CLASS_Car = ['car', 'Car']
CLASS_Bus = ['truck', 'bus', 'Truck', 'Bus']
ground_txt_path = './old_gt'
if not os.path.exists('./ground_txt_path_new'):
    os.mkdir('./ground_txt_path_new')

gt_list = os.listdir(ground_txt_path)

for txt in gt_list:
    new_txt = os.path.join('./ground_txt_path_new', txt)
    txt = os.path.join(ground_txt_path, txt)
    with open(txt, 'r') as file_handle, open(new_txt, 'w') as new_file_handle:
        for line in file_handle.readlines():
            cls = line.split(' ')[0]
            if cls in CLASS_Car:
                line = line.replace(cls, 'Car')
            elif cls in CLASS_Bus:
                line = line.replace(cls, 'Bus')

            if cls in CLASS_Car or cls in CLASS_Bus:
                new_file_handle.write(line)
            else:
                continue
