import pickle
import numpy as np
import cv2

img_width, img_height = 300, 300
if True:
    box_configs = [
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 2, 'min_size': 30.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0]},

        {'layer_width': 19, 'layer_height': 19, 'num_prior': 3, 'min_size': 60.0,
         'max_size': 105.0, 'aspect_ratios': [1.0, 2.0, 1 / 2.0]},

        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 105.0,
         'max_size': 150.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 5, 'layer_height': 5, 'num_prior': 6, 'min_size': 150.0,
         'max_size': 195.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 3, 'layer_height': 3, 'num_prior': 6, 'min_size': 195.0,
         'max_size': 240.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 2, 'layer_height': 2, 'num_prior': 6, 'min_size': 240.0,
         'max_size': 285.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 1, 'layer_height': 1, 'num_prior': 6, 'min_size': 285.0,
         'max_size': 300.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
    ]
else:
    box_configs_old = [
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 3, 'min_size': 60.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0, 1 / 2.0]},

        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 105.0,
         'max_size': 150.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 5, 'layer_height': 5, 'num_prior': 6, 'min_size': 150.0,
         'max_size': 195.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 3, 'layer_height': 3, 'num_prior': 6, 'min_size': 195.0,
         'max_size': 240.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 2, 'layer_height': 2, 'num_prior': 6, 'min_size': 240.0,
         'max_size': 285.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 1, 'layer_height': 1, 'num_prior': 6, 'min_size': 285.0,
         'max_size': 300.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
    ]

    box_configs_pro = [
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 4, 'min_size': 30.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0, 1 / 2.0, 3.0]},

        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 75.0,
         'max_size': 120.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 5, 'layer_height': 5, 'num_prior': 6, 'min_size': 120.0,
         'max_size': 165.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 3, 'layer_height': 3, 'num_prior': 6, 'min_size': 165.0,
         'max_size': 210.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 2, 'layer_height': 2, 'num_prior': 6, 'min_size': 210.0,
         'max_size': 255.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},

        {'layer_width': 1, 'layer_height': 1, 'num_prior': 6, 'min_size': 255.0,
         'max_size': 300.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
    ]
    # This is demo for prior create
    box_configs = [
        {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size': 30.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0, 1 / 2.0]},
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size': 60.0,
         'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
         'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        {'layer_width': 5, 'layer_height': 5, 'num_prior': 6, 'min_size': 168.0,
         'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        {'layer_width': 3, 'layer_height': 3, 'num_prior': 6, 'min_size': 222.0,
         'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
        {'layer_width': 1, 'layer_height': 1, 'num_prior': 6, 'min_size': 276.0,
         'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0]},
    ]
variance = [0.1, 0.1, 0.2, 0.2]
boxes_paras = []


def create_prior_box():
    cnt = 0
    for layer_config in box_configs:
        layer_width, layer_height = layer_config["layer_width"], layer_config["layer_height"]
        num_priors = layer_config["num_prior"]
        aspect_ratios = layer_config["aspect_ratios"]
        min_size = layer_config["min_size"]
        max_size = layer_config["max_size"]

        step_x = float(img_width) / float(layer_width)
        step_y = float(img_height) / float(layer_height)

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        assert (num_priors == len(aspect_ratios))
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)

        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        box_widths = []
        box_heights = []
        for ar in aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            elif ar != 1:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        print('uuuuuuuuuuuuuuuuuuuuuuu')
        print(box_heights.shape)
        print(prior_boxes.shape)
        # Normalize to 0-1
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        # clip to 0-1
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        piror_variances = np.tile(variance, (len(prior_boxes), 1))
        boxes_para = np.concatenate((prior_boxes, piror_variances), axis=1)
        boxes_paras.append(boxes_para)
        cnt += boxes_para.shape[0]
        print('*******boxes_para.shape[0]: ', boxes_para.shape[0])
    print('Total prior boxes is: ', cnt)
    return np.concatenate(boxes_paras, axis=0)


if __name__ == "__main__":
    boxes_paras = create_prior_box()

    print(boxes_paras.shape)
    with open('../priorFiles/new_prior_boxes_ssd300MobileNetV2.pkl', 'wb') as f:
        pickle.dump(boxes_paras, f, protocol=2)

    priors = pickle.load(open('../priorFiles/new_prior_boxes_ssd300MobileNetV2.pkl', 'rb'))

    cnt_1_1 = 19 * 19 * 2
    cnt_1_2 = 19 * 19 * 3
    cnt_2 = 10 * 10 * 6
    cnt_3 = 5 * 5 * 6
    cnt_4 = 3 * 3 * 6
    cnt_5 = 3 * 3 * 6
    cnt_6 = 3 * 3 * 6

    cnt = 0
    for idx, layer_config in enumerate(box_configs):
        img = np.zeros((300, 300)).astype('uint8')
        layer_width, layer_height = layer_config["layer_width"], layer_config["layer_height"]
        num_priors = layer_config["num_prior"]
        aspect_ratios = layer_config["aspect_ratios"]
        min_size = layer_config["min_size"]
        max_size = layer_config["max_size"]
        total_bboxes_num = layer_width * layer_height * num_priors

        begin_idx = 0
        for i in range(idx):
            pre_config = box_configs[i]
            begin_idx += pre_config["layer_width"] * pre_config["layer_height"] * pre_config["num_prior"]

        total_bboxes = priors[begin_idx:begin_idx + total_bboxes_num]

        const_num = (layer_width - 1)*num_priors
        if layer_width > 5:
            aim_idx = [0, const_num, total_bboxes_num//2 - num_priors//2, total_bboxes_num - const_num, total_bboxes_num - num_priors]
        elif layer_width == 1:
            aim_idx = [0]
        else:
            aim_idx = [total_bboxes_num//2 - num_priors//2]
        for idx in aim_idx:
            for n in range(num_priors):
                bx = total_bboxes[idx + n]
                x1 = int(bx[0] * img.shape[0])
                y1 = int(bx[1] * img.shape[1])
                x2 = int(bx[2] * img.shape[0])
                y2 = int(bx[3] * img.shape[1])
                print('---------: ', (x1, y1), (x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        img_a = img.copy()
        cv2.imshow('Prior_box', img_a)
        img = cv2.resize(img, (1280, 720)).astype('uint8')
        cv2.imshow('lager', img)
        cv2.waitKey(0)
