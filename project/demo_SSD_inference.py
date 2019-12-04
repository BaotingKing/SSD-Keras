import cv2
import os
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import imread
import tensorflow as tf

# from model.ssd300VGG16 import SSD
from model.ssd300MobileNetV2Lite import SSD
from ssd_utils import BBoxUtility


# np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
# set_session(tf.Session(config=config))


tf.reset_default_graph()

# voc_classes = ['Yellow',

#         'Red',  # 'RedStraight',
# #         'RedRight',
# #         'RedLeft',   # 'RedStraightLeft',

#         'Green', # 'GreenStraight',
# #         'GreenRight', # 'Green Straight Right',
# #         'GreenLeft',  # 'GreenStraightLeft'

#         'off',
#               ]
# NUM_CLASSES = len(voc_classes) + 1
voc_classes = ['car', 'motobike']

NUM_CLASSES = 1 + 2

input_shape = (300, 300, 3)
model = SSD(input_shape, num_classes=NUM_CLASSES)
# model.load_weights('./weights/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model.load_weights('../weights/checkpoints/w_test/weights.04-3.26_3.28.hdf5')
# model.load_weights('./other_tf_dataset/300_300_vgg/weights.20-2.95.hdf5',by_name=True)


bbox_util = BBoxUtility(NUM_CLASSES)

inputs = []
images = []

test_path = '../example/img'
for parent, dirnames, filenames in os.walk(test_path):  # 分别得到根目录，子目录和根目录下文件
    for file_name in filenames:
        img_path = os.path.join(parent, file_name)  # 获取文件全路径
        print(img_path)

        # img = image.load_img(img_path, target_size=(512,512))
        # img = image.img_to_array(img)

        img_cv2 = cv2.imread(img_path)
        img = cv2.resize(img_cv2, (300, 300)).astype('float32')

        inputs.append(img.copy())

        if file_name[-4:] == '.jpg':

            img_ = imread(img_path, cv2.IMREAD_UNCHANGED)
            # img_ = cv2.cvtColor(img_, cv2.COLOR_BAYER_GB2BGR)
            # img_ = np.right_shift(img_, 4)
            # img_ = img_.astype(np.uint8)
            # img_ = img_[:, :, [2, 1, 0]]
            images.append(img_)

        else:
            images.append(imread(img_path))

inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
# print(preds)
results = bbox_util.detection_out(preds)
print(results[0])
print(len(results[0]))

plt.rcParams['figure.figsize'] = (40, 40)
plt.rcParams['image.interpolation'] = 'nearest'

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]

    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if 0.65 <= conf]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()
