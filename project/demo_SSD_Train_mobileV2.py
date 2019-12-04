import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pickle
import multiprocessing
from random import shuffle, seed
from scipy.misc import imread
from scipy.misc import imresize
import os
import shutil
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from model.ssd300MobileNetV2Lite import SSD
from model.ssd300Mv2Lite_Pro import SSD
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility


# some constants
batch_size = 32
NUM_CLASSES = 1 + 2  # background and classes
input_shape = (300, 300, 3)
MODEL_PATH = '../weights/checkpoints/SSDv2_pro/'

priors = pickle.load(open('../priorFiles/new_prior_boxes_ssd300MobileNetV2.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

DATASET = 'BDD'
if DATASET == 'BDD':
    gt = pickle.load(open('../project/BDD_train.pkl', 'rb'))
    # gt_val = pickle.load(open('../project/BDD_val.pkl', 'rb'))
    # gt.update(gt_val)  # merge train dataset and val dataset for BDD100K
    keys = sorted(gt.keys())
    shuffle(keys, random=seed(3))
    num_train = int(round(0.85 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)
    path_prefix = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/train/'  # This is BDD100K dataset
elif DATASET == 'cityscape':
    gt = pickle.load(open('../project/Cityscape_train.pkl', 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)
    path_prefix = '/home/zack/studio/train_old/'
elif DATASET == 'm':
    gt = pickle.load(open('../project/BDD_train.pkl', 'rb'))
    gt_other = pickle.load(open('../project/Cityscape_train.pkl', 'rb'))
    gt.update(gt_other)  # merge all train dataset
    keys = sorted(gt.keys())
    shuffle(keys, random=seed(3))
    num_train = int(round(0.9 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)
    path_prefix = '/home/zack/studio/train_old/'
elif DATASET == 'test':
    gt = pickle.load(open('../project/small_BDD.pkl', 'rb'))
    keys = sorted(gt.keys())
    train_keys = keys[:]
    val_keys = keys[:]
    num_val = len(val_keys)
    path_prefix = '/eDisk/FCWS_dataset/BDD100k/bdd100k/images/100k/train/'  # This is BDD100K dataset
    MODEL_PATH = '../weights/checkpoints/test/'

save_type = False
if save_type:
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        '../weights/checkpoints/weights.{epoch:02d}-{loss:.2f}_{val_loss:.2f}.hdf5',
        verbose=1,
        save_weights_only=True)
else:
    save_path = os.path.join(MODEL_PATH, 'best_weights.hdf5')
    if os.path.exists(save_path):
        shutil.copyfile(save_path, os.path.join(MODEL_PATH, str(int(time.time())) + '.hdf5'))
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        save_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=True)


class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3. / 4., 4. / 3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y + h, x:x + w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                    y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                # img = cv2.imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                # img = cv2.resize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                # imshow(img)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets


gen = Generator(gt, bbox_util, batch_size, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)

model = SSD(input_shape, num_classes=NUM_CLASSES)
flag = 0
if flag == 0:
    model.load_weights(os.path.join(MODEL_PATH, 'best_weights.hdf5'), by_name=True)

elif flag == 1:
    MODEL_PATH = '../model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
    model.load_weights(MODEL_PATH, by_name=True)

elif flag == 2:
    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3']  # ,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']
    for L in model.layers:
        if L.name in freeze:
            L.trainable = False


base_lr = 0.01 * 0.1
# optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=0.9, nesterov=True)

model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['acc'])


def schedule(epoch, decay=0.9):
    """keras.callbacks.LearningRateScheduler(schedule)"""
    return base_lr * decay ** epoch


tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_PATH, 'tfLog'),
                                         # histogram_freq=1,
                                         write_graph=True,
                                         write_images=True)

callbacks = [model_checkpoint,
             tbCallBack,
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=1)
             # keras.callbacks.LearningRateScheduler(schedule, verbose=1)
             ]

nb_epoch = 1500
if False:
    history = model.fit_generator(gen.generate(True), gen.train_batches,
                                  nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  nb_val_samples=gen.val_batches,
                                  nb_worker=1)
else:
    if os.name is 'nt':
        workers = 0
    else:
        workers = multiprocessing.cpu_count()

    history = model.fit_generator(gen.generate(True),
                                  steps_per_epoch=int(gen.train_batches // batch_size),
                                  epochs=nb_epoch,
                                  initial_epoch=0,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=gen.generate(False),
                                  validation_steps=gen.val_batches // batch_size,
                                  max_queue_size=100,
                                  workers=workers,
                                  use_multiprocessing=True,
                                  )
