"""Keras implementation of SSD."""
# -*- coding: utf-8 -*-
import keras.backend as K
from keras.layers import Conv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Flatten, Add
from keras.layers import Input
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers import Reshape
from keras.models import Model
from ssd_layers import PriorBox
from keras.applications import MobileNetV2


# from keras_applications import mobilenet_v2 as MobileNetV2


def relu6(x):
    return K.relu(x, max_value=6)


def LiteConv(x, i, filter_num):
    x = Conv2D(filter_num // 2, (1, 1), padding='same', use_bias=False, name=str(i) + '_pwconv1')(x)
    x = BatchNormalization(momentum=0.99, name=str(i) + '_pwconv1_bn')(x)
    x = Activation('relu', name=str(i) + '_pwconv1_act')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None, use_bias=False, padding='same',
                        name=str(i) + '_dwconv2')(x)
    x = BatchNormalization(momentum=0.99, name=str(i) + '_sepconv2_bn')(x)
    x = Activation('relu', name=str(i) + '_sepconv2_act')(x)
    net = Conv2D(filter_num, (1, 1), padding='same', use_bias=False, name=str(i) + '_pwconv3')(x)
    x = BatchNormalization(momentum=0.99, name=str(i) + '_pwconv3_bn')(net)
    x = Activation('relu', name=str(i) + '_pwconv3_act')(x)
    print(x.shape)
    return x, net


def Conv(x, filter_num):
    net = Conv2D(filter_num, kernel_size=1, strides=(2, 2), use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(net)
    x = Activation(relu6, name='Conv_1_relu')(x)
    print(x.shape)
    return x, net


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """ MobileNet V2与V1的比较，主要改进的两点：
        1. Inverted residuals:
            传统方法： 通常的residuals block是先经过一个1*1的Conv layer，把feature map的通道数“压”下来，再经过3*3 Conv layer，
            最后经过一个1*1 的Conv layer，将feature map 通道数再“扩张”回去。即先“压缩”，最后“扩张”回去。 
            V2改进方法： inverted residuals就是 先“扩张”，最后“压缩”。主要也是为了提取更多的通道信息，得到更多的特征线信息。
        2. Linear bottlenecks：
            激活函数在高维空间能够有效的增加非线性，而在低维空间时则会破坏特征，不如线性的效果好
            最后不采用Relu，而是Linear，目的是防止Relu破坏特征。

        MobileNet V2与ResNet比较：
        1. 相同点：
            都采用了 1*1 --》 3*3 --》 1*1 的模式。同样使用 Shortcut 将输出与输入相加
        2. 不同点：
            Inverted Residual Block和Bottleneck

    """
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
               name='mobl%d_conv_expand' % block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_bn_expand' % block_id)(x)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                        name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_depthwise' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
               name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_bn_project' % block_id)(x)
    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])
    return x


def _isb4conv13(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    # Expand
    conv = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
                  name='mobl%d_conv_expand' % block_id)(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_bn_expand' % block_id)(conv)
    x = Activation(relu6, name='conv_%d_relu' % block_id)(x)
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                        name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_depthwise' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
               name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn%d_conv_bn_project' % block_id)(x)
    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])
    return x, conv


def prediction(x, i, num_priors, min_s, max_s, aspect, num_classes, img_size):
    a = Conv2D(num_priors * 4, (3, 3), padding='same', name=str(i) + '_mbox_loc')(x)
    mbox_loc_flat = Flatten(name=str(i) + '_mbox_loc_flat')(a)
    b = Conv2D(num_priors * num_classes, (3, 3), padding='same', name=str(i) + '_mbox_conf')(x)
    mbox_conf_flat = Flatten(name=str(i) + '_mbox_conf_flat')(b)
    mbox_priorbox = PriorBox(img_size, min_size=min_s, max_size=max_s, aspect_ratios=aspect,
                             variances=[0.1, 0.1, 0.2, 0.2], name=str(i) + '_mbox_priorbox')(x)
    return mbox_loc_flat, mbox_conf_flat, mbox_priorbox


def SSD(input_shape, num_classes):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    alpha = 1.0      # alpha = [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]
    alpha_t = 1.0
    img_size = (input_shape[1], input_shape[0])
    input_shape = (input_shape[1], input_shape[0], 3)
    # mobilenetv2_input_shape = (224, 224, 3)   # ??????
    mobilenetv2_input_shape = input_shape

    Input0 = Input(input_shape)
    mobilenetv2 = MobileNetV2(input_shape=mobilenetv2_input_shape, include_top=False, weights='imagenet')
    # mobilenetv2 = MobileNetV2(input_shape=mobilenetv2_input_shape, include_top=False, weights=None)
    # print(mobilenetv2.summary())
    FeatureExtractor=Model(inputs=mobilenetv2.input, outputs=mobilenetv2.get_layer('block_13_expand_relu').output)
    # FeatureExtractor = Model(inputs=mobilenetv2.input, outputs=mobilenetv2.get_layer('block_12_add').output)
    # get_3rd_layer_output = K.function([mobilenetv2.layers[114].input, K.learning_phase()],
    #                                  [mobilenetv2.layers[147].output])

    x = FeatureExtractor(Input0)
    x, pwconv3 = _isb4conv13(x, filters=160, alpha=alpha_t, stride=1, expansion=6, block_id=13)
    # x=get_3rd_layer_output([x,1])[0]
    x = _inverted_res_block(x, filters=160, alpha=alpha_t, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha_t, stride=1, expansion=6, block_id=15)
    x = _inverted_res_block(x, filters=320, alpha=alpha_t, stride=1, expansion=6, block_id=16)
    x, pwconv4 = Conv(x, 1280)
    x, pwconv5 = LiteConv(x, 5, 512)
    x, pwconv6 = LiteConv(x, 6, 256)
    x, pwconv7 = LiteConv(x, 7, 256)
    x, pwconv8 = LiteConv(x, 8, 128)

    # def prediction(x, i, num_priors, min_s, max_s, aspect, num_classes, img_size):
    pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 5, 60.0, None, [2, 3],
                                                                                      num_classes, img_size)
    pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6, 105.0, 150.0,
                                                                                      [2, 3], num_classes, img_size)
    pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(pwconv5, 5, 6, 150.0, 195.0,
                                                                                      [2, 3], num_classes, img_size)
    pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(pwconv6, 6, 6, 195.0, 240.0,
                                                                                      [2, 3], num_classes, img_size)
    pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(pwconv7, 7, 6, 240.0, 285.0,
                                                                                      [2, 3], num_classes, img_size)
    pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(pwconv8, 8, 6, 285.0, 300.0,
                                                                                      [2, 3], num_classes, img_size)

    # Gather all predictions
    mbox_loc = concatenate(
        [pwconv3_mbox_loc_flat, pwconv4_mbox_loc_flat, pwconv5_mbox_loc_flat, pwconv6_mbox_loc_flat,
         pwconv7_mbox_loc_flat, pwconv8_mbox_loc_flat], axis=1, name='mbox_loc')
    mbox_conf = concatenate(
        [pwconv3_mbox_conf_flat, pwconv4_mbox_conf_flat, pwconv5_mbox_conf_flat, pwconv6_mbox_conf_flat,
         pwconv7_mbox_conf_flat, pwconv8_mbox_conf_flat], axis=1, name='mbox_conf')
    mbox_priorbox = concatenate(
        [pwconv3_mbox_priorbox, pwconv4_mbox_priorbox, pwconv5_mbox_priorbox, pwconv6_mbox_priorbox,
         pwconv7_mbox_priorbox, pwconv8_mbox_priorbox], axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc, mbox_conf, mbox_priorbox], axis=2, name='predictions')
    model = Model(inputs=Input0, outputs=predictions)
    print('==============************************======================:')
    print('==============************************======================:')
    model.summary()
    return model


if __name__ == '__main__':
    input_shape = (300, 300, 3)
    NUM_CLASSES = 1 + 2  # background and classes
    model = SSD(input_shape, num_classes=NUM_CLASSES)