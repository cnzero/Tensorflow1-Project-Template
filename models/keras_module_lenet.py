# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-07
# @author: cnzero


import numpy as np
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import sys
sys.path.append(os.getcwd())


class KerasModuleLeNet(object):
    def __init__(self,
                 configs: dict):
        self.configs = configs

        # 0) insert another dim for 'NHWC'
        # 1) conv2d + sigmoid
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
                                               kernel_size=(5, 5),
                                               strides=(1, 1),
                                               padding='valid',
                                               data_format='channels_last',
                                               activation='sigmoid',
                                               kernel_constraint=None,
                                               bias_constraint=None)
        # 2) avgpool2d
        self.avgpool2d_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                            strides=(2, 2),
                                                            data_format='channels_last')
        # 3) conv2d + sigmoid
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16,
                                               kernel_size=(5, 5),
                                               activation='sigmoid',
                                               data_format='channels_last')
        # 4) avgpool2d
        self.avgpool2d_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                            strides=(2, 2),
                                                            data_format='channels_last')
        # 5) flatten
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')
        # 6) dense to 120
        self.dense1 = tf.keras.layers.Dense(units=120, activation='sigmoid')
        # 7) dense to 84
        self.dense2 = tf.keras.layers.Dense(units=84, activation='sigmoid')
        # 8) dense to 10
        self.dense3 = tf.keras.layers.Dense(units=10, activation='sigmoid')

    def __call__(self,
                 inputs: np.ndarray):
        # 0) insert another dim for 'NHWC'
        x = tf.convert_to_tensor(inputs.reshape(inputs.shape + (1, )), dtype=inputs.dtype)
        # 1) conv2d + sigmoid
        x = self.conv2d_1(x)
        # 2) avgpool2d
        x = self.avgpool2d_1(x)
        # 3) conv2d + sigmoid
        x = self.conv2d_2(x)
        # 4) maxpool2d
        x = self.avgpool2d_2(x)
        # 5) flatten
        x = self.flatten(x)
        # 6) dense to 120
        x = self.dense1(x)
        # 7) dense to 84
        x = self.dense2(x)
        # 8) dense to 10
        x = self.dense3(x)

        return x


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)

    batch_size, height, width = 2, 28, 28
    in_channels = 1
    images = np.zeros(shape=(batch_size, height, width))

    configs = None
    lenet = KerasModuleLeNet(configs=configs)
    output = lenet(images)
    print('Keras Module LeNet shape: ', output.shape) # shape(2, 10)