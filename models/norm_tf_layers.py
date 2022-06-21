# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-5-15
# @author: cnzero

import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    print('\n', '*'*20, 'self-test Hello World in ', __file__, '*'*20)

    batch_size = 2
    height = 3
    width = 4
    num_channels = 1
    np_inputs = np.random.rand(batch_size, height, width, num_channels)
    inputs_placeholder = tf.random_normal(shape=(batch_size, height, width, num_channels))

    with tf.Session() as sess:

        print('inputs shape: \t\t\t\t\t', np_inputs.shape)
        # batch normalization, 1) tf.nn.bath_normalization
        #                      2) tf.layers.batch_normalization
        mean, variance = tf.nn.moments(inputs_placeholder, axes=0, keepdims=True)
        offset = tf.get_variable(name='bn_offset',
                                 shape=(1, height, width, num_channels),
                                 initializer=None)
        # tf.contrib.layers.variance_scaling_initializer()
        # tf.contrib.layers.xavier_initializer()
        scale = tf.get_variable(name='bn_scale',
                                shape=(1, height, width, num_channels),
                                initializer=None)
        tf_bn_outputs1 = tf.nn.batch_normalization(x=inputs_placeholder,
                                                   mean=mean,
                                                   variance=variance,
                                                   offset=offset,
                                                   scale=scale,
                                                   variance_epsilon=1e-8)
        # functional interface for the batch normalization layer within `tf.layers.BatchNormalization()`
        tf_bn_outputs2 = tf.layers.batch_normalization(inputs=inputs_placeholder)
        tf_bn_outputs3 = tf.layers.BatchNormalization()(inputs=inputs_placeholder)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        np_bn_outputs1, \
        np_bn_outputs2, \
        np_bn_outputs3  = sess.run([tf_bn_outputs1, tf_bn_outputs2, tf_bn_outputs3],
                                   feed_dict={inputs_placeholder: np_inputs})
        # tf.layers.batch_normalization()
        print('tf.nn.batch_normalization outputs shape: \t',     np_bn_outputs1.shape)
        print('tf.layers.batch_normalization outputs shape: \t', np_bn_outputs2.shape)
        print('tf.layers.BatchNormalization outputs shape: \t',  np_bn_outputs3.shape)

        # layer normalization

        # group normalization