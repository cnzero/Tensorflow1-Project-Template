# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-15
# @author: cnzero

import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.contrib.eager as tfe

# tf.enable_eager_execution()

layers = tf.keras.layers


class TemporalBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='causal',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation='relu',
                 dropout=0.0,
                 mode='train'):
        """the basic TCN block, as shown in Figure 1.(b) in ref[1]
        references:
        [1]: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

        Args:
            filters (int): the dim of the output space,
                           the number of output filters in the convolution
                           means 'out_channels'
            kernel_size (int or list of int):
            strides (int or list of int):
            padding (str): 'valid', 'causal', 'same'
            data_format (str): 'channels_last', 'channels_first'
            dilation_rate (int or list of int):
            activation (str): activation function to use, default 'relu' after normalization
                              'linear': a(x) = x
            dropout (float): dropout
        """
        super(TemporalBlock, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']
        self.mode = mode

        # block1: a) dilated causal conv, b) norm, c) ReLU, d) Dropout
        self.conv1 = layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=padding,
                                   data_format=data_format,
                                   dilation_rate=dilation_rate,
                                   kernel_initializer=initializer)
        self.batch_norm1 = layers.BatchNormalization(axis=-1)
        self.activation1 = layers.Activation(activation)
        self.dropout1 = layers.Dropout(rate=dropout)

        # block2: a) dilated causal conv, b) norm, c) ReLU, d) Dropout
        self.conv2 = layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   dilation_rate=dilation_rate,
                                   padding=padding,
                                   kernel_initializer=initializer)
        self.batch_norm2 = layers.BatchNormalization(axis=-1)
        self.activation2 = layers.Activation(activation)
        self.dropout2 = layers.Dropout(rate=dropout)

        # 1x1 conv for different dimension residual add
        self.residual_conv = layers.Conv1D(filters=filters,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=initializer)
        self.activation3 = layers.Activation(activation)

    def __call__(self,
                 x,
                 training=True,
                 **kwargs):
        prev_x = x  # for later residual add
        # block1: a) dilated causal conv, b) norm, c) ReLU, d) Dropout
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x) if self.mode in {'train', 'Train'} else x

        # block2: a) dilated causal conv, b) norm, c) ReLU, d) Dropout
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x) if self.mode in {'train', 'Train'} else x

        if prev_x.shape[-1] != x.shape[-1]:  # match the dimension
            prev_x = self.residual_conv(prev_x)
        assert prev_x.shape == x.shape

        # residual add
        outputs = self.activation3(prev_x + x)

        return outputs


class TemporalConvNet(tf.keras.Model):
    def __init__(self,
                 list_num_channels,
                 kernel_size=2,
                 strides=1,
                 padding='causal',
                 data_format='channels_last',
                 dropout=0.2):
        """stacking 'TemporalBlock' for sequence modeling

        Args:
            list_num_channels (list of int): len(list_num_channels) means the number of 'TemporalBlock'
                                             each element means in/out_channels for the 'TemporalBlock'
            kernel_size (int): kernel size for the 'TemporalBlock'
            strides (int or list of int):
            padding (str): 'valid', 'causal', 'same'
            data_format (str): 'channels_last', 'channels_first'
            dilation_rate (int or list of int):
            activation (str): activation function to use, default 'relu' after normalization
                              'linear': a(x) = x
            dropout (float): dropout
        """
        # num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(list_num_channels, list)

        # 初始化 model
        model = tf.keras.Sequential()

        # The model contains "num_levels" TemporalBlock
        for i in range(len(list_num_channels)):
            dilation_rate = 2 ** i  # exponential growth
            model.add(TemporalBlock(filters=list_num_channels[i],
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=2**i,     # exponential growth
                                    dropout=dropout))
        self.network = model

    def __call__(self, x,
                 training=True,
                 **kwargs):
        return self.network(x, training=training)


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)
    print('test for TemporalBlock.')
    batch_size = 100
    seq_length = 10
    in_channels = 50
    seq_inputs = tf.convert_to_tensor(np.random.random((batch_size, seq_length, in_channels)))
    temporal_block = TemporalBlock(filters=70,
                                   kernel_size=3,
                                   padding='causal',
                                   dilation_rate=3,
                                   dropout=0.1)

    seq_outputs = temporal_block(seq_inputs)

    print('TemporalBlock \tinputs shape: ', seq_inputs.shape,
          '\n\t\t outputs shape: ', seq_outputs.shape)

    print('test for TemporalConvNet.')
    seq_inputs = tf.convert_to_tensor(np.random.random((batch_size, seq_length, in_channels)))
    temporal_convnet = TemporalConvNet(list_num_channels=[70, 80])
    seq_outputs = temporal_convnet(seq_inputs)
    print('TemporalConv \tinputs shape: ', seq_inputs.shape,
          '\n\t\t outputs shape: ', seq_outputs.shape)
