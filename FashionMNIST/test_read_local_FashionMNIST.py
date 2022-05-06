# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-06
# @author: cnzero

"""python how to read local FashionMNIST dataset to numpy.ndarray
"""


import gzip
import numpy as np

import os, sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    print('self-test Hello World in ', __file__)

    with gzip.open(filename='FashionMNIST/raw/train-images-idx3-ubyte.gz', mode='rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 28, 28))

    with gzip.open(filename='FashionMNIST/raw/train-labels-idx1-ubyte.gz', mode='rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print('train image shape: ', train_images.shape)  # np.ndarray shape(60000, 28, 28)
    print('train label shape: ', train_labels.shape)  # np.ndarray shape(60000, )