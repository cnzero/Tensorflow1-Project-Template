# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-06
# @author: cnzero


import tensorflow as tf


class BaseModule(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    print('self-testing Hello world in ', __file__)