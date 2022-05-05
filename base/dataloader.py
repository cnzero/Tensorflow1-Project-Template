# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-06
# @author: cnzero


import tensorflow as tf
from typing import Tuple, Dict, Sized


class BaseDataLoader(Sized):
    def __init__(self,
                 configs: dict,
                 mode: str) -> None:
        self.configs = configs
        self.mode = mode

    def input_fn(self) -> tf.data.Dataset:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)