# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-06
# @author: cnzero


import os, sys
sys.path.append(os.getcwd())
from base.generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(self,
                 configs: dict,
                 mode: str):
        super(Generator, self).__init__(configs, mode)
        self.configs = configs
        self.mode = mode

    def next_batch(self,
                   batch_size: int):
        pass


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)