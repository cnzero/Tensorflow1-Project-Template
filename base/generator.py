# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-06
# @author: cnzero

"""generator abstract base class architecture
"""


import numpy as np


class BaseGenerator(object):
    def __init__(self,
                 configs: dict,
                 mode: str):
        """basic settings

        Args:
            configs (dict): get dict parameters from configs files
            mode (str): train/test/validation/predict
        """
        self.configs = configs
        self.mode = mode

    def next_batch(self,
                   batch_size: int):
        # yield images, labels
        raise NotImplementedError


if __name__ == '__main__':
    print('self-test Hello World in ', __file__)