# !/Users/username/anaconda3/envs/conda_env/bin/python
# -*- coding: utf-8 -*-
# @time: 2022-05-02
# @author: cnzero

import yaml


def parse_yml(yml_file):
    with open(yml_file, 'rb') as f:
        configs = yaml.safe_load(f)

    return configs


if __name__ == '__main__':
    print('self-testing Hello World in ', __file__)

    yml_file = 'configs/base_config.yml'
    print(parse_yml(yml_file=yml_file))
