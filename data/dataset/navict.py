# -*- coding: utf-8 -*-
"""Navict dataset"""

import os
import logging
import json
import random
import math

import torch
import numpy as np
from torchvision import transforms
import requests


log = logging.getLogger(__name__)


class Navict(torch.utils.data.Dataset):
    """Navict dataset"""

    def __init__(self, cfg: object, mode: str) -> None:
        """Initialization
    
        Get dataset.
        Args:
            cfg: Config.
            mode: Mode. 
                trainval: For trainning and validation.
                test: For test.
        """

        json_open = open('/workspace/datasets/navict/data.json', 'r')
        roadmaps = json.load(json_open)

        # data generation for test: begin
        # num_roadmaps = 100
        # library_ids = [ i for i in range(6, 106)]
        # nums = [ i for i in range(0, 10)]
        # roadmaps = []
        # num_category_unit = 20
        # num_pool = math.ceil(len(library_ids) // num_category_unit)

        # pools = []
        # for i in range(num_pool):
        #     pool = library_ids[i * num_category_unit: (i+1) * num_category_unit]
        #     if len(pool) < num_category_unit:
        #         pool = library_ids[-num_category_unit:]
        #     pools.append(pool)

        # for i in range(num_roadmaps):
        #     pool = pools[i % num_pool]
        #     l = random.sample( pool, 10)
        #     l.sort()
        #     # shuffle_ids = random.sample(nums, 2)
        #     # tmp = l[shuffle_ids[0]]
        #     # l[shuffle_ids[0]] = l[shuffle_ids[1]]
        #     # l[shuffle_ids[1]] = tmp
        #     roadmaps.append(l)
        # data generation for test: end

        x = []
        y = []
        len_input = cfg.data.dataset.len_input

        for roadmap in roadmaps:
            roadmap = [0]*len_input + roadmap

            for i in range(len_input, len(roadmap)):
                x.append(roadmap[i-len_input:i])
                y.append(roadmap[i])

        self.x = torch.tensor(x)
        self.y = torch.tensor(y).long()
        self.data_len = len(self.x)


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        out_data = self.x[idx]
        out_label =  self.y[idx]

        return out_data, out_label