# -*- coding: utf-8 -*-
"""Navict dataset"""

import os
import logging
import json
import random

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
        num_roadmaps = 1000
        library_ids = [ i for i in range(6, 106)]
        nums = [ i for i in range(0, 10)]
        roadmaps = []
        num_category_unit = 20

        for i in range(num_roadmaps):
            c = i // num_category_unit
            pool = library_ids[ c * num_category_unit: c * (num_category_unit+1)]
            if len(pool) < num_category_unit:
                pool = library_ids[-num_category_unit:]
            l = random.sample( pool, 10)
            # l.sort()
            shuffle_ids = random.sample(nums, 2)
            tmp = l[shuffle_ids[0]]
            l[shuffle_ids[0]] = l[shuffle_ids[1]]
            l[shuffle_ids[1]] = tmp
            roadmaps.append(l)
        # data generation for test: end

        x = []
        y = []

        for roadmap in roadmaps:
            roadmap = [0, 0] + roadmap

            for i in range(len(roadmap) - 3):
                x.append(roadmap[i:i+3])
                y.append(roadmap[i+3])

        self.x = torch.tensor(x)
        self.y = torch.tensor(y).long()
        self.data_len = len(self.x)


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        out_data = self.x[idx]
        out_label =  self.y[idx]

        return out_data, out_label