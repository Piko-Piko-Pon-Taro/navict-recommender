# -*- coding: utf-8 -*-
"""CBOW Embedding"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel


class Net(nn.Module):
    """Network for CBOW"""
    """ CBOW """
    
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        """
        Args:
            vocab_size
            emb_size
        """

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        x = self.softmax(x)

        return x


class SimpleNN(BaseModel):
    """SimpleNN"""

    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)
        self.vocab_size = cfg.model.vocab_size
        self.emb_size = cfg.model.emb_size
        self.num_class = self.cfg.data.dataset.num_class

        self.network = Net(vocab_size=self.vocab_size, emb_size=self.emb_size)

        self.build()