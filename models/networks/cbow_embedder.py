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

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        """
        :param batch_X: torch.Tensor(dtype=torch.long), (batch_size, window*2)
        :param batch_Y: torch.Tensor(dtype=torch.long), (batch_size,)
        :return loss: torch.Tensor(dtype=torch.float), CBOWのloss
        """
        x = self.embedding(x) # (batch_size, window*2, embedding_size)
        x = torch.sum(x, dim=1) # (batch_size, embedding_size)
        x = self.linear(x) # (batch_size, vocab_size)
        x=  self.softmax(x) # (batch_size, vocab_size)

        return x


class CBOW(BaseModel):
    """CBOW"""

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