# -*- coding: utf-8 -*-
"""CBOW Embedding"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel
from models.networks.cbow_embedder import Net as CBOW


class Net(nn.Module):
    """Network for CBOW"""
    """ CBOW """
    
    def __init__(self, embedder):
        super().__init__()
        """
        Args:
            vocab_size
            emb_size
        """

        self.embedding = embedder.embedding
        self.embedding.weight.requires_grad = False
        self.emb_size = embedder.emb_size
        self.vocab_size = embedder.vocab_size

        self.net = nn.Sequential(
            nn.Linear(self.emb_size, 128, bias=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(128, self.vocab_size, bias=False),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.net(x)

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

        self.embedder = CBOW(vocab_size=self.cfg.model.embedder.vocab_size, emb_size=self.cfg.model.embedder.emb_size)
        ckpt_path = self.cfg.model.embedder.initial_ckpt
        if torch.cuda.is_available():
            ckpt = torch.load(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, torch.device('cpu'))
            
        self.embedder.load_state_dict(ckpt['model_state_dict'])

        self.num_class = self.cfg.data.dataset.num_class

        self.network = Net(embedder=self.embedder)

        self.build()