# -*- coding: utf-8 -*-
"""SimpleRNN"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_model import BaseModel


class Net(nn.Module):
    """Network for SimpleRNN"""


    def __init__(self, vocab_size, emb_size, hidden_dim, num_layers, output_size) -> None:
        """Initialization

        Args:
            vocab_size: Vocabulary size for embedding.
            emb_size: Dimension of embedding.
            hidden_dim: Dimension of hidden state.
            num_layers: Number of RNN hidden layers.
            output_size: Output size.
        """

        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(emb_size, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.rnn(x.view(len(x[0]), -1, self.emb_size))
        x = self.fc(h.view(-1, self.hidden_dim))
        x = self.softmax(x)

        return x


class SimpleRNN(BaseModel):
    """SimpleRNN"""


    def __init__(self, cfg: object) -> None:
        """Initialization
    
        Build model.

        Args:
            cfg: Config.

        """

        super().__init__(cfg)
        self.vocab_size = self.cfg.model.vocab_size
        self.emb_size = self.cfg.model.emb_size
        self.hidden_dim = self.cfg.model.hidden_dim
        self.num_layers = self.cfg.model.num_layers
        self.output_size = self.cfg.model.output_size

        self.network = Net(
            vocab_size=self.vocab_size, 
            emb_size=self.emb_size, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers, 
            output_size=self.output_size
            )

        self.build()