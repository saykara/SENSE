import numpy as np

import torch
import torch.nn as nn

from .common_v2 import *

class EgoRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000, layers=2, linear_dim=128):
        super(EgoRnn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.linear_dim = linear_dim
        
        self.maxpool = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layers)
        self.linear1 = nn.Linear(self.hidden_dim, self.linear_dim)
        self.linear2 = nn.Linear(self.linear_dim, 6)

    def forward(self, seqs):
        for i in range(seqs.size(1)):
            item = seqs[:, i, :, :, :]
            item = self.maxpool(item)
            item = item.view(item.size(0), item.size(1), item.size(2) * item.size(3))
            hn, cn = self.init_states(seqs.size(2))
            out, (hn, cn) = self.lstm(item, (hn, cn))
        x = self.linear1(out[:, -1, :])
        x = self.linear2(x)
        return x

    def init_states(self, size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h0 = torch.zeros(self.layers, size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layers, size, self.hidden_dim).to(device)
        return h0, c0
    