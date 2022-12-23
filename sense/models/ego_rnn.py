import numpy as np

import torch
import torch.nn as nn

from .common_v2 import *

class EgoRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000, layers=2):
        super(EgoRnn, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        
        self.maxpool = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers)
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, 6)

    def forward(self, x, hn, cn):
        x = self.maxpool(x)
        x, (hn, cn) = self.lstm(x, (hn, cn))
        x = self.linear1(x)
        x = self.linear2(x)
        return x, hn, cn

    def init_states(self, batch_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        h0 = torch.zeros(self.layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layers, batch_size, self.hidden_dim).to(device)
        return h0, c0
    