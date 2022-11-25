import numpy as np

import torch
import torch.nn as nn

from .common_v2 import *

class EgoRnn(nn.Module):
	def __init__(self, input_dim, hidden_dim=1000, layers=2):
		super(EgoRnn, self).__init__()
		self.maxpool = nn.MaxPool2d(2)
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers)
		self.linear1 = nn.Linear(hidden_dim, 128)
		self.linear2 = nn.Linear(128, 6)

	def forward(self, x):
		x = self.maxpool(x)
		x = self.lstm(x)
		x = self.linear1(x)
		x = self.linear2(2)
		return x
    