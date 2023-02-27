import torch.nn as nn
import torch.nn.functional as F
import torch
from math import prod


class Classifier(nn.Module):

  def __init__(self, shape):
    super().__init__()
    self.shape  = shape
    self.input  = nn.Conv2d(1, 20, 7, padding=1)
    self.pool   = nn.MaxPool2d(2, 2)
    self.hidden = nn.LazyConv2d(10, 7, padding=1)
    self.output = nn.Linear(1690, 3)

  def forward(self, x):
    #print(x.shape)
    batch = x.shape[0]
    x = F.relu(self.input(x))
    x = self.pool(x)
    x = F.relu(self.hidden(x))
    x = self.pool(x)
    #print(x.shape)
    x = F.softmax(self.output(x.reshape((batch, -1))))
    return x
