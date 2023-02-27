import torch.nn as nn
import torch.nn.functional as F
import torch
from math import prod

class VAE(nn.Module):
  def __init__(self, shape):
    super(VAE, self).__init__()
    self.shape = shape
    self.fc1 = nn.Linear(prod(self.shape), 400)
    self.fc21 = nn.Linear(400, 2)
    self.fc22 = nn.Linear(400, 2)
    self.fc3 = nn.Linear(2, 400)
    self.fc4 = nn.Linear(400, prod(self.shape))

  def encode(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

  def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, prod(self.shape)))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar
