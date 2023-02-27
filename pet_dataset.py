import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pet_loader import PetIterator
import glob

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    image = sample['image']

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    image = image.transpose((2, 0, 1))
    return {'image': torch.from_numpy(image), "label": torch.tensor(sample["label"])}

class ToTensorGray(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    image = sample['image']

    return {'image': torch.from_numpy(image), "label": torch.tensor(sample["label"])}


class PetDataset(Dataset):

  def __init__(self, root, transform=ToTensor(), seed=1137, shape=(64, 64)):
    cats = [("cat", c) for c in sorted(glob.glob(f"./{root}/cat/*.jpg"))]
    dogs = [("dog", d) for d in sorted(glob.glob(f"./{root}/dog/*.jpg"))]
    wild = [("wild", d) for d in sorted(glob.glob(f"./{root}/wild/*.jpg"))]
    self._cats = np.array(cats)
    self._dogs = np.array(dogs)
    self._wild = np.array(wild)
    self.rng = np.random.default_rng(seed)
    self.rng.shuffle(self._cats)
    self.rng.shuffle(self._dogs)
    self.rng.shuffle(self._wild)
    self.transform = transform
    self.shape = shape
    self.iterator = None
    i = [(np.array(i.gray()), i.label) for i in self.train()()]
    print(type(i[0]))
    self.items = np.array([it[0] for it in i])
    self.labels = np.array([it[1] for it in i])

  def train(self):
    items = np.append(self._cats[:int(len(self._cats)*.8),:],
                      self._dogs[:int(len(self._dogs)),:], axis=0)
    items = np.append(items,
                      self._wild[:int(len(self._wild)*.5),:], axis=0)
    self.rng.shuffle(items)
    return PetIterator(items[:,1], items[:, 0], shape=self.shape)

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    sample = {"image": self.items[idx], "label": self.labels[idx]}
    if self.transform is not None:
      sample = self.transform(sample)
    return sample
