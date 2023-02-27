import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import PIL
import glob

from pet_image import PetImage

class PetIterator:

  def __init__(self, filenames, classes, shape=(128, 128)):
    self.filenames  = filenames
    self.classes    = classes
    self.shape      = shape

  def __call__(self):
    for i in range(len(self.filenames)):
      yield PetImage(self.filenames[i], self.classes[i], self.shape)

  def __len__(self):
    return len(self.filenames)

class PetLoader:
    
  def __init__(self, root="PetImages", seed=1337, shape=(128, 128)):
    cats = [("cat", c) for c in sorted(glob.glob(f"./{root}/Cat/*.jpg"))]
    dogs = [("dog", d) for d in sorted(glob.glob(f"./{root}/Dog/*.jpg"))]
    wild = [("wild", d) for d in sorted(glob.glob(f"./{root}/Wild/*.jpg"))]
    self._cats = np.array(cats)
    self._dogs = np.array(dogs)
    self._wild = np.array(wild)
    self.rng = np.random.default_rng(seed)
    self.rng.shuffle(self._cats)
    self.rng.shuffle(self._dogs)
    self.rng.shuffle(self._wild)

  def train(self):
    items = np.append(self._cats[:int(len(self._cats)*.8),:],
                      self._dogs[:int(len(self._dogs)*.8),:], axis=0)
    items = np.append(items,
                      self._wild[:int(len(self._wild)*.8),:], axis=0)
    self.rng.shuffle(items)
    return PetIterator(items[:,1], items[:, 0])

  def validate(self):
    items = np.append(self._cats[int(len(self._cats)*.8):,:],
                      self._dogs[int(len(self._dogs)*.8):,:], axis=0)
    items = np.append(items,
                      self._wild[int(len(self._wild)*.8):,:], axis=0)
    self.rng.shuffle(items)
    return PetIterator(items[:,1], items[:, 0])

  def train_cats(self):
    items = self._cats[:int(len(self._cats)*.8),:]
    self.rng.shuffle(items)
    return PetIterator(items[:,1], items[:, 0])
