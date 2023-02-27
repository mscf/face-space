import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import PIL

labels = {
  "cat": [1, 0, 0],
  "dog": [0, 1, 0],
  "wild": [0, 0, 1]
}

class PetImage:

  def __init__(self, filename, image_class, shape=(128, 128)):
    self.image        = PIL.Image.open(filename)
    self.shape        = shape
    self.image_class  = image_class

  def gray(self):
    image   = self.image.convert("L")
    return self.to_array(image)

  def rgb(self):
    return self.to_array(self.image)

  def to_array(self, image):
    w_ratio = self.width / image.width
    h_ratio = self.height / image.width
    ratio   = w_ratio if w_ratio > h_ratio else h_ratio
    result  = image.resize((int(image.width*ratio), int(image.height*ratio)))
    left    = (result.width - self.width) / 2
    top     = (result.height - self.height) / 2
    return (np.array(result.crop((left, top, left + self.width, top + self.height))) / 255).astype("float")

  @property
  def width(self):
    return self.shape[0]

  @property
  def height(self):
    return self.shape[1]

  @property
  def label(self):
    return labels[self.image_class]
