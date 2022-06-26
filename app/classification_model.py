import os  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt

import pathlib
import PIL
import PIL.Image
import numpy as np

batch_size = 5
img_height = 224
img_width = 224


def read_data(data_dir='dataset'):
  return pathlib.Path('dataset')
#data_dir = read_data()

def train_validation_split(data_dir):
  """
  :param data_dir is the returned value from read_data function
  """
  train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  class_names = train_ds.class_names
  return train_ds,val_ds,class_names

def transform_data(train_ds):
  normalization_layer = tf.keras.layers.Rescaling(1./255)
  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  return normalized_ds

def train_model(normalized_ds,val_ds,epochs=5):
  model = tf.keras.applications.MobileNet()
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history = model.fit(
  normalized_ds,
  validation_data=val_ds,
  epochs=epochs
  )
  return model

def save_model(model):  
  #TO CHANGE: change saved location according to version
  model.save('my_model/1') 




