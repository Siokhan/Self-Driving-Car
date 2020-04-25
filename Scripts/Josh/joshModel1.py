# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:37:47 2020

@author: swift
"""

# python standard libraries
import os
import random
import fnmatch
import datetime
import pickle

# data processing
import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})
# tensorflow
import tensorflow 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def label_images(directory):
    return 0


def josh_data_generator():
    imgGen = ImageDataGenerator(width_shift_range=10,
                               height_shift_range=10,
                               rotation_range = 0,
                               rescale=1./255,
                               fill_mode = 'constant',
                               cval = 0)
    return imgGen
    
def test_generator():
    imgGen = ImageDataGenerator(rescale=1./255)
    return imgGen

def josh_model():
    model = Sequential(name = 'Josh_Model')
    
    n = 16
    
    model.add(Conv2D(n, (5,5), strides = (1, 1), activation = 'elu', padding = 'same',
                     input_shape = (140, 320, 3),))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(2 * n, (5,5), strides = (2, 2), activation = 'elu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(4 * n, (3,3), strides = (1, 1), activation = 'elu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(8 * n, (3,3), strides = (1, 1), activation = 'elu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16 * n, (3,3), strides = (1, 1), activation = 'elu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(500, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(1))
    
    model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
    
    return model