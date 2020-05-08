# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:06:11 2020

@author: Siokhan Kouassi
"""

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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import image_filtering_functions as pre


#my experimental cnn model. 
def sio_model_speed():
    model = Sequential(name='Sio_model_speed')
    
    model.add(Dense(100, input_dim=(200, 320, 3), activation = 'elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    
    # output layer: return speed
    model.add(Dense(1, activation='softmax'))
    
    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-4) # lr is learning rate
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return model


#from looking at the images i decided to crop top 100 pixels and normalize. 
def sio_img_preprocess_speed(imagearray):
    newimagearray=np.zeros((len(imagearray),200,320,3))
    for i in range(len(imagearray)):
        image=imagearray[i,:,:,:]
        height, _, _ = image.shape
        lane_lines = pre.detect_lane(image)
        steering_angle = pre.compute_steering_angle(image, lane_lines)
        image = pre.display_heading_line(image, steering_angle)
        image = pre.display_lines(image, lane_lines)
        image = image[int(height/6):,:,:]  # remove top half of the image, as it is not relevant for lane following
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
        #image = cv2.GaussianBlur(image, (3,3), 0)
        image = image / 255 # normalizing
        newimagearray[i,:,:,:]=image
    return newimagearray


model = sio_model_speed()
print(model.summary())