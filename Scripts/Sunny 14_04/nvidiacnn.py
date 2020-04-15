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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#the nvidia cnn model. 
def nvidia_model():
    model = Sequential(name='Nvidia_Model')
    
    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data
    
    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu')) 
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu')) 
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu')) 
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='elu')) 
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='elu')) 

    model.add(Dropout(0.2))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    
    # output layer: return angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(2)) 
    
    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3) # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer)
    return model



#Nvidia model requires us to convert image into 2YUV and also must have dimensions of 
#200x66x3. The following function takes the array of images and applies processing to them.
def nvidia_img_preprocess(imagearray):
    newimagearray=np.zeros((len(imagearray),66,200,3))
    for i in range(len(imagearray)):
        image=imagearray[i,:,:,:]
        height, _, _ = image.shape
        image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relevant for lane following
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
        image = cv2.GaussianBlur(image, (3,3), 0)
        image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
        image = image / 255 # normalizing
        newimagearray[i,:,:,:]=image
    return newimagearray

model = nvidia_model()
print(model.summary())