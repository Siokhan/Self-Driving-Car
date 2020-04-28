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
def sio_model():
    model = Sequential(name='Sio_model')
    
    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data
    
    # Convolution Layers
    model.add(Conv2D(32, (5, 5), strides=(2, 2), input_shape=(200, 320, 3), activation='elu')) 
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='elu')) 
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='elu'))
    model.add(Conv2D(24, (3, 3), strides=(2, 2), activation='elu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(24, (2, 2), strides=(2, 2), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,1)))
    model.add(Dropout(0.2))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    
    # output layer: return angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(2)) 
    
    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3) # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer)
    
    return model


#from looking at the images i decided to crop top 100 pixels and normalize. 
def sio_img_preprocess(imagearray):
    newimagearray=np.zeros((len(imagearray),200,320,3))
    for i in range(len(imagearray)):
        image=imagearray[i,:,:,:]
        height, _, _ = image.shape
        lane_lines = pre.detect_lane(image)
        image = pre.display_lines(image, lane_lines)
        image = image[int(height/6):,:,:]  # remove top half of the image, as it is not relevant for lane following
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
        #image = cv2.GaussianBlur(image, (3,3), 0)
        image = image / 255 # normalizing
        newimagearray[i,:,:,:]=image
    return newimagearray


model = sio_model()
print(model.summary())