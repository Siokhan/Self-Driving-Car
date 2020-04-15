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
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from functions import detect_edges
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#my experimental cnn model. 
def edges_model():
    model = Sequential(name='Sunny_Model')
    
    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data
    
    # Convolution Layers
    model.add(Conv1D(24, 5, strides=2, input_shape=(140, 320), activation='elu')) 
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(24, 5, strides=2, activation='elu')) 
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(24, 3, strides=1, activation='elu')) 
    model.add(MaxPool1D(pool_size=2))
    

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



#from looking at the images i decided to crop top 100 pixels and normalize. 
def edges_img_preprocess(imagearray):
    newimagearray=np.zeros((len(imagearray),140,320))
    for i in range(len(imagearray)):
        image=imagearray[i,:,:,:]
        edges = detect_edges(image)
        edges = edges[100:,:]
        newimagearray[i,:,:]=edges
    return newimagearray

model = edges_model()
print(model.summary())