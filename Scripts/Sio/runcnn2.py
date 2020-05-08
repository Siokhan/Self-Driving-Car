#####
#this is the file to run the cnn. You have the ability to either use
#the nvidia cnn or the sio cnn. 
###

from functions import (detect_edges, detect_line_segments, plot_line_segments,
                       show_image, extract_data)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

n=3 #1 for nvidia cnn, 2 for sio cnn, 3 for speed model

#keras.models.load_model('model1.hf')


#extract data from folders.
data_dir1 = '../../Data/training_data/training_data'

#first we extract images
file_list = np.asarray(os.listdir(data_dir1))
a=0
x1 = []
for f in file_list:
    frame = cv2.imread(data_dir1 + '/' + f)
    x1.append(frame)
    #print(a)
    a+=1

x1=np.asarray(x1)
#print(x1.shape)
#now extract labels
labels = np.genfromtxt('../../Data/training_norm.csv',delimiter = ',')

y1 = labels[1:,1:]
#print(y1.shape)

data_dir2 = '../../Data/captureOVAL-28_02_2020'
x2,y2 = extract_data(data_dir2)
x2 = np.asarray(x2)
#print(x2.shape)

y2[:,0] = (y2[:,0] -50)/80
y2[:,1] = (y2[:,1])/35
#print(y2.shape)

data_dir3 = '../../Data/captureOVAL2-28_02_2020'
x3,y3 = extract_data(data_dir3)
x3 = np.asarray(x3)
#print(x3.shape)

y3[:,0] = (y3[:,0] -50)/80
y3[:,1] = (y3[:,1])/35
#print(y3.shape)

data_dir4 = '../../Data/captureOVAL03_03_2020'
x4,y4 = extract_data(data_dir4)
x4 = np.asarray(x4)
#print(x4.shape)

y4[:,0] = (y4[:,0] -50)/80
y4[:,1] = (y4[:,1])/35
#print(y4.shape)

data_dir5 = '../../Data/human_stopped_data'
x5,y5 = extract_data(data_dir5)
x5 = np.asarray(x5)
#print(x5.shape)

y5[:,0] = (y5[:,0] -50)/80
y5[:,1] = (y5[:,1])/35
#print(y5.shape)

x = np.append(x1,x2,axis = 0)
#x = np.append(x,x3,axis = 0)
x = np.append(x,x4,axis = 0)
x = np.append(x,x5,axis = 0)
print(x.shape)

y = np.append(y1,y2,axis=0)
#y = np.append(y,y3,axis=0)
y = np.append(y,y4,axis=0)
y = np.append(y,y5,axis=0)
print(y.shape)
y_angles = y[:,0]
y_speeds = y[:,1]
    


if n==1: #if nvidiacnn is selected
    from nvidiacnn import nvidia_model, nvidia_img_preprocess
    #normalize data and get it ready for cnn. 
    x = nvidia_img_preprocess(x)
    model = nvidia_model()
    #split into train and test
    x_train,  x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3)
elif n==2: #if siocnn is selected
    from siocnn import sio_model, sio_img_preprocess
    x = sio_img_preprocess(x)
    model = sio_model()
    #split into train and test
    x_train,  x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3)
elif n==3: #speed model
    from siospeedcnn import sio_model_speed, sio_img_preprocess_speed
    x = sio_img_preprocess_speed(x)
    model = sio_model_speed()
    #split into train and test
    x_train,  x_valid, y_train, y_valid = train_test_split(x,y_speeds,test_size=0.3)

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

history = model.fit(x=x_train, y=y_train, batch_size=10, shuffle = True,
                    epochs=20,validation_split=0.3)#, callbacks = [callback])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(('Train Loss','Valid Loss'),loc = 'best')
plt.xlabel('Epoch')
plt.ylabel('MSE')

print('Testing')

accuracy = model.evaluate(x=x_valid, y=y_valid, batch_size=2)

predictions_validation = model.predict(x_valid)

#save the model
model.save('sio_model_testing')

#plt.figure()

# predictions = 5 * np.round(predictions/5)

# replace repdictions with validation data for graphs
# mse = (predictions-y_valid)**2

#plt.plot(range(len(y_valid)),y_valid,'*')
#plt.plot(range(len(y_valid)),predictions,'r*')
#plt.legend(('Labels','Predictions'),loc = 'best')
#plt.xlabel('test example number')
#plt.ylabel('angle of rotation')