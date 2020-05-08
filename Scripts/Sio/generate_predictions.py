# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:00:01 2020

@author: Siokhan Kouassi
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd

n=3 #1 for nvidia cnn, 2 for sio cnn, 3 for speed model

#load model
model = tf.keras.models.load_model('sio_model_testing')

#print(model.summary())

#extract test data
test_data_dir = '../../Data/test_data/test_data'
test_file_list = np.asarray(os.listdir(test_data_dir))
b = 0
x_test = []
test_image_id = []
for f in test_file_list:
    frame = cv2.imread(test_data_dir + '/' + f)
    x_test.append(frame)
    test_image_id.append(f.split('.')[0])
    #print(b)
    b+=1

x_test = np.asarray(x_test)

if n==1: #if nvidiacnn is selected
    from nvidiacnn import nvidia_img_preprocess 
    x_test = nvidia_img_preprocess(x_test)
    print('Test data pre - processed')
    #make predictions
    predictions_test = model.predict(x_test)
    test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])
    angles = pd.DataFrame(predictions_test[:,0], columns = ['angle'])
    speeds = pd.DataFrame(predictions_test[:,1], columns = ['speed'])
    #making sure no speed is above 1
    for index, row in speeds.iterrows():
        if row['speed'] > 1:
            speeds.iloc[index] = 1
        
    output = pd.concat([test_image_id, angles, speeds], axis = 1, sort = False)
    output.sort_values(by=['image_id'], inplace = True)
    #output.to_csv('predictions_4.csv', index = False)
    
elif n==2: #if siocnn is selected
    from siocnn import sio_img_preprocess
    x_test = sio_img_preprocess(x_test)
    print('Test data pre - processed')
    #make predictions
    predictions_test = model.predict(x_test)
    test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])
    angles = pd.DataFrame(predictions_test[:,0], columns = ['angle'])
    speeds = pd.DataFrame(predictions_test[:,1], columns = ['speed'])
    #making sure no speed is above 1
    for index, row in speeds.iterrows():
        if row['speed'] > 1:
            speeds.iloc[index] = 1
        
    output = pd.concat([test_image_id, angles, speeds], axis = 1, sort = False)
    output.sort_values(by=['image_id'], inplace = True)
    #output.to_csv('predictions_4.csv', index = False)
    
elif n==3: #speed model
    from edgescnn import edges_img_preprocess
    x_test = edges_img_preprocess(x_test)
    print('Test data pre - processed')
    #make predictions
    predictions_test = model.predict(x_test)
    test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])
    speeds = pd.DataFrame(predictions_test[:,0], columns = ['speed'])
    #making sure no speed is above 1
    for index, row in speeds.iterrows():
        if row['speed'] > 1:
            speeds.iloc[index] = 1
        
    output = pd.concat([test_image_id, speeds], axis = 1, sort = False)
    output.sort_values(by=['image_id'], inplace = True)
    #output.to_csv('predictions_4.csv', index = False)
'''
#make predictions
predictions_test = model.predict(x_test)
test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])
angles = pd.DataFrame(predictions_test[:,0], columns = ['angle'])
speeds = pd.DataFrame(predictions_test[:,1], columns = ['speed'])

#making sure no speed is above 1
for index, row in speeds.iterrows():
    if row['speed'] > 1:
        speeds.iloc[index] = 1
        
output = pd.concat([test_image_id, angles, speeds], axis = 1, sort = False)
output.sort_values(by=['image_id'], inplace = True)
#output.to_csv('predictions_4.csv', index = False)
'''
print('predictions generated')