# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:28:26 2020

@author: swift
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

from functions import (detect_edges, detect_line_segments, plot_line_segments, 
                       show_image, extract_data, traintestsplit)

data_dir1 = '../../Data/training_data/training_data'

#first we extract images
file_list = np.asarray(os.listdir(data_dir1))
a=0
x = []
for f in file_list:
    frame = cv2.imread(data_dir1 + '/' + f)
    x.append(frame)
    #print(a)
    a+=1

x1=np.asarray(x)
#now extract labels
y1 = np.genfromtxt('../../Data/training_norm.csv',delimiter = ',')[1:, 1:]

data_dir2 = '../../Data/training_data_full'
x2,y2 = extract_data(data_dir2)
x2 = np.asarray(x2)

y2[:,0] = (y2[:,0] -50)/80
y2[:,1] = (y2[:,1])/35

x = np.append(x1, x2, axis = 0)
y = np.append(y1, y2, axis = 0)
train_data, train_labels, valid_data, valid_labels = traintestsplit(x, y, 0.3)
np.savetxt('../../Data/training_data_labels.csv', y, delimiter = ',', fmt = '%.5f')
