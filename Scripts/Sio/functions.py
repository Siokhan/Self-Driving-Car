# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:42:09 2020

@author: ppysh3
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


def detect_edges(frame):
    frame = np.uint8(frame)
    edges = cv2.Canny(frame, 100, 200)
    return edges

def detect_line_segments(edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

#to detect if there is a number in a string.
def anynumber(string):
    return any(i.isdigit() for i in string)

def plot_line_segments(line_segments,frame):
    croppedframe = frame[100:,40:280,:]
    for t in range(len(line_segments[:,0,1])):
        plt.plot([line_segments[t,0,0],line_segments[t,0,2]],[line_segments[t,0,1],line_segments[t,0,3]],'r-')
        plt.imshow(croppedframe)
    plt.show
    
def show_image(image): #(for debugging)
    cv2.imshow( "Debug", image );   
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
#extracts data and detects the edges. 
def extract_data(data_directory):
    file_list = np.asarray(os.listdir(data_directory))
    
    data = []
    angles=[]
    speeds = []

    #we need to only keep files that have numbers in them
    logicarray=[]
    for f in file_list:
        logicarray=np.append(logicarray,anynumber(f))
        
    file_list=file_list[logicarray==1]
    
    for f in file_list:
        frame = cv2.imread(data_directory + '/' + f)
       # edges=detect_edges(frame)
        #data.append(edges)
        data.append(frame)
        angle = int(''.join(i for i in f[14:17] if i.isdigit()))
        angles.append(angle)
        speed= int(''.join(i for i in f[17:20] if i.isdigit()))
        speeds.append(speed)
    labels = np.zeros((len(speeds),2))
    labels[:,0] = angles
    labels[:,1] = speeds
    return data, labels

def traintestsplit(x,y):
    N = len(y)
    x = np.asarray(x)
    y = np.asarray(y)
    rand = random.sample(range(N),N)
    trainsplit = int(N*0.7)
    testsplit = int(N*0.3)
    train_data = x[rand[:trainsplit]]
    test_data = x[rand[trainsplit:]]
    train_labels = y[rand[:trainsplit]]
    test_labels = y[rand[trainsplit:]]
    return train_data, train_labels, test_data, test_labels
    

































