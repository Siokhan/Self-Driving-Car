# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:30:34 2020

@author: Siokhan Kouassi
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

#functions for image pre processing

#function to create a figure of 2 images
def double_figure(width, height, image1, image2, title1, title2, share_axis=True,
                  axis='off', tight=False):
    
    fig, (im1, im2) = plt.subplots(nrows=1, ncols=2, figsize=(width,height),
          sharex=share_axis, sharey=share_axis)
    
    im1.imshow(image1)
    im1.axis(axis)
    im1.set_title(title1)

    im2.imshow(image2)
    im2.axis(axis)
    im2.set_title(title2)
    
    if(tight==True):
        fig.tight_layout()
    
    return fig

#function to create a figure of 3 images
def triple_figure(width, height, image1, image2, image3, title1, title2, title3, 
                  share_axis=True, axis='off', tight=False):
    
    fig, (im1, im2, im3) = plt.subplots(nrows=1, ncols=3, figsize=(width,height),
          sharex=share_axis, sharey=share_axis)
    
    im1.imshow(image1)
    im1.axis(axis)
    im1.set_title(title1)

    im2.imshow(image2)
    im2.axis(axis)
    im2.set_title(title2)

    im3.imshow(image3)
    im3.axis(axis)
    im3.set_title(title3)
    
    if(tight==True):
        fig.tight_layout()
    
    return fig

#function to generate figure with 3 subplots and outline template match
def template_match_img(width, height, template, detected_img, result, peaks, 
                       title1, title2, title3, share_axis=True, axis='off', tight=False):
    
    fig, (im1, im2, im3) = plt.subplots(nrows=1, ncols=3, figsize=(width,height),
          sharex=share_axis, sharey=share_axis)
    im1.imshow(template, cmap=plt.cm.gray)
    im1.axis(axis)
    im1.set_title(title1)
    
    im2.imshow(detected_img, cmap=plt.cm.gray)
    im2.axis(axis)
    im2.set_title(title2)
    # highlight matched region
    hcoin, wcoin = template.shape
    for i in peaks:
        rect = plt.Rectangle((i[1], i[0]), wcoin, hcoin, edgecolor='r', facecolor='none')
        im2.add_patch(rect)
    
    im3.imshow(result)
    im3.axis(axis)
    im3.set_title(title3)
    # highlight matched region on result
    im3.autoscale(False)
    im3.plot(peaks[:,1], peaks[:,0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    
    if(tight==True):
        fig.tight_layout()
    
    return fig

#lane edge detection
def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(~frame, cv2.COLOR_BGR2HSV)
    #show_image("hsv", hsv)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 230])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    #show_image("black mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges
