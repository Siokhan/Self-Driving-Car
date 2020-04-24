# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:30:34 2020

@author: Siokhan Kouassi
"""

import matplotlib.pyplot as plt
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
