# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:30:34 2020

@author: Siokhan Kouassi
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import math

#functions for image pre processing

#### helper functions ####
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

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

#### lane edge detection ####
## adapted from: https://towardsdatascience.com/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96
def detect_edges(frame):
    #filter for black lane markings
    #first increase contrast in order to filter out noise
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #invert pixels as it facilitates detection
    hsv = cv2.cvtColor(~frame, cv2.COLOR_BGR2HSV)
    #show_image("hsv", hsv)
    #image is inverted so filter out white instead of black
    lower_black = np.array([0, 0, 253])
    upper_black = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen and cut off the car bumper
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, 220),
        (0, 220),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)
    #print(line_segments)

    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def detect_lane(frame):
    
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines

def compute_steering_angle(frame, lane_lines):

    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logging.debug('new steering angle: %s' % steering_angle)
    return steering_angle

#display lane lines on either side of the track
def display_lines(frame, lines, line_color=(239, 245, 66), line_width=4):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

#display middle line, essentially the trajecttory the car should , hence takes current optimal steering angle as arguments
def display_heading_line(frame, steering_angle, line_color=(0, 255, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image