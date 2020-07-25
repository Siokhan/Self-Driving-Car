# python standard libraries
import Nvidia_Model
import os
import re
import random
import fnmatch



import matplotlib.pyplot as plt

# data processing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from imgaug import augmenters as img_aug
from random import randint

# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from keras.models import load_model
from keras import models

def mse_steering(y_true, y_pred):
    return K.mean(K.square(y_pred[:, 0] - y_true[:, 0]))

def mse_speed(y_true, y_pred):
    return K.mean(K.square(y_pred[:, 1] - y_true[:, 1]))

def main():
    data_dir = 'Combined_DataSet'
    file_list = os.listdir(data_dir)
    predictions = []
    model = load_model('lane_navigation_check_CombinedDataSet(max_pool_FF).h5', custom_objects={'mse_steering': mse_steering, 'mse_speed': mse_speed})
    image_id = []

    image = cv2.imread(os.path.join(data_dir, '63723177491890_90_0.png'))
    processed_image = Nvidia_Model.img_preprocess(image)
    image_tensor = np.asarray([processed_image])

    layer_outputs = [layer.output for layer in model.layers[:7]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input,
                                    outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(image_tensor )
    # first_layer_activation = activations[0]
    # print(first_layer_activation.shape)
    # plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    # plt.show()

    layer_names = []
    for layer in model.layers[:7]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 24

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size_y = layer_activation.shape[1]  # The feature map has shape (1, size_y, size_x, n_features).
        size_x = layer_activation.shape[2]
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size_y * n_cols, images_per_row * size_x))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size_y: (col + 1) * size_y,  # Displays the grid
                    row * size_x: (row + 1) * size_x] = channel_image
        scale_y = 1. / size_y
        scale_x = 1. / size_x
        plt.figure(figsize=(scale_y * display_grid.shape[1],
                            scale_x * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

if __name__ == '__main__':
    main()