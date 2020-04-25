#####
#this is the file to run the cnn. You have the ability to either use
#the nvidia cnn or the sunny cnn. Thought it would be cool to be able to compare 
#our cnn to nvidias.

###

from functions import (detect_edges, detect_line_segments, plot_line_segments,
                       show_image, extract_data, traintestsplit)
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras import Model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


n=4 #1 for nvidia cnn, 2 for sunny cnn, 3 for edges...

#keras.models.load_model('model1.hf')

#extract data from folders.
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
labels = np.genfromtxt('../../Data/training_norm.csv',delimiter = ',')

y1 = labels[1:,1:]
#cat_labels = np.append(to_categorical(y1[:,0] * 16), to_categorical(y1[:,1]), axis = 1)

data_dir2 = '../../Data/training_data_full'
x2,y2 = extract_data(data_dir2)
x2 = np.asarray(x2)

y2[:,0] = (y2[:,0] -50)/80
y2[:,1] = (y2[:,1])/35
y2[:, 1][y2[:, 1] > 1] = 1
x2 = x2[abs(y2[:,0] -0.5) > 0.07]
y2 = y2[abs(y2[:,0] -0.5) > 0.07]


data_dir3= '../../Data/human_stopped_data'
x3,y3 = extract_data(data_dir3)
x3 = np.asarray(x3)

y3[:,0] = (y3[:,0] -50)/80
y3[:,1] = (y3[:,1])/35
y3[:, 1][y3[:, 1] > 1] = 1

x = ~np.vstack((x1, x2, x3))
y = np.vstack((y1, y2, y3))
x = x[:, 100:, :, :]

y = y[:, 1]

#for i in range(len(x)):
#    dest_dir = '../../Data/training_data_balanced'
#    string = '/%d.png' % i
#    cv2.imwrite(dest_dir + string , x[i])

#np.savetxt('../../Data/training_balanced_y.csv', y, delimiter = ',', fmt = '%.5f')

if n==1: #if nvidiacnn is selected
    from nvidiacnn import nvidia_model, nvidia_img_preprocess
    #normalize data and get it ready for cnn. 
    x = nvidia_img_preprocess(x)
    model = nvidia_model()
elif n==2: #if sunnycnn is selected
    from sunnycnn import sunny_model, sunny_img_preprocess
    x = sunny_img_preprocess(x)
    model = sunny_model()
elif n==3:
    from edgescnn import edges_model, edges_img_preprocess
    x = edges_img_preprocess(x)
    model = edges_model()
elif n==4:
    from joshModel1 import josh_model, josh_data_generator, test_generator
    data_gen = josh_data_generator()
    valid_data_gen = test_generator()
    model = josh_model()
elif n==5:
    from joshModel2 import josh_model, josh_data_generator, test_generator
    data_gen = josh_data_generator()
    model = josh_model()
    valid_data_gen = test_generator()

#split into train and test
x_train,  y_train, x_valid, y_valid = traintestsplit(x, y, 0.2)
#y_train = to_categorical(y_train_full[:,0]*16, num_classes=17)
#y_valid = to_categorical(y_valid_full[:,0]*16, num_classes=17)

#y_train = to_categorical(y_train_full[:,0], num_classes=2)
#y_valid = to_categorical(y_valid_full[:,0], num_classes=2)


# # Directory where the checkpoints will be saved
# checkpoint_dir = './training_checkpoints'
# # Name of the checkpoint files
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)

BATCH_SIZE = 16
TRAIN_SAMPLES = len(x_train)
VALID_SAMPLES = len(x_valid)

train_generator = data_gen.flow(x_train, y_train, batch_size = BATCH_SIZE)
valid_generator = valid_data_gen.flow(x_valid, y_valid, batch_size = BATCH_SIZE)
callback = EarlyStopping(monitor = 'val_loss',
                         patience = 10,
                         restore_best_weights = True)

history = model.fit(x = train_generator,
                    epochs = 100,
                    shuffle = True,
                    steps_per_epoch = TRAIN_SAMPLES//BATCH_SIZE,
                    validation_data = valid_generator,
                    validation_steps = VALID_SAMPLES//BATCH_SIZE,
                    callbacks = [callback])


model.save('CNN_model')
data_dir3 = '../../Data/test_data/test_data'

file_list = np.asarray(os.listdir(data_dir3))
a=0
x = []
for f in file_list:
    frame = cv2.imread(data_dir3 + '/' + f)
    x.append(frame)
    #print(a)
    a+=1

x_test=~np.asarray(x)/255
x_test = x_test[:, 100:, :, :]

predictions = model.predict(x_test, batch_size=None)
#np.savetxt('../../Data/josh_submission.csv', y_test, delimiter = ',', fmt = '%.5f')
np.savetxt('../../Data/josh_submission.csv', predictions, delimiter = ',', fmt = '%.5f')

