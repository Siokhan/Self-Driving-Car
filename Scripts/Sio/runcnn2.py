#####
#this is the file to run the cnn. You have the ability to either use
#the nvidia cnn or the sunny cnn. Thought it would be cool to be able to compare 
#our cnn to nvidias.

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

n=2 #1 for nvidia cnn, 2 for sunny cnn, 3 for edges...

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
print(x1.shape)
#now extract labels
labels = np.genfromtxt('../../Data/training_norm.csv',delimiter = ',')

y1 = labels[1:,1:]
print(y1.shape)

data_dir2 = '../../Data/captureOVAL-28_02_2020'
x2,y2 = extract_data(data_dir2)
x2 = np.asarray(x2)
print(x2.shape)

y2[:,0] = (y2[:,0] -50)/80
y2[:,1] = (y2[:,1])/35
print(y2.shape)

x = np.append(x1,x2,axis = 0)
y = np.append(y1,y2,axis=0)
print(x.shape)
print(y.shape)

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
    from nvidiacnn import nvidia_model, nvidia_img_preprocess
    #normalize data and get it ready for cnn. 
    x = nvidia_img_preprocess(x)
    x_test = nvidia_img_preprocess(x_test)
    model = nvidia_model()
elif n==2: #if sunnycnn is selected
    from sunnycnn import sunny_model, sunny_img_preprocess
    x = sunny_img_preprocess(x)
    x_test = sunny_img_preprocess(x_test)
    model = sunny_model()
elif n==3:
    from edgescnn import edges_model, edges_img_preprocess
    x = edges_img_preprocess(x)
    x_test = edges_img_preprocess(x_test)
    model = edges_model()
    
    
#split into train and test
x_train,  x_valid, y_train, y_valid = train_test_split(x,y,test_size=0.3)

# # Directory where the checkpoints will be saved
# checkpoint_dir = './training_checkpoints'
# # Name of the checkpoint files
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)

#introducing early stopping after 3 epochs of no improvement
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 2)

history = model.fit(x=x_train, y=y_train, batch_size=10, shuffle = True,
                    epochs=20,validation_split=0.3, callbacks = [callback])#,  callbacks=checkpoint_callback

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(('Train Loss','Valid Loss'),loc = 'best')
plt.xlabel('Epoch')
plt.ylabel('MSE')

print('Testing')

accuracy = model.evaluate(x=x_valid, y=y_valid, batch_size=2)

predictions_validation = model.predict(x_valid)
predictions_test = model.predict(x_test)
test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])
angles = pd.DataFrame(predictions_test[:,0], columns = ['angle'])
speeds = pd.DataFrame(predictions_test[:,1], columns = ['speed'])
output = pd.concat([test_image_id, angles, speeds], axis = 1, sort = False)
output.sort_values(by=['image_id'], inplace = True)
output.to_csv('sunny-early-stopping.csv', index = False)

#plt.figure()

# predictions = 5 * np.round(predictions/5)

# replace repdictions with validation data for graphs
# mse = (predictions-y_valid)**2

#plt.plot(range(len(y_valid)),y_valid,'*')
#plt.plot(range(len(y_valid)),predictions,'r*')
#plt.legend(('Labels','Predictions'),loc = 'best')
#plt.xlabel('test example number')
#plt.ylabel('angle of rotation')
