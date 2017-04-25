#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:39:02 2017

@author: benharris
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from skimage import io
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

#Step 1: Read list of images into a DataFrame
columns = ['image','x1','y1','x2','y2','class']
images = pd.read_csv('/Users/benharris/Documents/Projects/SeaLions/Results/SeaLions-R-CNN-Cleaned.csv',names=columns) 

#Step 2: Remove Negatives
images.ix[images.x1 < 0, 'x1'] = 0
images.ix[images.x2 < 0, 'x2'] = 0
images.ix[images.y1 < 0, 'y1'] = 0
images.ix[images.y2 < 0, 'y2'] = 0

#Step 3: Get Unique Images so we don't need to load the images for each iteration.
d_images = images.image.unique()

#Step 4: Get Train Number
c_train = int(len(images)*.7)-1

#Step 5: Set Arrays
X_train = []
X_test = []
y_train = []
y_test = []

#Step 6: split sea lions into training ans test sets
for image in d_images:
    #Create a list of unique Sea Lions in image
    i_images = images[images['image']==image]
    #Load the big image into memory
    img = io.imread(image)
    #Loop through sea lions
    for index,row in i_images.iterrows():
        #Extract each column for clear understanding. Unnecessary.
        x1 = row['x1']
        x2 = row['x2']
        y1 = row['y1']
        y2 = row['y2']
        cl = row['class']
        #Crop Image
        crop = img[x1:x2,y1:y2].reshape(3,x2-x1,y2-y1)
        #Split Train and Test Sets
        if (index <= c_train):
            X_train.append(crop)
            y_train.append(cl)
        else:
            X_test.append(crop)
            y_test.append(cl)
            
#Step 7: Get number of classes


#Step 7: Normalize Data
#X_train = pd.DataFrame(X_train)
#X_test = pd.DataFrame(X_test)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Step 8: Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#Step 9: Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())



#Step 10: Fit the training model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
#Step 11: Test 
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
