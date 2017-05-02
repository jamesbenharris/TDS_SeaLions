#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:27:54 2017

@author: benharris
"""
import numpy as np
import pandas as pd
import h5py
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling2D,FractionalMaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from skimage import io,transform
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

#Step 1: Load Sea Lions
h5f = h5py.File('/Users/benharris/Documents/Projects/sl/SeaLions.h5','r')
X_train = h5f['X_train'][:]*255.0
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]*255.0
y_test = h5f['y_test'][:]
h5f.close()

reduct = 1.414
num_classes = y_test.shape[1]

#Step 2: Create the model
model = Sequential()
model.add(Conv2D(10, (10, 10), input_shape=(3, 100, 100), padding='same', activation='relu'))
model.add(FractionalMaxPooling2D(pool_size=(reduct, reduct)))
model.add(Conv2D(20, (6, 6), activation='relu', padding='same'))
model.add(FractionalMaxPooling2D(pool_size=(reduct, reduct)))
model.add(Dropout(0.25))
model.add(Conv2D(30, (3, 3), activation='relu', padding='same'))
model.add(FractionalMaxPooling2D(pool_size=(reduct, reduct)))
model.add(Dropout(0.5))
model.add(Conv2D(40, (1, 1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#Step 3: Compile model
epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#Step 4: Fit the training model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
#Step 11: Test 
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))