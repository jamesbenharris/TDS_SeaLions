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
import matplotlib.pyplot as plt
import itertools
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
from sklearn.metrics import confusion_matrix

K.set_image_dim_ordering('th')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(4) 
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

seed = 7
np.random.seed(seed)

#Step 1: Load Sea Lions
h5f = h5py.File('/Users/benharris/Documents/Projects/SeaLions.h5','r')
X_train = h5f['X_train'][:]*255.0
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]*255.0
y_test = h5f['y_test'][:]
h5f.close()

reduct = 1.414
num_classes = y_test.shape[1]
#Step 2: Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 100, 100), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

#Step 3: Compile model
epochs = 3
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#Step 4: Fit the training model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
#Step 5: Test 


scores = model.evaluate(X_test, y_test, verbose=0)

# serialize weights to HDF5
model.save_weights("sl_model.h5")
print("Saved model to disk")

print("Accuracy: %.2f%%" % (scores[1]*100))


yp = model.predict_classes(X_test)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.argmax(1), yp)
np.set_printoptions(precision=2)

class_names = [1,2,3,4,5]

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()