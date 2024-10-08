import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import random
import requests
from PIL import Image
import cv2

# seed random number generator
np.random.seed(0)
# training and test data - 28pixel x 28pixel images
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# make sure data is properly imported
assert(X_train.shape[0] == y_train.shape[0]), "Number of train images is != to number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "Number of test images is != to number of labels"
assert(X_train.shape[1:] == (28, 28)), "The training images are != 28x28"
assert(X_test.shape[1:] == (28, 28)), "The testing images are != 28x28"

# one hot encode lables to categorical (digits are 0-9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# normalize, the max pixel value (255 is down to 1)
# all values are normalized between 0-1
X_train = X_train / 255
X_test = X_test / 255
# need to reshape to one dimensional, flatten into single row of 784 pixels
# 784 is 28x28
num_pixels = 784
num_classes = 10
cols = 5
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

def createModel():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def runModel():
    model = createModel()
    history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)
    score = model.evaluate(X_test, y_test, verbose = 0)
