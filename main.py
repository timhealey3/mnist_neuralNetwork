import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.np_utils import to_categorical
import random

# MNIST number data set

np.random.seed(0)

# load in data
(X_train, y_train), (X_test, y_text) = mnist.load_data()
# asserts to make sure data is loaded correctly
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to number of labels"
assert(X_train.shape[1:], (28, 28)), "The dimensions of the image pixels are not 28*28"
assert(X_test.shape[1:] == (28, 28)), "The dimensions of the image pixels are not 28*28"
# visualize data first
num_of_samples = []