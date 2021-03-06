from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt # DEBUG

def flip_images(X, y, doFlip):
    if not doFlip:
        return (X, y)

    X = X.reshape(-1, 128, 128)
    x_array = []
    for image in X:
        flipped = np.flip(image, 0)
        flipped = np.flip(flipped, 1)
        x_array.append(flipped)
    x_array = np.asarray(x_array)
    X = x_array.reshape(-1, 128, 128, 1)

    return (X, y)

def appear(X, y, isBase):
    if isBase:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if yValue > 9:
                x_array.append(X[i])
                y_array.append(yValue)

        y_array = to_categorical(y_array, 36)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 128, 128, 1)
        return (x_array, y_array)
    else:
        return (X, y)

def remap(X, y, firstHalf):
     if firstHalf:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if yValue < 10:
                x_array.append(X[i])
                y_array.append(yValue)

        y_array = to_categorical(y_array, 36)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 128, 128, 1)
        return (x_array, y_array)
     else:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if 9 < yValue < 20 :
                x_array.append(X[i])
                y_array.append(yValue - 10)
        
        y_array = to_categorical(y_array, 36)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 128, 128, 1)
        
        return (x_array, y_array)

def rot(X,y, angle):
    if angle == 0:
        return (X, y)
    else:
        X_result = []
        for image in X:
            X_rot = rotate(image, angle, reshape=False)
            X_result.append(X_rot)
        X_result = np.asarray(X_result)
        X_result = X_result.reshape(-1, 128, 128, 1)
        return (X_result, y)

def transfer(X, y, firstHalf):
    x_array = []
    y_array = []
    size = len(X)
    for i in range(size):
        yValue = np.argmax(y[i])
        if firstHalf:
            if yValue < 10:
                x_array.append(X[i])
                y_array.append(yValue)
        else:
            if yValue >= 10:
                x_array.append(X[i])
                y_array.append(yValue)
        
    x_array = np.asarray(x_array)
    x_array = x_array.reshape(-1, 128, 128, 1)
    y_array = to_categorical(y_array, 36)
    return (x_array, y_array)

    