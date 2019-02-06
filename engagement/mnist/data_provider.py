from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt # DEBUG

def flip_images(X, y, doFlip):
    if not doFlip:
        y = to_categorical(y, 10)
        return (X, y)

    X = X.reshape(-1, 28, 28)
    x_array = []
    for image in X:
        axis = bool(random.getrandbits(1))
        flipped = np.flip(image, axis)
        x_array.append(flipped)
    x_array = np.asarray(x_array)
    X = x_array.reshape(-1, 28, 28, 1)
    y = to_categorical(y, 10)
    
    return (X, y)

def appear(X, y, isBase):
    if isBase:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = y[i]
            if yValue < 5:
                x_array.append(X[i])
                y_array.append(yValue)
            
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array)
    else:
        y_array = to_categorical(y, 10)
        return (X, y_array)

def remap(X, y, firstHalf):
     if firstHalf:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = y[i]
            if yValue < 5:
                x_array.append(X[i])
                y_array.append(yValue)
            
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array)
     else:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = y[i]
            if yValue > 4:
                x_array.append(X[i])
                y_array.append(yValue - 5)
        
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array)

def rot(X,y, angle):
    if angle == 0:
        y = to_categorical(y, 10)
        return (X, y)
    else:
        X_result = []
        for image in X:
            X_rot = rotate(image, angle, reshape=False)
            X_result.append(X_rot)
        X_result = np.asarray(X_result)
        X_result = X_result.reshape(-1, 28, 28, 1)
        y = to_categorical(y, 10)
        return (X_result, y)

def transfer(X, y, firstHalf):
    x_array = []
    y_array = []
    for i in range(len(X)):
        yValue = y[i]
        if firstHalf:
            if yValue < 5:
                x_array.append(X[i])
                y_array.append(yValue)
        else:
            if yValue >= 5:
                x_array.append(X[i])
                y_array.append(yValue)
        

    y_array = to_categorical(y_array, 10)
    x_array = np.asarray(x_array)
    x_array = x_array.reshape(-1, 28, 28, 1)
    return (x_array, y_array)

    