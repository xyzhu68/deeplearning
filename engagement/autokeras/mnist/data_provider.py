from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import random
#import matplotlib.pyplot as plt # DEBUG

def flip_images(X, y, doFlip):
    if not doFlip:
        y = to_categorical(y, 10)
        y_E = np.zeros(len(y))
        return (X, y, y_E)

    X = X.reshape(-1, 28, 28)
    x_array = []
    for image in X:
        axis = bool(random.getrandbits(1))
        flipped = np.flip(image, axis)
        x_array.append(flipped)
    x_array = np.asarray(x_array)
    X = x_array.reshape(-1, 28, 28, 1)
    y_E = np.full(len(y), 1.0)
    y = to_categorical(y, 10)
    
    return (X, y, y_E)

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
            

        y_array_E = np.zeros(len(y_array))
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array, y_array_E)
    else:
        y_array_E = []
        # for yValue in y:
        #     if yValue < 5:
        #         y_array_E.append(0)
        #     else:
        #         y_array_E.append(1)
        y_array = to_categorical(y, 10)
        return (X, y_array, y_array_E)

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
            

        y_array_E = np.zeros(len(y_array))
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array, y_array_E)
     else:
        size = len(X)
        x_array = []
        y_array = []
        for i in range(size):
            yValue = y[i]
            if yValue > 4:
                x_array.append(X[i])
                y_array.append(yValue - 5)
        
        y_array_E = np.full(len(y_array), 1.0)
        y_array = to_categorical(y_array, 10)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, 28, 28, 1)
        return (x_array, y_array, y_array_E)

def rot(X,y, angle):
    if angle == 0:
        y_E = np.zeros(len(X))
        y = to_categorical(y, 10)
        return (X, y, y_E)
    else:
        X_result = []
        for image in X:
            X_rot = rotate(image, angle, reshape=False)
            X_result.append(X_rot)
        X_result = np.asarray(X_result)
        X_result = X_result.reshape(-1, 28, 28, 1)
        y_E = np.full(len(y), 1.0)
        y = to_categorical(y, 10)
        return (X_result, y, y_E)

def transfer(X, y, firstHalf):
    x_array = []
    y_array = []
    y_array_E = []
    for i in range(len(X)):
        yValue = y[i]
        if firstHalf:
            if yValue < 5:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(0)
        else:
            if yValue >= 5:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(1)
        

    #y_array_E = np.zeros(len(y_array)) if firstHalf else np.full(len(y_array), 1)
    y_array = to_categorical(y_array, 10)
    x_array = np.asarray(x_array)
    x_array = x_array.reshape(-1, 28, 28, 1)
    return (x_array, y_array, y_array_E)

# def combine_Ei_training_data(drift_type, X_org, y_org, X, y_E):
#     if drift_type == "flip" or drift_type == "rotate":
#         y_E_org = np.zeros(len(y_E))
#         X_combine = np.concatenate((X_org, X))
#         y_combine = np.concatenate((y_E_org, y_E))
#         return shuffle(X_combine, y_combine)
#     elif drift_type == "remap" or drift_type == "transfer":
#         size = len(X)
#         x_array = []
#         for i in range(size):
#             yValue = y_org[i]
#             if yValue < 5:
#                 x_array.append(X_org[i])
#         y_E_org = np.zeros(len(x_array))
#         x_array = np.asarray(x_array)
#         x_array = x_array.reshape(-1, 28, 28, 1)
#         X_combine = np.concatenate((x_array, X))
#         y_combine = np.concatenate((y_E_org, y_E))
#         return shuffle(X_combine, y_combine)
#     else:
#         return (X, y_E)
    