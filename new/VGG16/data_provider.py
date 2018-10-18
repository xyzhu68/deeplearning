from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt # DEBUG

nbClasses = 20
img_size = 150

# def flip_images(X, y, doFlip):
#     if not doFlip:
#         #y = to_categorical(y, 36)
#         y_E = np.zeros(len(y))
#         return (X, y, y_E)

#     X = X.reshape(-1, 128, 128)
#     x_array = []
#     for image in X:
#         axis = bool(random.getrandbits(1))
#         flipped = np.flip(image, axis)
#         x_array.append(flipped)
#     x_array = np.asarray(x_array)
#     X = x_array.reshape(-1, 128, 128, 1)
#     y_E = np.full(len(y), 1.0)
#     #y = to_categorical(y, 36)
    
#     return (X, y, y_E)

def appear(X, y, isBase):
    if isBase:
        size = len(X)
        x_array = []
        y_array = []
        y_array_E = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if yValue < nbClasses // 2:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(0)

        y_array = to_categorical(y_array, nbClasses)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, img_size, img_size, 3)
        return (x_array, y_array, y_array_E)
    else:
        y_array_E = []
        for yItem in y:
            yValue = np.argmax(yItem)
            if yValue >= nbClasses // 2
                y_array_E.append(0)
            else:
                y_array_E.append(1)
        y_array = to_categorical(y, nbClasses)
        return (X, y, y_array_E)

def remap(X, y, firstHalf):
     if firstHalf:
        size = len(X)
        x_array = []
        y_array = []
        y_array_E = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if yValue < nbClasses // 2:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(0)

        y_array = to_categorical(y_array, nbClasses)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, img_size, img_size, 3)
        return (x_array, y_array, y_array_E)
     else:
        size = len(X)
        x_array = []
        y_array = []
        y_array_E = []
        for i in range(size):
            yValue = np.argmax(y[i])
            if (nbClasses//2-1) < yValue < nbClasses : # 9 < yValue < 20
                x_array.append(X[i])
                y_array.append(yValue - 10)
                y_array_E.append(1)
        
        #y_array_E = np.full(len(y_array), 1.0)
        y_array = to_categorical(y_array, nbClasses)
        x_array = np.asarray(x_array)
        x_array = x_array.reshape(-1, img_size, img_size, 3)
        y_array = to_categorical(y_array, nbClasses)
        return (x_array, y_array, y_array_E)

# def rot(X,y, angle):
#     if angle == 0:
#         y_E = np.zeros(len(X))
#         #y = to_categorical(y, 10)
#         return (X, y, y_E)
#     else:
#         X_result = []
#         for image in X:
#             X_rot = rotate(image, angle, reshape=False)
#             X_result.append(X_rot)
#         X_result = np.asarray(X_result)
#         X_result = X_result.reshape(-1, 128, 128, 1)
#         y_E = np.full(len(y), 1.0)
#         #y = to_categorical(y, 10)
#         return (X_result, y, y_E)

def transfer(X, y, firstHalf):
    x_array = []
    y_array = []
    y_array_E = []
    for i in range(len(X)):
        yValue = np.argmax(y[i])
        if firstHalf:
            if yValue < nbClasses // 2:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(0)
        else:
            if yValue >= nbClasses // 2:
                x_array.append(X[i])
                y_array.append(yValue)
                y_array_E.append(1)
        
    x_array = np.asarray(x_array)
    x_array = x_array.reshape(-1, img_size, img_size, 3)
    y_array = to_categorical(y_array, nbClasses)
    return (x_array, y_array, y_array_E)

    