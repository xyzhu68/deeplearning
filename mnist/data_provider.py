from keras.utils import to_categorical
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import matplotlib.pyplot as plt # DEBUG

def flip_images(X, y, doFlip):
    if not doFlip:
        y = to_categorical(y, 10)
        y_E = np.zeros(len(y))
        return (X, y, y_E)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    datagen.fit(X)
    data_it = datagen.flow(X, y, batch_size=1)
    
    data_list = []
    y_list = []
    batch_index = 0
    while batch_index <= data_it.batch_index:
        data = data_it.next()
        data_list.append(data[0])
        y_list.append(data[1])
        batch_index = batch_index + 1

    data_array = np.asarray(data_list)
    data_array = data_array.reshape(-1, 28, 28, 1)
    y_array = np.asarray(y_list)

    y_array_E = np.full(len(data_array), 1.0)
    y_array = to_categorical(y_array, 10)
    return (data_array, y_array, y_array_E)

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
        for yValue in y:
            if yValue < 5:
                y_array_E.append(0)
            else:
                y_array_E.append(1)
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
        
    