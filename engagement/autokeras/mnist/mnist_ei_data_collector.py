from scipy.io import arff
import numpy as np
# from keras.models import load_model, Model, Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Input, Dense, Activation, Dropout, Flatten
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers, backend as K
# from keras import backend as K
import sys
import os
import matplotlib.pyplot as plt
# from keras.activations import relu, softmax, sigmoid
import datetime
import random

from data_provider import *
from model_provider import *
# from beta_distribution_drift_detector.bdddc import BDDDC

# # settings for GPU
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

filepath_train = "train.arff"
filepath_test = "test.arff"

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_data(dataPath):
    data, _ = arff.loadarff(dataPath)
    dataArray = np.asarray(data.tolist(), dtype=np.float)
    X = np.delete(dataArray, 784, 1) / 255
    X = X.reshape(-1, 28, 28, 1)
    y = dataArray[:,784]
    return (X, y)

def data_generator(streamSize): 
    X, y = load_data(filepath_train)
    count = 0
    while True:
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        count += streamSize
        print("count: %d" % count)
        yield X_result, y_result

###################################################################



beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_C0 = make_conv_model(64, False)

changePoint = 50
# appear?


model_C0 = make_conv_model(64, False)

# get data
gen = data_generator(sizeOneBatch)
# training in base phase
#for i in range(nbBaseBatches):
for i in range(1): # !!!!!!!!!!!!
    print(i)
    X_org, y_org = next(gen)
    changeData = bool(random.getrandbits(1)) # used for create data for Ei

    if drift_type == "flip":
        X, y, _ = flip_images(X_org, y_org, False)
    elif drift_type == "rotate":
        X, y, _ = rot(X_org, y_org, 0)
    elif drift_type == "appear":
        X, y, _ = appear(X_org, y_org, True)
    elif drift_type == "remap":
        X, y, _ = remap(X_org, y_org, True)
    elif drift_type == "transfer":
        X, y, _ = transfer(X_org, y_org, True)
    else:
        print("Unknown drift type")
        exit()

    model_C0.fit(X, y, batch_size=50, epochs = 10)

#C0Weights = "C0_weigths_{0}.h5".format(drift_type)
#model_C0.save_weights(C0Weights)


# adaption: data changed
trainEX = []
trainEY = []
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch number: {0}".format(i))
    
    X_org, y_org = next(gen)
    
    X = None
    y = None
    if drift_type == "flip":
        X, y, _ = flip_images(X_org, y_org, i >= nbBatches/2)
    elif drift_type == "appear":
        X, y, _ = appear(X_org, y_org, False)
    elif drift_type == "remap":
        X, y, _ = remap(X_org, y_org, i < nbBatches/2)
    elif (drift_type == "rotate"):
        if i > 50:
            angle += 5
            if angle > 180:
                angle = 180
        else:
            angle = 0
            data_changed = False
        X, y, _ = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        X, y, _ = transfer(X_org, y_org, i < nbBatches/2)

    # data for Ei
    for index in range(len(X)):
        predictC0 = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if np.argmax(predictC0) == np.argmax(y[index]):
            trainEX.append(X[index])
            trainEY.append(0)
        else:
            trainEX.append(X[index])
            trainEY.append(1)


trainEX = np.asarray(trainEX)
trainEY = np.asarray(trainEY)
print("trainEX: ", trainEX.shape)
print("trainEY: ", trainEY.shape)

# test data
testEX = []
testEY = []
test_X, test_y = load_data(filepath_test)
testGen = data_generator(100)
for i in range(100):
    X_org, y_org = next(testGen)

    X = None
    y = None
    label = 0
    change = bool(random.getrandbits(1))
    if drift_type == "flip":
        X, y, _ = flip_images(X_org, y_org, change)
        label = 1 if change else 0
    elif drift_type == "appear":
        isBase = change
        X, y, _ = appear(X_org, y_org, isBase)
        label = 0 if isBase else 1
    elif drift_type == "remap":
        firstHalf = change
        X, y, _ = remap(X_org, y_org, firstHalf)
        label = 0 if firstHalf else 1
    elif (drift_type == "rotate"):
        if not change:
            angle = 0
            label = 0
        else:
            angle = random.randint(1, 181)
            label = 1
        X, y, _ = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        firstHalf = change
        X, y, _ = transfer(X_org, y_org, firstHalf)
        label = 0 if firstHalf else 1

    testEX.extend(X)
    testEY.extend(np.full(len(X), label, int))
    

testEX = np.asarray(testEX)
testEY = np.asarray(testEY)
print("testEX: ", testEX.shape)
print("testEY: ", testEY.shape)

ei_data_file = "ei_data_{0}".format(drift_type)
np.savez(ei_data_file, trainEX = trainEX, trainEY = trainEY,
                        testEX = testEX, testEY = testEY)
   
endTime = datetime.datetime.now()
print(endTime - beginTime)




