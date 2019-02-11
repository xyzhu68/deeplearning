from scipy.io import arff
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import datetime
import random

from data_provider import *
from model_provider import *
from beta_distribution_drift_detector.bdddc import BDDDC

# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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

def data_generator(streamSize, fileToLoad): 
    X, y = load_data(fileToLoad)
    count = 0
    while True:
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        count += streamSize
        print("count: %d" % count)
        yield X_result, y_result

def is_drift(C0_model, detector, X, y):
    pred = C0_model.predict(X)
    pred_cls = np.argmax(pred, axis=1)
    y_cls = np.argmax(y, axis=1)
    
    detector.add_element(pred_cls, y_cls, classifier_changed=False)

    return detector.detected_change()

###################################################################



beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_C0 = make_conv_model(64, False)

# get data
gen = data_generator(sizeOneBatch, filepath_train)
# training in base phase
for i in range(nbBaseBatches):
#for i in range(1): # !!!!!!!!!!!!
    print(i)
    X_org, y_org = next(gen)
    # changeData = bool(random.getrandbits(1)) # used for create data for Ei

    if drift_type == "flip":
        X, y = flip_images(X_org, y_org, False)
    elif drift_type == "remap":
        X, y = remap(X_org, y_org, True)
    else:
        print("Unknown drift type")
        exit()

    model_C0.fit(X, y, batch_size=50, epochs = 10)

# drift detection
bdddc = BDDDC()

# adaption: data changed
trainEX = []
trainEY = []
trainCX = []
trainCY = []
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch number: {0}".format(i))
    
    X_org, y_org = next(gen)
    
    X = None
    y = None
    if drift_type == "flip":
        X, y = flip_images(X_org, y_org, i >= nbBatches/2)
    elif drift_type == "remap":
        X, y = remap(X_org, y_org, i < nbBatches/2)

    if is_drift(model_C0, bdddc, X, y):
        print("drift detected")
        # data for Ei
        for index in range(len(X)):
            predictC0 = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
            if np.argmax(predictC0) == np.argmax(y[index]):
                trainEX.append(X[index])
                trainEY.append(0)
            else:
                trainEX.append(X[index])
                trainEY.append(1)

        # data fo Ci
        trainCX.extend(X)
        ys = [np.argmax(yItem) for yItem in y]
        trainCY.extend(ys)

trainEX = np.asarray(trainEX)
trainEY = np.asarray(trainEY)
print("trainEX: ", trainEX.shape)
print("trainEY: ", trainEY.shape)
trainCX = np.asarray(trainCX)
trainCX = trainCX.reshape(-1, 28, 28, 1)
trainCY = np.asarray(trainCY)
trainCY = trainCY.reshape(-1,)
print("trainCX: ", trainCX.shape)
print("trainCY: ", trainCY.shape)

# test data
testEX = []
testEY = []
testCX = []
testCY = []
testGen = data_generator(100, filepath_test)
for i in range(100):
    X_org, y_org = next(testGen)

    X = None
    y = None
    change = bool(random.getrandbits(1)) # changed and non-changed data
    if drift_type == "flip":
        X, y = flip_images(X_org, y_org, change)
    elif drift_type == "remap":
        firstHalf = change
        X, y = remap(X_org, y_org, firstHalf)

    # test data for Ei
    for index in range(len(X)):
        predictC0 = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if np.argmax(predictC0) == np.argmax(y[index]):
            testEX.append(X[index])
            testEY.append(0)
        else:
            testEX.append(X[index])
            testEY.append(1)

    # test data for Ci
    testCX.extend(X)
    ys = [np.argmax(yItem) for yItem in y]
    testCY.extend(ys)

testEX = np.asarray(testEX)
testEY = np.asarray(testEY)
print("testEX: ", testEX.shape)
print("testEY: ", testEY.shape)
testCX = np.asarray(testCX)
testCX = testCX.reshape(-1, 28, 28, 1)
testCY = np.asarray(testCY)
testCY = testCY.reshape(-1,)
print("testCX: ", testCX.shape)
print("testCY: ", testCY.shape)

ei_data_file = "ei_data_{0}".format(drift_type)
np.savez(ei_data_file, trainEX = trainEX, trainEY = trainEY,
                        testEX = testEX, testEY = testEY)

ci_data_file = "ci_data_{0}".format(drift_type)
np.savez(ci_data_file, trainCX = trainCX, trainCY = trainCY,
                        testCX = testCX, testCY = testCY)
   
endTime = datetime.datetime.now()
print(endTime - beginTime)




