from scipy.io import arff
import numpy as np
# from keras.models import load_model, Model, Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Input, Dense, Activation, Dropout, Flatten
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers, backend as K
#from keras import backend as K
import sys
import os
import matplotlib.pyplot as plt
# from keras.activations import relu, softmax, sigmoid
import datetime
import random
from autokeras.image.image_supervised import ImageClassifier

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

hours = 12
if nbArgs > 2:
    hours = int(sys.argv[2])

filepath_train = "train.arff"

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

print("autokeras time limit: {0}".format(hours))

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_C0 = make_conv_model(64, False)

changePoint = 50
# appear?


model_C0 = make_conv_model(64, False)

# test data for fit_final of Ci
testX = []
testY = []

# get data
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
#for i in range(1): # !!!!!!!!!!!!
    print(i)
    X_org, y_org = next(gen)
    changeData = bool(random.getrandbits(1)) # used for create data for Ei

    if drift_type == "flip":
        X, y, _ = flip_images(X_org, y_org, False)
        x_t, y_t, _ = flip_images(X_org, y_org, True)
    elif drift_type == "rotate":
        X, y, _ = rot(X_org, y_org, 0)
        x_t, y_t, _ = rot(X_org, y_org, random.randint(1, 181))
    elif drift_type == "appear":
        X, y, _ = appear(X_org, y_org, True)
        x_t, y_t, _ = appear(X_org, y_org, False)
    elif drift_type == "remap":
        X, y, _ = remap(X_org, y_org, True)
        x_t, y_t, _ = remap(X_org, y_org, False)
    elif drift_type == "transfer":
        X, y, _ = transfer(X_org, y_org, True)
        x_t, y_t, _ = transfer(X_org, y_org, False)
    else:
        print("Unknown drift type")
        exit()

    #model_C0.fit(X, y, batch_size=50, epochs = 10)

    testX.extend(x_t)
    ys = [np.argmax(y) for y in y_t]
    testY.extend(ys)


#C0Weights = "C0_weigths_{0}.h5".format(drift_type)
#model_C0.save_weights(C0Weights)

testX = np.asarray(testX)
testX = testX.reshape(-1, 28, 28, 1)
testY = np.asarray(testY)
testY = testY.reshape(-1,)
print("testX: ", testX.shape)
print("testX value: ", testX[0])
print("testY: ", testY.shape)
print("testY value: ", testY[: 5])


# adaption: data changed
trainX = []
trainY = []
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

    # data for Ci
    if i >= changePoint:
        trainX.extend(X)
        ys = [np.argmax(yItem) for yItem in y]
        trainY.extend(ys)

trainX = np.asarray(trainX)
trainX = trainX.reshape(-1, 28, 28, 1)
trainY = np.asarray(trainY)
trainY = trainY.reshape(-1,)
print("X: ", trainX.shape)
print("y shape: ", np.asarray(trainY).shape)
print("y: ", trainY[:20])

clf = ImageClassifier(verbose=True)
clf.fit(trainX, trainY, time_limit=hours * 60 * 60) 
#clf.final_fit(trainX, trainY, testX, testY, retrain=True)
result = clf.evaluate(testX, testY)
print(result)

cls_type = "Ci"
fileOfBestModel = "autokeras_mnist_{0}_{1}_{2}.h5".format(cls_type, drift_type, hours)
#clf.load_searcher().load_best_model().produce_keras_model().save(fileOfBestModel)
clf.export_keras_model(fileOfBestModel)
   
endTime = datetime.datetime.now()
print(endTime - beginTime)




