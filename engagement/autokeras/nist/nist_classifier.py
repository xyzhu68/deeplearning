from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, backend as K
import sys
import os
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime
import random
from autokeras.image.image_supervised import ImageClassifier

from data_provider import *
from model_provider import *
# from beta_distribution_drift_detector.bdddc import BDDDC

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

searchEi = False
if nbArgs > 2:
    if sys.argv[2] == "Ei":
        searchEi = True

img_size = 128
nbFilters = 16

beginTime = datetime.datetime.now()

# settings
gen_batch_size = 720
epochs = 10

totalDataSize = 72000
sizeOneBatch = 720
nbBatches = totalDataSize // sizeOneBatch # devide dataset into batches
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches

changePoint = 50
if drift_type == "appear":
    nbBaseBatches = 30
    changePoint = 30

# prepare data
train_data_dir = os.path.abspath("../../../data/NIST")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

model_C0 = make_conv_model(nbFilters, False)

# test data for fit_final of Ci
testX = []
testY = []
# test data for fit_final of Ei
testEX = []
testEY = []
# training in base phase
for i in range(nbBaseBatches):
#for i in range(1): # !!!!!!!!!!!!
    print(i)
    X, y = train_generator.next()
    changeData = bool(random.getrandbits(1)) # used for create data for Ei

    if drift_type == "flip":
        X, y, _ = flip_images(X, y, False)
        x_t, y_t, _ = flip_images(X, y, True)
    elif drift_type == "rotate":
        X, y, _ = rot(X, y, 0)
        x_t, y_t, _ = rot(X, y, random.randint(1, 181))
    elif drift_type == "appear":
        X, y, _ = appear(X, y, True)
        x_t, y_t, _ = appear(X, y, False)
    elif drift_type == "remap":
        X, y, _ = remap(X, y, True)
        x_t, y_t, _ = remap(X, y, False)
    elif drift_type == "transfer":
        X, y, _ = transfer(X, y, True)
        x_t, y_t, _ = transfer(X, y, False)
    else:
        print("Unknown drift type")
        exit()

    model_C0.fit(X, y, batch_size=20, epochs = epochs)

    testX.extend(x_t)
    ys = [np.argmax(y) for y in y_t]
    testY.append(ys)

    if changeData:
        testEX.extend(x_t)
        testEY.append(1)
    else:
        testEX.extend(X)
        testEY.append(0)

#C0Weights = "C0_weigths_{0}.h5".format(drift_type)
#model_C0.save_weights(C0Weights)

testX = np.asarray(testX)
testX = testX.reshape(-1, 128, 128, 1)
testY = np.asarray(testY)
testY = testY.reshape(-1,)
print("testX: ", testX.shape)
print("testY: ", testY.shape)

testEX = np.asarray(testEX)
testEX = testEX.reshape(-1, 128, 128, 1)
testEY = np.asarray(testEY)
testEY = testEY.reshape(-1,)
print("testEX: ", testEX.shape)
print("testEY: ", testEY.shape)


# adaption: data changed
trainX = []
trainY = []
trainEX = []
trainEY = []
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch number: {0}".format(i))
    
    X_org, y_org = train_generator.next()
    
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
        predictC0 = model_C0.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)
        if np.argmax(predictC0) == np.argmax(y[index]):
            trainEX.append(X[index])
            trainEY.append(0)
        else:
            trainEX.append(X[index])
            trainEY.append(1)

    # data for Ci
    if i >= changePoint:
        trainX.extend(X)
        ys = [np.argmax(y) for y in y_org]
        trainY.append(ys)

trainX = np.asarray(trainX)
trainX = trainX.reshape(-1, 128, 128, 1)
trainY = np.asarray(trainY)
trainY = trainY.reshape(-1,)
print("X: ", trainX.shape)
print("y shape: ", np.asarray(trainY).shape)
print("y: ", trainY[:5])

trainEX = np.asarray(trainEX)
trainEY = np.asarray(trainEY)
print("EX: ", trainEX.shape)
print("EY: ", trainEY.shape)

clf = ImageClassifier(verbose=True)
if searchEi:
    clf.fit(trainEX, trainEY, time_limit=12 * 60 * 60)
    clf.fit_final(trainEX, trainEY, testEX, testEY, retrain=True)
    result = clf.evaluate(testEX, testEY)
    print(result)
else:
    clf.fit(trainX, trainY, time_limit=12 * 60 * 60)
    clf.final_fit(trainX, trainY, testX, testY, retrain=True)
    result = clf.evaluate(testX, testY)
    print(result)

cls_type = "Ci"
if searchEi:
    cls_type = "Ei"
fileOfBestModel = "autokeras_nist_{0}_{1}.h5".format(cls_type, drift_type)
clf.load_searcher().load_best_model().produce_keras_model().save(fileOfBestModel)
   
endTime = datetime.datetime.now()
print(endTime - beginTime)




