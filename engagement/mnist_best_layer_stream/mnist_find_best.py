from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime

from data_provider import *
from model_provider import *

# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
#filepath_test = "test.arff"

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]



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
        
# def calc_accuracy(modelC0, modelEi, modelCi, X, y):
#     predictEi = modelEi.predict(X)
#     index = 0
#     correct = 0
#     predict = None
#     for p in predictEi:
#         if p[0] > 0.5:
#             predict = modelCi.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
#         else:
#             predict = modelC0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
#         if (np.argmax(predict) == np.argmax(y[index])):
#             correct += 1
#         index += 1
#     return correct / len(X)


# def build_model(model_type, weights):
#     model = None
#     if layerToEngage == 0:
#         model = Sequential()
#         model.add(Dense(512, input_shape=(784,)))
#         model.add(Activation('relu'))
#     else:
#         model_conv = make_conv_model(64, True)
#         if weights:
#             model_conv.load_weights(weights, by_name=True)
#             # do freezing
#             for i in range(len(model_conv.layers)):
#                 model_conv.layers[i].trainable = False
#         else:
#             # do engagement
#             layersToPop = 7 - layerToEngage
#             if layersToPop < 0:
#                 print("layer to engage cannot be greater than 7")
#                 exit()
#             for i in range(layersToPop):
#                 model_conv.pop()
#         print(model_conv.summary())
#         model = Sequential()
#         model.add(model_conv)
#         model.add(Flatten(name="Flatten"))
#         model.add(Dense(512))
#         model.add(Activation('relu'))
    
#     model.add(Dropout(0.5))
#     if model_type == "E":
#         model.add(Dense(1, name="Ei_dense1"))
#         model.add(Activation('sigmoid', name="Ei_act"))
#         model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
#     elif model_type == "P":
#         model.add(Dense(10))
#         model.add(Activation('softmax'))
#         model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
#     else:
#         print("invalid model type: {0}".format(model_type))

#     print(model.summary())

#     return model


#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches
bundleSize = 1
epochs = 10

model_C0 = make_conv_model(64, False)


# get data
gen = data_generator(sizeOneBatch * bundleSize)
# training in base phase
for i in range(nbBaseBatches//bundleSize):
#for i in range(1):
    print(i)
    X, y = next(gen)

    if drift_type == "flip":
        X, y, _ = flip_images(X, y, False)
    elif drift_type == "rotate":
        X, y, _ = rot(X, y, 0)
    elif drift_type == "appear":
        X, y, _ = appear(X, y, True)
    elif drift_type == "remap":
        X, y, _ = remap(X, y, True)
    elif drift_type == "transfer":
        X, y, _ = transfer(X, y, True)
    else:
        print("Unknown drift type")
        exit()

    model_C0.fit(X, y, batch_size=50, epochs = epochs)

C0Weights = "C0_weigths_{0}.h5".format(drift_type)
model_C0.save(C0Weights)
del model_C0


# adaption: data changed
angle = 0 # for rotate
totalNumberOfLayers = 7
initPopulation = 3
nextPopulation = 1
nbOfFitnessToCompare = 1
engageResults = []
PAResults = []
for b in range(nbBaseBatches//bundleSize, nbBatches//bundleSize):
    print("bundle number: {0}".format(b))
    i = b * bundleSize

    loopbegin = datetime.datetime.now()
    
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

    results = Evolutionary(totalNumberOfLayers,
                            nbOfFitnessToCompare,
                            initPopulation,
                            nextPopulation,
                            28,
                            b,
                            X,
                            y,
                            epochs,
                            drift_type,
                            "engage")
    engageResults.append(results)
    results = Evolutionary(10, 1, 3, 1, 28, b, X, y, epochs, drift_type, "PA", results[0])
    PAResults.append(results)
   
    loopend = datetime.datetime.now()
    print("loop time: ", loopend - loopbegin)
    # break # !!!!!!!! TEST
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)
fileName = "mnist_best_layer_{0}.npz".format(drift_type)
np.savez(fileName, engageResults = engageResults, PAResults = PAResults, duration=str(endTime - beginTime))


