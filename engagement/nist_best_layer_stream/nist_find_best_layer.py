
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

from data_provider import *
from model_provider import *
import random # TEST

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

# blockToEngage = 4
# if nbArgs > 2:
#     blockToEngage =  int(sys.argv[2])
#     valid = 0 < blockToEngage < 5 # 1, 2, 3, 4
#     if not valid:
#         print("block to engage can only be 1, 2, 3, or 4")
#         exit()



# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
def calc_accuracy(img_size, modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)


def build_model(model_type, weights, nbFilters, layerToEngage):
    model = None

    model_conv = make_conv_model(nbFilters, True)
    if weights:
        model_conv.load_weights(weights, by_name=True)
        # do freezing
        for i in range(len(model_conv.layers)):
            if i < layerToEngage:
                model_conv.layers[i].trainable = False
    else:
        # do engagement
        layersToPop = 12 - layerToEngage
        for i in range(layersToPop):
            model_conv.pop()
    model = Sequential()
    model.add(model_conv)
    model.add(Dropout(0.5)) # changed
    model.add(Flatten(name="Flatten"))

    model.add(Dense(512))
    #if model_type == "P":
    model.add(BatchNormalization(momentum=0.9)) # changed
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    if model_type == "E":
        model.add(Dense(1, name="Ei_dense1"))
        model.add(Activation('sigmoid', name="Ei_act"))
        #model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
        #changed
        sgd = optimizers.SGD(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])
    elif model_type == "P":
        model.add(Dense(36))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    else:
        print("invalid model type: {0}".format(model_type))

    return model

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


beginTime = datetime.datetime.now()

# settings
img_size = 128
nbFilters = 16
gen_batch_size = 720
epochs = 10

totalDataSize = 72000
sizeOneBatch = 720
nbBatches = totalDataSize // sizeOneBatch # devide dataset into batches
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches
if drift_type == "appear":
    nbBaseBatches = 30

# prepare data
train_data_dir = os.path.abspath("../../data/NIST")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

bundleSize = 10 # we bundle batches

train_generator_bundle = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size * bundleSize,
    color_mode="grayscale",
    class_mode="categorical")

# training the C0
model_C0 = make_conv_model(nbFilters, False)

for i in range(nbBaseBatches):
#for i in range(1): # !!!!!!!!!!!!
    print(i)
    X, y = train_generator.next()

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

    model_C0.fit(X, y, batch_size=20, epochs = epochs, verbose=0)

C0Weights = "C0_weigths_{0}.h5".format(drift_type)
model_C0.save_weights(C0Weights)



# # model_E = build_model("E", "")
# model_P = build_model("P", "", nbFilters, layerToEngage)
# model_ms = build_model("E", "", nbFilters, layerToEngage)

# build all 12 possible patching networks
models_P = []
models_ms = []
totalNumberOfLayers = 12
for i in range(totalNumberOfLayers):
    model_P = build_model("P", "", nbFilters, i+1) # layer number is 1-based!
    models_P.append(model_P)
    model_ms = build_model("E", "", nbFilters, i+1)
    models_ms.append(model_ms)


# adaption: data changed
angle = 0 # for rotate
initPopulation = 4
nextPopulation = 2
nbOfFitnessToCompare = 2
for b in range(nbBaseBatches/bundleSize, nbBatches/bundleSize):
    print("bundle number: {0}".format(b))
    i = b * 10
    
    X_org, y_org = train_generator_bundle.next()
    
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
        X, y, _ = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        X, y, _ = transfer(X_org, y_org, i < nbBatches/2)


    # evolutionary algorithm
    generation = 1
    layers = range(totalNumberOfLayers)
    results = [] # elements are tuple of bundle number, layer number (0-based) and performance
    while len(layers) > 0:
        nbOfLayersToEngage = initPopulation if generation == 1 else nextPopulation
        random.shuffle(layers)
        selectedLayers = layersToEngage[0 : nbOfLayersToEngage]
        layers = layers[nbOfLayersToEngage : ]

        currentResult = []
        for l in layers:
            model_P = models_P[l]
            model_ms = models_ms[l]
            # evaluate on data first (equivalent to validation)
            acc = calc_accuracy(imag_size, model_C0, model_ms, model_P, X, y)
            # build data for ms
            x_ms = []
            y_ms = []
            for index in range(len(X)):
                predictC0 = model_C0.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)
                predictP = model_P.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)

                yLabel = np.argmax(y[index])
                if predictC0[0][yLabel] > predictP[0][yLabel]:
                    x_ms.append(X[index])
                    y_ms.append(0)
                else:
                    x_ms.append(X[index])
                    y_ms.append(1)
            # train the models
            model_ms.fit(x_ms, y_ms, batch_size = 20, epochs = epochs, verbose = 0)
            model_P.fit(X, y, batch_size = 20, epochs epochs, verbose = 0)
            # save result
            currentResult.append((b, l, acc))
        # sort  current results
        currentResult.sort(key=lambda tup:tup[2], reverse=True)
        # update results
        hasBetterLayer = False
        if generation == 1:
            results = currentResult
        else:
            for a in range(nbOfFitnessToCompare):
                acc_a = currentResult[a]
                for b in range(nbOfFitnessToCompare):
                    acc_b = results[b]
                    if acc_a > acc_b:
                        hasBetterLayer = True
                        results[b] = currentResult[a] # replace
            results.sort(key=lambda tup:tup[2], reverse=True)
            # check stop condition
            if hasBetterLayer == False:
                break
        generation += 1
    # save results
    print("RESULT: ", results[0])

endTime = datetime.datetime.now()
print(endTime - beginTime)




