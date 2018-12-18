import random
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, backend as K
import sys
import os
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
from pathlib import Path

from model_provider import *

nbFilters = 16

def MakePatchHiddenLayers(model, variant):
    if variant == 1:
        model.add(Dense(128))
    elif variant == 2:
        model.add(Dense(256))
    
    elif variant == 3:
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
    elif variant == 4:
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
    elif variant == 5:
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization(momentum=0.9))
    elif variant == 6:
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
    elif variant == 7:
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(BatchNormalization(momentum=0.9))
    elif variant == 8:
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
    elif variant == 9:
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(BatchNormalization(momentum=0.9))
    elif variant == 10:
        model.add(Dense(512))


def build_model(model_type, weights, nbFilters, layerToEngage, PVA = 10):
    model = None

    model_conv = make_conv_model(nbFilters, True)
    if weights:
        model_conv.load_weights(weights, by_name=True)
    else:
        # do engagement
        layersToPop = 12 - layerToEngage
        for i in range(layersToPop):
            model_conv.pop()
    model = Sequential()
    model.add(model_conv)
    model.add(Flatten(name="Flatten"))
    # patching
    # model.add(Dense(512))
    MakePatchHiddenLayers(model, PVA)
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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

# opt_type: engage or PA (patch architecture)
# model_type: P or E
# drift_type: flip, rotate, appear, remap, transfer
# layer: engaging layer (int)
# PAV: patch architecture variant (int)
def buildModelName(opt_type, model_type, drift_type, layer, PAV=None): 
    if PAV is None:
        return "models/nist_{0}_{1}_{2}_{3}.h5".format(opt_type, model_type, drift_type, layer)
    else:
        return "models/nist_{0}_{1}_{2}_{3}_{4}.h5".format(opt_type, model_type, drift_type, layer, PAV)

def buildOrLoadModel(h5FileName, model_type, layerToEngage, PVA):
    h5file = Path(h5FileName)
    if h5file.is_file():
        return load_model(h5FileName)
    else:
        return build_model(model_type, "", nbFilters, layerToEngage, PVA)

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

def UpdateResults(whole_results, current_results, nbOfFitness):
    hasBetterCurrentResult = False
    whole_results = whole_results[:nbOfFitness]
    current_results = current_results[:nbOfFitness]
    for wr in whole_results:
        for cr in current_results:
            if cr[2] > wr[2]:
                hasBetterCurrentResult = True
                break
    if hasBetterCurrentResult:
        whole_results = whole_results + current_results
        whole_results.sort(key=lambda tup:tup[2], reverse=True)
        whole_results = whole_results[:nbOfFitness]
    return (hasBetterCurrentResult, whole_results)


def Evolutionary(   totalNumberOfSelections, 
                    nbOfFitnessToCompare,
                    initPopulation,
                    nextPopulation,
                    # model_C0,
                    # models_P,
                    # models_ms,
                    img_size,
                    bundleIndex,
                    X,
                    y,
                    epochs,
                    drift_type,
                    opt_type,
                    engagementResult=None):
    generation = 1
    if opt_type == "PA":
        totalNumberOfSelections -= 1 # variant 10 is already calculated by engagement
    selections = list(range(totalNumberOfSelections))
    results = [] # elements are tuple of bundle number, selected index (0-based) and accuracy
    while len(selections) > 0:
        nbOfSelections = initPopulation if generation == 1 else nextPopulation
        random.shuffle(selections)
        selectionsInUse = selections[0 : nbOfSelections]
        selections = selections[nbOfSelections : ]

        currentResult = []
        for s in selectionsInUse:
            model_P = None
            model_ms = None
            model_C0 = None
            model_C0 = load_model("C0_weigths_{0}.h5".format(drift_type))
            if opt_type == "PA":
                engageLayer = engagementResult[1]
                model_P_name = buildModelName(opt_type, "P", drift_type, engageLayer+1, s+1)
                model_ms_name = buildModelName(opt_type, "E", drift_type, engageLayer+1, s+1)
                model_P = buildOrLoadModel(model_P_name, "P", engageLayer+1, s+1)
                model_ms = buildOrLoadModel(model_ms_name, "E", engageLayer+1, s+1)
            else:
                model_P_name = buildModelName(opt_type, "P", drift_type, s+1)
                model_ms_name = buildModelName(opt_type, "E", drift_type, s+1)
                # model_P = load_model(model_P_name)
                # model_ms = load_model(model_ms_name)
                model_P = buildOrLoadModel(model_P_name, "P", s+1, 10)
                model_ms = buildOrLoadModel(model_ms_name, "E", s+1, 10)
            # evaluate on data first (equivalent to validation)
            acc = calc_accuracy(img_size, model_C0, model_ms, model_P, X, y)
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
            x_ms = np.asarray(x_ms)
            x_ms = x_ms.reshape(-1, img_size, img_size, 1)
            model_ms.fit(x_ms, y_ms, batch_size = 20, epochs = epochs, verbose = 0)
            model_P.fit(X, y, batch_size = 20, epochs = epochs, verbose = 0)
            # save result
            currentResult.append((bundleIndex, s, acc))

            # save models and delete them to free memory
            model_P.save(model_P_name)
            model_ms.save(model_ms_name)
            del model_P
            del model_ms
            del model_C0
            K.clear_session()
        # sort  current results
        currentResult.sort(key=lambda tup:tup[2], reverse=True)
        print("CurrentResult: ", currentResult)
        # update results
        if generation == 1:
            if opt_type == "PA":
                results = currentResult + [(engagementResult[0], 9, engagementResult[2])] # add the result of engagement (variant 10)
                results.sort(key=lambda tup:tup[2], reverse=True)
                print("engaged result: ", results)
            else:
                results = currentResult
        else:
            hasBetterResult, results = UpdateResults(results, currentResult, nbOfFitnessToCompare)
            print("Results: ", results)
            # check stop condition
            if hasBetterResult == False and generation >=3:
                break
        generation += 1
    # save results
    print("RESULT: ", results[0])
    print("\n")
    return results