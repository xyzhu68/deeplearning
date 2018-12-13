import random
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

Evolutionary(totalNumberOfSelections, 
             nbOfFitnessToCompare,
             initPopulation,
             nextPopulation,
             model_C0,
             models_P,
             models_ms,
             img_size,
             bundleIndex):
    generation = 1
    selections = range(totalNumberOfSelections)
    results = [] # elements are tuple of bundle number, selected index (0-based) and accuracy
    while len(selections) > 0:
        nbOfSelections = initPopulation if generation == 1 else nextPopulation
        random.shuffle(selections)
        selectionsInUse = selections[0 : nbOfSelections]
        selections = selections[nbOfSelections : ]

        currentResult = []
        for s in selectionsInUse:
            model_P = models_P[s]
            model_ms = models_ms[s]
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
            currentResult.append((bundleIndex, s, acc))
        # sort  current results
        currentResult.sort(key=lambda tup:tup[2], reverse=True)
        print("CurrentResult: ", currentResult)
        # update results
        hasBetterLayer = False
        if generation == 1:
            results = currentResult
        else:
            for a in range(nbOfFitnessToCompare):
                acc_current = currentResult[a]
                for b in range(len(nbOfFitnessToCompare)-1, -1, -1): # update the lower value first
                    acc_whole = results[b]
                    if acc_current > acc_whole:
                        hasBetterLayer = True
                        results[b] = currentResult[a] # replace
                        break
            results.sort(key=lambda tup:tup[2], reverse=True)
            print("Results: ", results)
            # check stop condition
            if hasBetterLayer == False and generation >=3:
                break
        generation += 1
    # save results
    print("RESULT: ", results[0])
    print("\n")
    return results