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

def buildModelName(opt_type, model_type, drift_type, layer): 
    return "models/nist_{0}_{1}_{2}_{3}.h5".format(opt_type, model_type, drift_type, layer)

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
    selections = list(range(totalNumberOfSelections))
    results = [] # elements are tuple of bundle number, selected index (0-based) and accuracy
    while len(selections) > 0:
        nbOfSelections = initPopulation if generation == 1 else nextPopulation
        random.shuffle(selections)
        selectionsInUse = selections[0 : nbOfSelections]
        selections = selections[nbOfSelections : ]

        currentResult = []
        for s in selectionsInUse:
            # model_P = models_P[s]
            # model_ms = models_ms[s]
            model_C0 = load_model("C0_weigths_{0}.h5".format(drift_type))
            model_P_name = buildModelName(opt_type, "P", drift_type, s+1)
            model_ms_name = buildModelName(opt_type, "E", drift_type, s+1)
            model_P = load_model(model_P_name)
            model_ms = load_model(model_ms_name)
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