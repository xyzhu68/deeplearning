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
from beta_distribution_drift_detector.bdddc import BDDDC

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

# 1 -- 7
layerToEngage = 7
if nbArgs > 2:
    layerToEngage =  int(sys.argv[2])


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
        

def calc_accuracy(modelC0, modelE, modelC, X, y, intermedia_input):
    predictE = modelE.predict(intermedia_input)
    index = 0
    correct = 0
    predict = None
    for p in predictE:
        if p[0] > 0.5:
            predict = modelC.predict(intermedia_input[index].reshape((1,) + intermedia_input.shape[1:]), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)

        if np.argmax(predict) == np.argmax(y[index]):
            correct += 1

        index += 1

    return correct / len(X)


def build_C0_intermedia(C0_model):
    layer = C0_model.layers[layerToEngage - 1]
    intermediate_layer_model = Model(inputs = C0_model.input, outputs = layer.output)
    return intermediate_layer_model

def build_patch_network(model_type, input_shape):
    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if model_type == "E":
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
    elif model_type == "P":
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    else:
        print("invalid model type: {0}".format(model_type))
        exit()

    print(model.summary())

    return model

def freeze_model(model):
    for i in range(len(model.layers)):
        if i < layerToEngage:
            model.layers[i].trainable = False
            print("frozen layer {0}".format(model.layers[i].name))
        else:
            print("free layer {0}".format(model.layers[i].name))

def build_freez_model(weights):
    model_conv = make_conv_model(64, False)
    model_conv.set_weights(weights)
    freeze_model(model_conv)
    return model_conv

def is_drift(C0_model, detector, X, y):
    pred = C0_model.predict(X)
    pred_cls = np.argmax(pred, axis=1)
    y_cls = np.argmax(y, axis=1)
    
    detector.add_element(pred_cls, y_cls, classifier_changed=False)

    return detector.detected_change()

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_C0 = make_conv_model(64, False)
model_Base_Updated = make_conv_model(64, False)


accArray_E = [] # accuracy of Ei
accArray_P = [] # accuracy of Pi
accArray_MS = [] # accuracy of MS
indices = [] # index of batches for E and P ...
indices_C0 = [] # index of batches for C0
accArray_Base = [] # accuray of C0 (base model)
accArray_Base_Updated = []
accEiPi = [] # accuracy of Ei + Pi
accMSPi = [] # accuracy of Model Selector + P
accArray_Freezing = [] # accuracy of freezing model

# for change point detection
bdddc = BDDDC()

# get data
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
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

    model_C0.fit(X, y, batch_size=50, epochs = 10)


C0_intermedia = build_C0_intermedia(model_C0)
model_E = build_patch_network("E", C0_intermedia.layers[-1].output_shape[1:])
model_P = build_patch_network("P", C0_intermedia.layers[-1].output_shape[1:])
model_ms = build_patch_network("E", C0_intermedia.layers[-1].output_shape[1:])

model_freezing = build_freez_model(model_C0.get_weights())

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
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


    if is_drift(model_C0, bdddc, X, y):
        print("drift detected at batch {0}".format(i))
        C0_inter_data = C0_intermedia.predict(X)
        accEiPi.append(calc_accuracy(model_C0, model_E, model_P, X, y, C0_inter_data))
        accMSPi.append(calc_accuracy(model_C0, model_ms, model_P, X, y, C0_inter_data))
        x_Ei = []
        y_Ei = []
        x_ms = []
        y_ms = []
        # build data for E and ms
        for index in range(len(X)):
            predictC0 = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
            predictP = model_P.predict(C0_inter_data[index].reshape((1,) + C0_inter_data.shape[1:]), batch_size=1)
            if np.argmax(predictC0) == np.argmax(y[index]):
                x_Ei.append(C0_inter_data[index])
                y_Ei.append(0)
            else:
                x_Ei.append(C0_inter_data[index])
                y_Ei.append(1)

            yLabel = np.argmax(y[index])
            if predictC0[0][yLabel] > predictP[0][yLabel]:
                x_ms.append(C0_inter_data[index])
                y_ms.append(0)
            else:
                x_ms.append(C0_inter_data[index])
                y_ms.append(1)

    
        x_Ei = np.asarray(x_Ei)
        x_Ei = x_Ei.reshape((-1, ) + C0_inter_data.shape[1:])
        loss_E, acc_E = model_E.evaluate(x_Ei, y_Ei, batch_size=50)
        accArray_E.append(acc_E)
        model_E.fit(x_Ei, y_Ei, batch_size = 50, epochs = 10)
        
        x_ms = np.asarray(x_ms)
        x_ms = x_ms.reshape((-1, ) + C0_inter_data.shape[1:])
        loss_ms, acc_ms = model_ms.evaluate(x_ms, y_ms, batch_size=50)
        accArray_MS.append(acc_ms)
        model_ms.fit(x_ms, y_ms, batch_size = 50, epochs = 10)

        loss_P, acc_P = model_P.evaluate(C0_inter_data, y, batch_size=50)
        accArray_P.append(acc_P)
        model_P.fit(C0_inter_data, y, batch_size=50, epochs=10)


        loss_bu, acc_bu = model_Base_Updated.evaluate(X, y, batch_size=50)
        accArray_Base_Updated.append(acc_bu)
        model_Base_Updated.fit(X, y, batch_size=50, epochs=10)

        loss_freezing, acc_freezing = model_freezing.evaluate(X, y, batch_size=50)
        accArray_Freezing.append(acc_freezing)
        model_freezing.fit(X, y, batch_size=50, epochs=10)

        indices.append(i)
    
    loss_c0, acc_c0 = model_C0.evaluate(X, y, batch_size=50)
    accArray_Base.append(acc_c0)
    indices_C0.append(i)
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "mnist_engage_{0}_{1}_weighted_beta.npz".format(drift_type, layerToEngage)
np.savez(npFileName, accBase = accArray_Base,
                     indicesC0 = indices_C0,
                     accBaseUpdated = accArray_Base_Updated,
                     accE = accArray_E,
                     accMS = accArray_MS,
                     accP = accArray_P,
                     accMSPi = accMSPi,
                     accEiPi = accEiPi,
                     accFreezing = accArray_Freezing,
                     indices=indices,
                     duration=str(endTime - beginTime))



