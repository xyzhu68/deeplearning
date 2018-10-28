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
        
def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)


def build_model(model_type, weights):
    model = None
    if layerToEngage == 0:
        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
    else:
        model_conv = make_conv_model(64, True)
        if weights:
            model_conv.load_weights(weights, by_name=True)
            # do freezing
            for i in range(len(model_conv.layers)):
                model_conv.layers[i].trainable = False
        else:
            # do engagement
            layersToPop = 7 - layerToEngage
            if layersToPop < 0:
                print("layer to engage cannot be greater than 7")
                exit()
            for i in range(layersToPop):
                model_conv.pop()
        print(model_conv.summary())
        model = Sequential()
        model.add(model_conv)
        model.add(Flatten(name="Flatten"))
        model.add(Dense(512))
        model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    if model_type == "E":
        model.add(Dense(1, name="Ei_dense1"))
        model.add(Activation('sigmoid', name="Ei_act"))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
    elif model_type == "P":
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    else:
        print("invalid model type: {0}".format(model_type))

    print(model.summary())

    return model

def freeze_model(model):
    for i in range(len(model.layers)):
        if i < layerToEngage:
            model.layers[i].trainable = False
            print("frozen layer {0}".format(model.layers[i].name))
        else:
            print("free layer {0}".format(model.layers[i].name))

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_C0 = make_conv_model(64, False)
model_Base_Updated = make_conv_model(64, False)


#lossArray_E = [] # loss of Ei
accArray_E = [] # accuracy of Ei
#lossArray_P = [] # loss of Pi
accArray_P = [] # accuracy of Pi
accArray_MS = []
indices = [] # index of batches
accArray_Base = [] # accuray of C0 (base model)
accArray_Base_Updated = []
#lossArray_Base = [] # loss of C0
accEiPi = [] # accuracy of Ei + Pi
accMSPi = [] # accuracy of Model Selector + P
accArray_Freezing = [] # accuracy of freezing model

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
    #model_Base_Updated.fit(X, y, batch_size=50, epochs=10)

C0Weights = "C0_weigths_{0}.h5".format(drift_type)
model_C0.save_weights(C0Weights)

model_E = build_model("E", "")
model_P = build_model("P", "")
model_ms = build_model("E", "")
model_freezing = build_model("P", C0Weights) #make_conv_model(64, False)
#model_freezing.load_weights(C0Weights)
#freeze_model(model_freezing)
#model_freezing.compile(loss='categorical_crossentropy', 
#                optimizer='adadelta', metrics=['categorical_accuracy'])

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

    if layerToEngage == 0:
        X = X.reshape(-1, 784)

    accEiPi.append(calc_accuracy(model_C0, model_E, model_P, X, y))
    accMSPi.append(calc_accuracy(model_C0, model_ms, model_P, X, y))
    x_Ei = []
    y_Ei = []
    x_ms = []
    y_ms = []
    base_correct = 0
    for index in range(len(X)):
        predictC0 = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        predictP = model_P.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if np.argmax(predictC0) == np.argmax(y[index]):
            x_Ei.append(X[index])
            y_Ei.append(0)
            base_correct += 1
        else:
            x_Ei.append(X[index])
            y_Ei.append(1)

        yLabel = np.argmax(y[index])
        #print(predictC0[0])
        if predictC0[0][yLabel] > predictP[0][yLabel]:
            x_ms.append(X[index])
            y_ms.append(0)
        else:
            x_ms.append(X[index])
            y_ms.append(1)

    x_Ei = np.asarray(x_Ei)
    x_Ei = x_Ei.reshape(-1, 28, 28, 1)
    loss_E, acc_E = model_E.evaluate(x_Ei, y_Ei, batch_size=50)
    accArray_E.append(acc_E)
    h_Ei = model_E.fit(x_Ei, y_Ei, batch_size = 50, epochs = 10)
    #accArray_E.append(np.mean(h_Ei.history["binary_accuracy"]))
    
    x_ms = np.asarray(x_ms)
    x_ms = x_ms.reshape(-1, 28, 28, 1)
    loss_ms, acc_ms = model_ms.evaluate(x_ms, y_ms, batch_size=50)
    accArray_MS.append(acc_ms)
    h_ms = model_ms.fit(x_ms, y_ms, batch_size = 50, epochs = 10)
    #accArray_MS.append(np.mean(h_ms.history["binary_accuracy"]))
    
    
    # if len(x_Pi) > 0:
    #     x_Pi = np.asarray(x_Pi)
    #     x_Pi = x_Pi.reshape(-1, 28, 28, 1)
    #     y_Pi = np.asarray(y_Pi)
    #     y_Pi = y_Pi.reshape(-1, 10)
    #     h_Pi = model_Ci.fit(x_Pi, y_Pi, batch_size = 50, epochs = 10)
    #     accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
    #     lossArray_P.append(np.mean(h_Pi.history["loss"]))

    #     # accEiPi.append(calc_accuracy(model_C0, model_Ei, model_Ci, x_Pi, y_Pi))
    # else:
    #     accArray_P.append(None)
    #     lossArray_P.append(None)
    #     # accEiPi.append(None)

    loss_P, acc_P = model_P.evaluate(X, y, batch_size=50)
    accArray_P.append(acc_P)
    h_Pi = model_P.fit(X, y, batch_size=50, epochs=10)
    #accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
    

    loss_bu, acc_bu = model_Base_Updated.evaluate(X, y, batch_size=50)
    accArray_Base_Updated.append(acc_bu)
    h_base_updated = model_Base_Updated.fit(X, y, batch_size=50, epochs=10)
    #accArray_Base_Updated.append(np.mean(h_base_updated.history["categorical_accuracy"]))

    loss_freezing, acc_freezing = model_freezing.evaluate(X, y, batch_size=50)
    accArray_Freezing.append(acc_freezing)
    h_freezing = model_freezing.fit(X, y, batch_size=50, epochs=10)
    #accArray_Freezing.append(np.mean(h_freezing.history["categorical_accuracy"]))
    
    
    indices.append(i)

    accArray_Base.append(base_correct / len(X))
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "mnist_engage_{0}_{1}.npz".format(drift_type, layerToEngage)
np.savez(npFileName, accBase = accArray_Base,
                     accBaseUpdated = accArray_Base_Updated,
                     accE = accArray_E,
                     accMS = accArray_MS,
                     accP = accArray_P,
                     accMSPi = accMSPi,
                     accEiPi = accEiPi,
                     accFreezing = accArray_Freezing,
                     indices=indices,
                     duration=str(endTime - beginTime))

# save models
# model_C0.save("model_C0_simple_{0}_{1}.h5".format(drift_type, nbFilters))
# model_Ei.save("model_Ei_simple_{0}_{1}.h5".format(drift_type, nbFilters))
# model_Ci.save("model_Pi_simple_{0}_{1}.h5".format(drift_type, nbFilters))

# result of accuracy
# plt.plot(indices, accArray_Base, label="acc base")
# plt.plot(indices, accArray_E, label="acc Ei")
# plt.plot(indices, accArray_P, label="acc Pi")
# plt.plot(indices, accEiPi, label="acc Ei+Pi")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Batch")
# plt.legend()
# plt.show()


