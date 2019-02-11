from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, backend as K
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

hours = 12
if nbArgs > 2:
    hours = int(sys.argv[2])

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
    #print("predict ei: ", predictEi)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[1] > p[0]:
            predict = modelCi.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    print("ACC: ", correct / len(X))
    return correct / len(X)

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
indices = [] # index of batches
accArray_Base = [] # accuray of C0 (base model)
accEiPi = [] # accuracy of Ei + Pi

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
        X, y = flip_images(X, y, False)
    elif drift_type == "remap":
        X, y = remap(X, y, True)
    else:
        print("Unknown drift type")
        exit()

    model_C0.fit(X, y, batch_size=50, epochs = 10)

# C0Weights = "C0_weigths_{0}.h5".format(drift_type)
# model_C0.save_weights(C0Weights)

model_E = load_model('autokeras_mnist_Ei_{0}_{1}.h5'.format(drift_type, hours))
x = model_E.output
x = Activation('softmax', name='activation_add')(x)
model_E = Model(model_E.input, x)

if drift_type == "flip":
    model_E.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    sgd = optimizers.SGD(lr=0.001)
    model_E.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model_P = load_model('autokeras_mnist_Ci_{0}_{1}.h5'.format(drift_type, hours))
x = model_P.output
x = Activation('softmax', name='activation_add')(x)
model_P = Model(model_P.input, x)

if drift_type == "flip":
    model_P.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    sgd2 = optimizers.SGD(lr=0.001)
    model_P.compile(optimizer=sgd2, loss='categorical_crossentropy', metrics=['accuracy'])

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch nb: ", i)
    loopBegin = datetime.datetime.now()
    
    X_org, y_org = next(gen)
    
    X = None
    y = None
    if drift_type == "flip":
        X, y = flip_images(X_org, y_org, i >= nbBatches/2)
    elif drift_type == "remap":
        X, y = remap(X_org, y_org, i < nbBatches/2)

    if is_drift(model_C0, bdddc, X, y):
        accEiPi.append(calc_accuracy(model_C0, model_E, model_P, X, y))
        x_Ei = []
        y_Ei = []
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

        y_Ei = to_categorical(y_Ei, 2)
        x_Ei = np.asarray(x_Ei)
        x_Ei = x_Ei.reshape(-1, 28, 28, 1)
        loss_E, acc_E = model_E.evaluate(x_Ei, y_Ei, batch_size=50)
        accArray_E.append(acc_E)
        model_E.fit(x_Ei, y_Ei, batch_size = 50, epochs = 10)
        
        if drift_type == "remap":
            y = np.argmax(y, axis=1)
            y = to_categorical(y, 5)
            # print(y)
            # print(y.shape)
        loss_P, acc_P = model_P.evaluate(X, y, batch_size=50)
        accArray_P.append(acc_P)
        print("Ci ACC: ", acc_P)
        model_P.fit(X, y, batch_size=50, epochs=10)
        
        indices.append(i)

        accArray_Base.append(base_correct / len(X))

        loopEnd = datetime.datetime.now()
        print("loop time: ", loopEnd - loopBegin)
        # break # !!!!!!!!!!!!!!!
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "mnist_ak_{0}_{1}_sgd.npz".format(drift_type, hours)
np.savez(npFileName, accBase = accArray_Base,
                     accE = accArray_E,
                     accP = accArray_P,
                     accEiPi = accEiPi,
                     indices=indices,
                     duration=str(endTime - beginTime))



