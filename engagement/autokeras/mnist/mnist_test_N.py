from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
#from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, backend as K
import sys
#import matplotlib.pyplot as plt
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



# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch nb: ", i)
    loopBegin = datetime.datetime.now()
    
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

    pred = model_C0.predict(X)
    pred_cls = np.argmax(pred, axis=1)
    y_cls = np.argmax(y, axis=1)
    leng = len(pred_cls)
    count = 0
    for i in range(leng):
        if pred_cls[i] == y_cls[i]:
            count += 1
    print("ACC: ", count / leng)
    bdddc.add_element(pred_cls, y_cls, classifier_changed=False)

    changed = bdddc.detected_change()

    print("changed: ", changed)

    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)


