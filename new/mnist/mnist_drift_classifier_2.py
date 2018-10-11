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
filepath_test = "test.arff"

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

resnet = False
if nbArgs > 2:
    resnet =  sys.argv[2] == "resnet"

freeze_add_block = -1
if nbArgs > 3:
    freeze_add_block = int(sys.argv[3])

nbFilters = 64
if nbArgs > 4:
    nbFilters = int(sys.argv[4])

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

def make_resnet_model(Ei, n, weights):
    print("resnet used")
    model_resnet = make_resnet()
    if n >= 0 and weights:
        model_resnet.load_weights(weights)
    else:
        print("from scratch")
    
    count_add = 0
    for l in model_resnet.layers:
        if l.name.startswith("add"):
            count_add += 1
        if count_add < n and not l.name.startswith("input_"):
            l.trainable = False
            print("Frozen layer {0}".format(l.name))
        else:
            print("Free layer {0}".format(l.name))
                
    
    input = Input(shape=(28,28,1))
    out = model_resnet(input)
    out = Flatten()(out)
    out = Dense(units=128)(out)
    out = Activation(relu)(out)
    if Ei:
        out = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(out)
        out = Activation(sigmoid)(out)
    else:
        out = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(out)
        out = Activation(softmax)(out)
    model = Model(inputs=input, outputs=out)
    lossfunc = "binary_crossentropy" if Ei else "categorical_crossentropy"
    met = "binary_accuracy" if Ei else "categorical_accuracy"
    model.compile(loss=lossfunc, optimizer='adam', metrics=[met])
    
    return model, model_resnet

def make_simple_model(Ei, n, weights):
    print("simple model used")
    model = make_simple_conv(Ei, nbFilters)
    if weights:
        model.load_weights(weights)
    for i in range(len(model.layers)):
        if i < n:
            model.layers[i].trainable = False

    return model
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

model_resnet = None
if resnet:
    model_C0, model_resnet = make_resnet_model(False, -1, "") 
else:
    model_C0 = make_simple_conv(False, nbFilters)
 
# model_Ci = make_resnet_model(False, freeze_add_block) if resnet else make_model(False)

# model_Ei = make_resnet_model(True, freeze_add_block) if resnet else make_model(True)


lossArray_E = []
accArray_E = []
lossArray_P = []
accArray_P = []
indices = []
accArray_Base = []
lossArray_Base = []
accEiPi = []

# get data
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
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

    history = model_C0.fit(X, y, batch_size=50, epochs = 10)
    accArray_Base.append(np.mean(history.history["categorical_accuracy"]))
    lossArray_Base.append(np.mean(history.history["loss"]))
    indices.append(i)

    lossArray_E.append(None)
    accArray_E.append(None)
    lossArray_P.append(None)
    accArray_P.append(None)
    accEiPi.append(None)

if resnet:
    C0Weights = "C0_weights_resnet_{0}.h5".format(drift_type)
    model_resnet.save_weights(C0Weights)
else:
    C0Weights = "C0_weigths_simple_{0}.h5".format(drift_type)
    model_C0.save_weights(C0Weights)

# C0Weights = "C0_weights_resnet_{0}.h5".format(drift_type) if resnet else "C0_weigths_simple_{0}.h5".format(drift_type)
# model_C0.save_weights(C0Weights)

freeze_Ci = 6
freeze_Ei = -1
if freeze_add_block >= 0:
    freeze_Ci = freeze_add_block
    freeze_Ei = freeze_add_block
# model_Ci, _ = make_resnet_model(False, freeze_Ci, C0Weights) if resnet else make_simple_model(False, 4, C0Weights)
# model_Ei, _ = make_resnet_model(True, freeze_Ei, "") if resnet else make_simple_model(True, 0, "")
if resnet:
    model_Ci, _ = make_resnet_model(False, freeze_Ci, C0Weights)
    model_Ei, _ = make_resnet_model(True, freeze_Ei, C0Weights)
else:
    model_Ci = make_simple_model(False, 4, C0Weights)
    model_Ei = make_simple_model(True, 0, "")

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X_org, y_org = next(gen)
    
    data_changed = True
    X = None
    y = None
    if drift_type == "flip":
        X, y, _ = flip_images(X_org, y_org, i >= nbBatches/2)
        data_changed = i >= nbBatches/2
    elif drift_type == "appear":
        X, y, _ = appear(X_org, y_org, False)
    elif drift_type == "remap":
        X, y, _ = remap(X_org, y_org, i < nbBatches/2)
        data_changed = i >= nbBatches/2
    elif (drift_type == "rotate"):
        if i > 50 and i < 85 and angle <= 180:
            angle += 5
        else:
            angle = 0
            data_changed = False
        X, y, _ = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        X, y, _ = transfer(X_org, y_org, i < nbBatches/2)
        data_changed = i >= nbBatches/2

    accEiPi.append(calc_accuracy(model_C0, model_Ei, model_Ci, X, y))
    x_Ei = []
    y_Ei = []
    x_Pi = []
    y_Pi = []
    for index in range(len(X)):
        predict = model_C0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if np.argmax(predict) == np.argmax(y[index]):
            x_Ei.append(X[index])
            y_Ei.append(0)
        else:
            x_Ei.append(X[index])
            y_Ei.append(1)
            x_Pi.append(X[index])
            y_Pi.append(y[index])

    if len(x_Ei) > 0:
        x_Ei = np.asarray(x_Ei)
        x_Ei = x_Ei.reshape(-1, 28, 28, 1)
        h_Ei = model_Ei.fit(x_Ei, y_Ei, batch_size = 50, epochs = 10)
        accArray_E.append(np.mean(h_Ei.history["binary_accuracy"]))
        lossArray_E.append(np.mean(h_Ei.history["loss"]))
    else:
        accArray_E.append(None)
        lossArray_E.append(None)

    if len(x_Pi) > 0:
        x_Pi = np.asarray(x_Pi)
        x_Pi = x_Pi.reshape(-1, 28, 28, 1)
        y_Pi = np.asarray(y_Pi)
        y_Pi = y_Pi.reshape(-1, 10)
        h_Pi = model_Ci.fit(x_Pi, y_Pi, batch_size = 50, epochs = 10)
        accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
        lossArray_P.append(np.mean(h_Pi.history["loss"]))

        # accEiPi.append(calc_accuracy(model_C0, model_Ei, model_Ci, x_Pi, y_Pi))
    else:
        accArray_P.append(None)
        lossArray_P.append(None)
        # accEiPi.append(None)
    
    
    indices.append(i)

    accArray_Base.append(None)
    lossArray_Base.append(None)
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "mnist_drift_{0}_simple_{1}.npz".format(drift_type, nbFilters)
if resnet:
    if freeze_add_block >= 0:
        npFileName = "mnist_drift_{0}_resnet_{1}.npz".format(drift_type, freeze_add_block)
    else:
        npFileName = "mnist_drift_{0}_resnet_fs.npz".format(drift_type)
np.savez(npFileName, accBase = accArray_Base,
                     lossBae = lossArray_Base,
                     accE = accArray_E,
                     lossE = lossArray_E,
                     accP = accArray_P,
                     lossP = lossArray_P,
                     accEiPi = accEiPi,
                     indices=indices,
                     duration=str(endTime - beginTime))

# save models
if resnet:
    freeze = str(freeze_add_block) if freeze_add_block >= 0 else "fs"
    model_C0.save("model_C0_resnet_{0}_{1}.h5".format(drift_type, freeze))
    model_Ei.save("model_Ei_resnet_{0}_{1}.h5".format(drift_type, freeze))
    model_Ci.save("model_Pi_resnet_{0}_{1}.h5".format(drift_type, freeze))
else:
    model_C0.save("model_C0_simple_{0}_{1}.h5".format(drift_type, nbFilters))
    model_Ei.save("model_Ei_simple_{0}_{1}.h5".format(drift_type, nbFilters))
    model_Ci.save("model_Pi_simple_{0}_{1}.h5".format(drift_type, nbFilters))

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


