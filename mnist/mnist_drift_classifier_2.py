from scipy.io import arff
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import matplotlib.pyplot as plt
from data_provider import *

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

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

# build models using weights from base
model = load_model('model_base.h5')
digit_input = model.input
out_flatten = model.get_layer("Flatten")
visual_model = Model(digit_input, out_flatten.output)

class_input = Input(shape=(28,28,1))
out = visual_model(class_input)
out = Dense(128, activation="relu")(out)
out = Dropout(0.5)(out)

# error clf
out_Ei = Dense(1, activation="sigmoid")(out)
model_Ei = Model(class_input, out_Ei)
model_Ei.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# patching clf
out_Ci = Dense(10, activation="softmax")(out)
model_Ci = Model(class_input, out_Ci)
model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

lossArray_E = []
accArray_E = []
lossArray = []
accArray = []
indices = []
accChainedArray = []

# get data
#X_train, y_train = load_data(filepath_train)
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
    print(i)
    
    X, y = next(gen)
    #y_E = np.zeros(sizeOneBatch)
    #y = to_categorical(y, 10)

    if drift_type == "flip":
        X, y, y_E = flip_images(X, y, False)
    elif drift_type == "appear":
        X, y, y_E = appear(X, y, True)
    elif drift_type == "remap":
        X, y, y_E = remap(X, y, True)
    elif (drift_type == "rotate"):
        X, y, y_E = rot(X, y, 0)

    print(X.shape)
    print(len(X))

    result_E = model_Ei.fit(X, y_E, batch_size=50, epochs=10)
    result_C = model_Ci.fit(X, y, batch_size=50, epochs=10)

    lossArray.append(np.mean(result_C.history["loss"]))
    accArray.append(np.mean(result_C.history["acc"]))
    lossArray_E.append(np.mean(result_E.history["loss"]))
    accArray_E.append(np.mean(result_E.history["acc"]))
    indices.append(i)
    accChained = calc_accuracy(model, model_Ei, model_Ci, X, y)
    accChainedArray.append(accChained)

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X, y = next(gen)
    # X, y = flip_images(X, y)
    # y_E = np.full(sizeOneBatch, 1.0)
    # y = to_categorical(y, 10)

    if drift_type == "flip":
        X, y, y_E = flip_images(X, y, i >= nbBatches/2)
    elif drift_type == "appear":
        X, y, y_E = appear(X, y, False)
    elif drift_type == "remap":
        X, y, y_E = remap(X, y, i < nbBatches/2)
    elif (drift_type == "rotate"):
        if i > 50 and i < 90:
            angle += 6
        else:
            angle = 0
        X, y, y_E = rot(X, y, angle)

    # evaluate
    loss_E, acc_E = model_Ei.evaluate(X, y_E, batch_size=50)
    lossArray_E.append(loss_E)
    accArray_E.append(acc_E)
    loss, acc = model_Ci.evaluate(X, y, batch_size=50)
    lossArray.append(loss)
    accArray.append(acc)
    indices.append(i)
    accChained = calc_accuracy(model, model_Ei, model_Ci, X, y)
    accChainedArray.append(accChained)

    # training
    model_Ei.fit(X, y_E, batch_size=50, epochs=10)
    model_Ci.fit(X, y, batch_size=50, epochs=10)

npFileName = "mnist_drift_{0}_weighted.npz".format(drift_type)
np.savez(npFileName, acc=accArray, acc_E=accArray_E, 
                    loss=lossArray, loss_E=lossArray_E,
                    accChained=accChainedArray,
                    indices=indices)

# result of accuracy
plt.plot(indices, accArray, label="acc patching clf")
plt.plot(indices, accArray_E, label="acc error clf")
plt.plot(indices, accChainedArray, label="acc Ei+Ci")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch")
plt.legend()
plt.show()

# # result of loss
# plt.plot(indices, lossArray, label="loss patching clf")
# plt.plot(indices, lossArray_E, label="loss error clf")
# plt.title("Loss")
# plt.ylabel("Loss")
# plt.xlabel("Batch")
# plt.show()