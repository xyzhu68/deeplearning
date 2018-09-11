from scipy.io import arff
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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

# model from scratch
def make_model(Ei):
    nb_filters = 64
    nb_conv = 3
    img_rows = 28
    img_cols = 28
    nb_pool = 2

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    padding='valid',
                    input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid'))
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten(name="Flatten"))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if Ei:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    else:
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    

    return model

def make_weighted_model(Ei):
    model = load_model('model_base.h5')
    digit_input = model.input
    out_flatten = model.get_layer("Flatten")
    visual_model = Model(digit_input, out_flatten.output)

    class_input = Input(shape=(28,28,1))
    out = visual_model(class_input)
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.5)(out)

    if Ei: # error clf
        out_Ei = Dense(1, activation="sigmoid")(out)
        model_Ei = Model(class_input, out_Ei)
        model_Ei.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return model_Ei
    else: # patching clf
        out_Ci = Dense(10, activation="softmax")(out)
        model_Ci = Model(class_input, out_Ci)
        model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return model_Ci

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

# build models using weights from base
# model = load_model('model_base.h5')
# digit_input = model.input
# out_flatten = model.get_layer("Flatten")
# visual_model = Model(digit_input, out_flatten.output)

# class_input = Input(shape=(28,28,1))
# out = visual_model(class_input)
# out = Dense(128, activation="relu")(out)
# out = Dropout(0.5)(out)

# # error clf
# out_Ei = Dense(1, activation="sigmoid")(out)
# model_Ei = Model(class_input, out_Ei)
# model_Ei.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# # patching clf
# out_Ci = Dense(10, activation="softmax")(out)
# model_Ci = Model(class_input, out_Ci)
# model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model_Ci = make_weighted_model(False)
model_Ei = make_weighted_model(True)

lossArray_E = []
accArray_E = []
lossArray = []
accArray = []
indices = []
accChainedArray = []

# get data
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
    print(i)
    
    X, y = next(gen)

    y_E = None
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
    
    X_org, y_org = next(gen)

    if drift_type == "flip":
        X, y, y_E = flip_images(X_org, y_org, i >= nbBatches/2)
    elif drift_type == "appear":
        X, y, y_E = appear(X_org, y_org, False)
    elif drift_type == "remap":
        X, y, y_E = remap(X_org, y_org, i < nbBatches/2)
    elif (drift_type == "rotate"):
        if i > 50 and i < 85 and angle <= 180:
            angle += 5
        else:
            angle = 0
        X, y, y_E = rot(X_org, y_org, angle)

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
    y_E_org = np.zeros(len(y_E))
    X_combine = np.concatenate((X_org, X))
    y_combine = np.concatenate((y_E_org, y_E))
    X_combine, y_combine = shuffle(X_combine, y_combine)
    model_Ei.fit(X_combine, y_combine, batch_size=50, epochs=10)
    #model_Ei.fit(X, y_E, batch_size=50, epochs=10)
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

