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
        

# def flip_images(X, y):
#     datagen = ImageDataGenerator(
#         horizontal_flip=True,
#         vertical_flip=True,
#     )
#     datagen.fit(X)
#     data_it = datagen.flow(X, y, batch_size=1)
    
#     data_list = []
#     y_list = []
#     batch_index = 0
#     while batch_index <= data_it.batch_index:
#         data = data_it.next()
#         #x_data = data[0].reshape((784,))
#         data_list.append(data[0])
#         y_list.append(data[1])
#         batch_index = batch_index + 1

#     data_array = np.asarray(data_list)
#     data_array = data_array.reshape(-1, 28, 28, 1)
#     y_array = np.asarray(y_list)
#     return (data_array, y_array)

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

# get data
#X_train, y_train = load_data(filepath_train)
gen = data_generator(sizeOneBatch)
# training in base phase
for i in range(nbBaseBatches):
    print(i)
    
    X, y = next(gen)
    #y_E = np.zeros(sizeOneBatch)
    #y = to_categorical(y, 10)

    #X, y, y_E = flip_images(X, y, True)
    X, y, y_E = appear(X, y, True)
    print(X.shape)
    print(len(X))

    result_E = model_Ei.fit(X, y_E, batch_size=50, epochs=10)
    result_C = model_Ci.fit(X, y, batch_size=50, epochs=10)

    lossArray.append(np.mean(result_C.history["loss"]))
    accArray.append(np.mean(result_C.history["acc"]))
    lossArray_E.append(np.mean(result_E.history["loss"]))
    accArray_E.append(np.mean(result_E.history["acc"]))
    indices.append(i)

# adaption: data changed
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X, y = next(gen)
    # X, y = flip_images(X, y)
    # y_E = np.full(sizeOneBatch, 1.0)
    # y = to_categorical(y, 10)

    #X, y, y_E = flip_images(X, y, False)
    X, y, y_E = appear(X, y, False)

    # evaluate
    loss_E, acc_E = model_Ei.evaluate(X, y_E, batch_size=50)
    lossArray_E.append(loss_E)
    accArray_E.append(acc_E)
    loss, acc = model_Ci.evaluate(X, y, batch_size=50)
    lossArray.append(loss)
    accArray.append(acc)
    indices.append(i)

    # training
    model_Ei.fit(X, y_E, batch_size=50, epochs=10)
    model_Ci.fit(X, y, batch_size=50, epochs=10)


np.savez("mnist_drift_clf_results_weighted.npz", acc=accArray, acc_E=accArray_E, loss=lossArray, loss_E=lossArray_E, indices=indices)
# plt.plot(indices, lossArray, label="loss")
# plt.legend()
# plt.show()

# plt.plot(indices, accArray, label="accuracy")
# plt.legend()
# plt.show()

# plt.plot(indices, lossArray_E, label="loss Error Clf")
# plt.legend()
# plt.show()

# plt.plot(indices, accArray_E, label="accuracy Error Clf")
# plt.legend()
# plt.show()

# result of accuracy
plt.plot(indices, accArray, label="acc patching clf")
plt.plot(indices, accArray_E, label="acc error clf")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch")
plt.show()

# result of loss
plt.plot(indices, lossArray, label="loss patching clf")
plt.plot(indices, lossArray_E, label="loss error clf")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.show()