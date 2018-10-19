from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import os
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


#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

nbFreeze = -1
if nbArgs > 2:
    nbFreeze = int(sys.argv[2])

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

        
def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, 128, 128, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 128, 128, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
img_size = 128
gen_batch_size = 700 #20
epochs = 20

totalDataSize = 70000
sizeOneBatch = 700
nbBatches = totalDataSize // sizeOneBatch # devide dataset into batches
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches

# prepare data
train_data_dir = os.path.abspath("../../../by_class_2")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

model_C0 = make_simple_model(False, "")

lossArray_E = []
accArray_E = []
lossArray_P = []
accArray_P = []
indices = []
accArray_Base = []
lossArray_Base = []
accEiPi = []


# training in base phase
for i in range(nbBaseBatches):
    print(i)
    X, y = train_generator.next()
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

    history = model_C0.fit(X, y, batch_size=10, epochs = 10)
    accArray_Base.append(np.mean(history.history["categorical_accuracy"]))
    lossArray_Base.append(np.mean(history.history["loss"]))
    indices.append(i)

    lossArray_E.append(None)
    accArray_E.append(None)
    lossArray_P.append(None)
    accArray_P.append(None)
    accEiPi.append(None)

freeze = "fs" if nbFreeze <= 0 else str(nbFreeze)
C0Weights = "C0_weigths_simple_{0}_{1}.h5".format(drift_type, freeze)
model_C0.save_weights(C0Weights)

model_Ci = make_simple_model(False, "Ci")
if nbFreeze > 0:
    model_Ci.load_weights(C0Weights, by_name=True)
for i in range(len(model_Ci.layers)):
    l = model_Ci.layers[i]
    if i < nbFreeze:
        l.trainable = False
        print("frozen layer {0}".format(l.name))
    else:
        print("free layer {0}".format(l.name))

model_Ei = make_simple_model(True, "Ei")
if nbFreeze > 0:
     model_Ei.load_weights(C0Weights, by_name=True)
for i in range(len(model_Ei.layers)):
    l = model_Ei.layers[i]
    if i < nbFreeze:
        l.trainable = False
        print("frozen layer {0}".format(l.name))
    else:
        print("free layer {0}".format(l.name))

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X_org, y_org = train_generator.next()
    
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
        if i > nbBatches // 2:
            angle += 5
            if angle > 180:
                angle = 180
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
    base_correct = 0
    for index in range(len(X)):
        predict = model_C0.predict(X[index].reshape(1, img_size, img_size, 1), batch_size=1)
        if np.argmax(predict) == np.argmax(y[index]):
            x_Ei.append(X[index])
            y_Ei.append(0)
            base_correct += 1
        else:
            x_Ei.append(X[index])
            y_Ei.append(1)
            x_Pi.append(X[index])
            y_Pi.append(y[index])

    if len(x_Ei) > 0:
        x_Ei = np.asarray(x_Ei)
        x_Ei = x_Ei.reshape(-1, img_size, img_size, 1)
        h_Ei = model_Ei.fit(x_Ei, y_Ei, batch_size = 10, epochs = 10)
        accArray_E.append(np.mean(h_Ei.history["binary_accuracy"]))
        lossArray_E.append(np.mean(h_Ei.history["loss"]))
    else:
        accArray_E.append(None)
        lossArray_E.append(None)

    # if len(x_Pi) > 0:
    #     x_Pi = np.asarray(x_Pi)
    #     x_Pi = x_Pi.reshape(-1, img_size, img_size, 1)
    #     y_Pi = np.asarray(y_Pi)
    #     y_Pi = y_Pi.reshape(-1, 36)
    #     h_Pi = model_Ci.fit(x_Pi, y_Pi, batch_size = 10, epochs = 10)
    #     accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
    #     lossArray_P.append(np.mean(h_Pi.history["loss"]))
    # else:
    #     accArray_P.append(None)
    #     lossArray_P.append(None)
    h_Pi = model_Ci.fit(X, y, batch_size=10, epochs=10)
    accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
    lossArray_P.append(np.mean(h_Pi.history["loss"]))
    
    indices.append(i)

    accArray_Base.append(base_correct / len(X))
    lossArray_Base.append(None)
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "nist_drift_{0}_simple_{1}.npz".format(drift_type, freeze)
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
model_C0.save("model_C0_simple_{0}_{1}.h5".format(drift_type, freeze))
model_Ei.save("model_Ei_simple_{0}_{1}.h5".format(drift_type, freeze))
model_Ci.save("model_Pi_simple_{0}_{1}.h5".format(drift_type, freeze))

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


