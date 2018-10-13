
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import applications, optimizers
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

# settings
img_size = 150
gen_batch_size = 200
epochs = 20
nbClasses = 20

# totalDataSize = 70000
# sizeOneBatch = 700
nbBatches = 100
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)

def make_vgg_model(Ei):
    input_tensor = Input(shape=(img_size,img_size,3))
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    fc_model.add(Dense(256, activation='relu'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(nbClasses, activation='softmax'))

    # add the model on the convolutional base
    model = Model(input=base_model.input, output=fc_model(base_model.output))
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['categorical_accuracy'])
                
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# prepare data
train_data_dir = os.path.abspath("../../../dogs") # Anpassen !!!!!!!!!!!!!!!!!
train_datagen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    class_mode="categorical")

model_C0 = make_vgg_model(False)

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

    history = model_C0.fit(X, y, batch_size=10, epochs = epochs)
    accArray_Base.append(np.mean(history.history["categorical_accuracy"]))
    lossArray_Base.append(np.mean(history.history["loss"]))
    indices.append(i)

    lossArray_E.append(None)
    accArray_E.append(None)
    lossArray_P.append(None)
    accArray_P.append(None)
    accEiPi.append(None)


model_Ci = make_vgg_model(False)

model_Ei = make_vgg_model(True)

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
    for index in range(len(X)):
        predict = model_C0.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
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
        x_Ei = x_Ei.reshape(-1, img_size, img_size, 3)
        h_Ei = model_Ei.fit(x_Ei, y_Ei, batch_size = 10, epochs = epochs)
        accArray_E.append(np.mean(h_Ei.history["binary_accuracy"]))
        lossArray_E.append(np.mean(h_Ei.history["loss"]))
    else:
        accArray_E.append(None)
        lossArray_E.append(None)

    h_Pi = model_Ci.fit(X, y, batch_size=10, epochs=epochs)
    accArray_P.append(np.mean(h_Pi.history["categorical_accuracy"]))
    lossArray_P.append(np.mean(h_Pi.history["loss"]))
    
    indices.append(i)

    accArray_Base.append(None)
    lossArray_Base.append(None)
    
   
endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "vgg_drift_{0}.npz".format(drift_type)
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
model_C0.save("model_C0_vgg_{0}.h5".format(drift_type))
model_Ei.save("model_Ei_vgg_{0}.h5".format(drift_type))
model_Ci.save("model_Pi_vgg_{0}.h5".format(drift_type))

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


