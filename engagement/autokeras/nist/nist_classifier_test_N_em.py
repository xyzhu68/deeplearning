from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, backend as K
import sys
import os
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime
import random

from beta_distribution_drift_detector.bdddc import BDDDC

from data_provider import *
from model_provider import *


# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def make_simple_model(Ei, suffix):
    nb_filters = 64
    nb_conv = 3
    img_rows = 128
    img_cols = 128
    nb_pool = 4

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    name="layer1",
                    padding='valid',
                    input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', name="layer2"))
    model.add(Activation('relu', name="layer3"))
    model.add(MaxPooling2D(name="layer4", pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer5"))
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer6"))
    model.add(Activation('relu', name="layer7"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name="layer8"))
    model.add(Dropout(0.25, name="layer9"))


    model.add(Flatten(name="Flatten"+suffix))
    model.add(Dense(128, name="dense1"+suffix))
    model.add(Activation('relu', name="act"+suffix))
    model.add(Dropout(0.5, name="dropout"+suffix))
    if Ei:
        model.add(Dense(1, name="dropout2"+suffix))
        model.add(Activation('sigmoid', name="sigmoid"+suffix))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
    else:
        model.add(Dense(36, name="dense"+suffix))
        model.add(Activation('softmax', name="softmax"+suffix))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    

    return model

def is_drift(C0_model, detector, X, y):
    pred = C0_model.predict(X)
    pred_cls = np.argmax(pred, axis=1)
    y_cls = np.argmax(y, axis=1)
    
    detector.add_element(pred_cls, y_cls, classifier_changed=False)

    return detector.detected_change()


#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

searchEi = False
if nbArgs > 2:
    if sys.argv[2] == "Ei":
        searchEi = True

img_size = 128
nbFilters = 16

beginTime = datetime.datetime.now()

# settings
gen_batch_size = 720
epochs = 10

totalDataSize = 72000
sizeOneBatch = 720
nbBatches = totalDataSize // sizeOneBatch # devide dataset into batches
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches

changePoint = 50
if drift_type == "appear":
    nbBaseBatches = 30
    changePoint = 30

# prepare data
train_data_dir = os.path.abspath("../../../../NIST")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

# model_C0 = make_conv_model(nbFilters, False)
model_C0 = make_simple_model(False, "")
model_C0.load_weights("C0_weigths_simple_flip_5.h5")

bdddc = BDDDC( warn_level=0.99, drift_level=0.9997) #0.99, 0.999


angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print("batch number: {0}".format(i))
    
    X_org, y_org = train_generator.next()
    
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
    # changed = is_drift(model_C0, bdddc, X, y)

    print("changed: ", changed)


   
endTime = datetime.datetime.now()
print(endTime - beginTime)




