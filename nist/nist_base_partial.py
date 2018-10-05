import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.activations import relu, softmax
from keras import applications, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import sys
from keras import backend as K
from model_provider import *
from data_provider import *

nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define which classes used as base")
    exit()
base_classes = sys.argv[1]

# settings
img_size = 128
gen_batch_size = 700 # 20
epochs = 10 # 20 !!!

# Model
#model = make_resnet()
model = make_simple_model(False)


# prepare data
train_data_dir = os.path.abspath("../../by_class_2")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")


angle = 0
for i in range(70000 // gen_batch_size):
    print(i)
    X, y = next(train_generator)
    if base_classes == "0_9" or base_classes == "A_Z":
        X_used = []
        y_used = []
        for i in range(len(X)):
            yValue = np.argmax(y[i])
            if base_classes == "0_9":
                if yValue < 10:
                    X_used.append(X[i])
                    y_used.append(y[i])
            elif base_classes == "A_Z":
                if yValue >= 10:
                    X_used.append(X[i])
                    y_used.append(y[i])
        if len(X_used) == 0:
            continue
        if len(X_used) <gen_batch_size // 2:
            X2, y2 = next(train_generator)
            yValue = np.argmax(y2[i])
            if base_classes == "0_9":
                if yValue < 10:
                    X_used.append(X2[i])
                    y_used.append(y2[i])
            elif base_classes == "A_Z":
                if yValue >= 10:
                    X_used.append(X2[i])
                    y_used.append(y2[i])
            
        X_used = X_used[:gen_batch_size]
        y_used = y_used[:gen_batch_size]

        X = np.asarray(X_used)
        X = X.reshape(-1, 128, 128, 1)
        y = np.asarray(y_used)
        y = y.reshape(-1, 36)
    elif base_classes == "flip":
        X, y, _ = flip_images(X, y, True)
    elif base_classes == "rotate":
        angle += 5
        if angle > 180:
            angle = 5
        X, y, _ = rot(X, y, angle)
    else:
        print("Unknown argument: {0}".format(base_classes))
        exit()


    model.fit(X, y, batch_size=20, epochs=epochs)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=3500,
#     epochs=epochs)

fileName = "model_base_{0}.h5".format(base_classes)
model.save(fileName)

K.clear_session()  