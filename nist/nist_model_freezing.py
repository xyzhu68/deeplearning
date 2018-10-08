from scipy.io import arff
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import sys
import os
from data_provider import *
from model_provider import *

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

freeze_no = 0
if nbArgs < 3:
    print("please define number of freezing layers")
    exit()
freeze_no = int(sys.argv[2])

model = None
if drift_type == "flip" or drift_type == "rotate":
    model = load_model("model_base.h5")
elif drift_type == "appear":
    model = load_model("model_base_A_Z.h5")
elif drift_type == "remap" or drift_type == "transfer":
    model = load_model("model_base_0_9.h5")

# freeze layers
dict = {1: ["layer1"],
        2: ["layer1", "layer2"],
        3: ["layer1", "layer2", "layer5"] }
layersToFreeze = dict[freeze_no]
for layer in model.layers:
    if layer.name in layersToFreeze:
        layer.trainable = False

# prepare data
img_size = 128
gen_batch_size = 700
epochs = 10
train_data_dir = os.path.abspath("../../by_class_2")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

angle = 0
accArray = []
lossArray = []
indices = []
nbBatches = 70000 // gen_batch_size
if drift_type == "appear":
    nbBatches = nbBatches // 5 # use 80% of data
else:
    nbBatches = nbBatches // 2 # use 50% of data
for i in range(nbBatches):
    print(i)
    X, y = next(train_generator)

    if drift_type == "flip":
        X, y, _ = flip_images(X, y, True)
    elif drift_type == "rotate":
        angle += 5
        if angle > 180:
            angle = 5
        X, y, _ = rot(X, y, angle)
    elif drift_type == "appear":
        pass
    elif drift_type == "remap":
        X, y, _ = remap(X, y, False)
    elif drift_type == "transfer":
        X, y, _ = transfer(X, y, False)
    else:
        print("unkown drift type")
        exit()

    history = model.fit(X, y, batch_size=20, epochs=epochs)
    lossArray.append(np.mean(history.history["loss"]))
    accArray.append (np.mean(history.history["categorical_accuracy"]))
    indices.append(i)

fileName = "model_{0}_freeze_{1}".format(drift_type, freeze_no)
model.save(fileName + ".h5")
np.savez(fileName + ".npz", acc=accArray, loss=lossArray, indices=indices)