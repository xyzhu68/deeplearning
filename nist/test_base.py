import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
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

model = load_model("model_base_0_9.h5")

img_size = 128
gen_batch_size = 20
epochs = 20

train_data_dir = os.path.abspath("../../by_class_2")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

# X, y = train_generator.next()
# for i in range(len(X)):
#     print(np.argmax(y[i]))
#     plt.imshow(X[i].reshape(128, 128))
#     plt.show()

# exit()

for i in range(10):
    X, y = train_generator.next()
    X_used = []
    y_used = []
    for j in range(len(X)):
        yValue = np.argmax(y[j])
        if yValue < 10:
            X_used.append(X[j])
            y_used.append(y[j])
            
    X = np.asarray(X_used)
    X = X.reshape(-1, 128, 128, 1)
    y = np.asarray(y_used)
    y = y.reshape(-1, 36)
    l, a = model.evaluate(X, y, batch_size=10)
    print(a)


