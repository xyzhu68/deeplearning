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
from keras import backend as K
from model_provider import *



# settings
img_size = 128
gen_batch_size = 20
epochs = 20

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

# X, y = train_generator.next()
# for i in range(len(X)):
#     x1 = X[i]
#     y1 = y[i]
#     print(y1)
#     plt.imshow(x1)
#     plt.show()
# exit()


model.fit_generator(
    train_generator,
    steps_per_epoch=3500,
    epochs=epochs)


model.save("model_base.h5")

K.clear_session()  