import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

img_width, img_height = 150, 150
train_data_dir = os.path.abspath("../../training/training")
validation_data_dir = os.path.abspath("../../validation/validation")
batch_size = 20

datagen = ImageDataGenerator(rescale=1. / 255)

# pre-trained VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

train_features = model.predict_generator(generator, 1040 // batch_size) 
print("shape train: ", train_features.shape)
np.save("monkey_feature_train.npy", train_features)


generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

validation_features = model.predict_generator(generator, 260 // batch_size)
print("shape val: ", validation_features.shape)
np.save("monkey_feature_validation.npy", validation_features)
