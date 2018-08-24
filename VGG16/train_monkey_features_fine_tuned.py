import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 20
epochs = 50
img_width, img_height = 150, 150
monkey_VGG_path = "monkey_vgg_weighs.h5"
train_data_dir = os.path.abspath("../../training/training")
validation_data_dir = os.path.abspath("../../validation/validation")

# pre-trained VGG16 network
input_tensor = Input(shape=(img_width,img_height,3))
base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

fc_model = Sequential()
fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
fc_model.add(Dense(256, activation='relu'))
fc_model.add(Dropout(0.5))
fc_model.add(Dense(10, activation='softmax'))

fc_model.load_weights(monkey_VGG_path)

# add the model on the convolutional base
model = Model(input=base_model.input, output=fc_model(base_model.output))
for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=1040,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=260)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

