import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 20
epochs = 50
monkey_VGG_path = "monkey_vgg_weighs.h5"

train_data = np.load("monkey_feature_train.npy")
train_labels = np.array([])
for i in range(10):
    labels = [i] * 104
    train_labels = np.append(train_labels, labels)

train_labels = to_categorical(train_labels)

validation_data = np.load("monkey_feature_validation.npy")
validation_labels = np.array([])
for i in range(10):
    labels = [i] * 26
    validation_labels = np.append(validation_labels, labels)

validation_labels = to_categorical(validation_labels)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))
model.save_weights(monkey_VGG_path)

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