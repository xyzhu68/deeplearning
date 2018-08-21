from scipy.io import arff
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
filepath_test = "test.arff"



# read data
data, meta = arff.loadarff(filepath_train)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_train = np.delete(dataArray, 784, 1) / 255
X_train = X_train.reshape(-1, 28, 28, 1)
y_train = dataArray[:,784]

X_train = X_train[50000:]
y_train = y_train[50000:]

y_train_Ci = to_categorical(y_train, 10)


# augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,)

datagen.fit(X_train)


model = load_model('model_base.h5')
digit_input = model.input
out_flatten = model.get_layer("Flatten")
visual_model = Model(digit_input, out_flatten.output)

class_input = Input(shape=(28,28,1))
out = visual_model(class_input)
out = Dense(128, activation="relu")(out)
out = Dropout(0.5)(out)


out_Ci = Dense(10, activation="softmax")(out)
model_Ci = Model(class_input, out_Ci)
model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model_Ci.fit(X_train, y_train_Ci, batch_size=100, epochs=20)
model_Ci.fit_generator(datagen.flow(X_train, y_train_Ci, batch_size=100), steps_per_epoch=len(X_train)/100, epochs=20)
model_Ci.save("Ci_Aug.h5")
