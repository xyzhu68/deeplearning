from scipy.io import arff
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
filepath_test = "test.arff"


nb_filters = 64 # 32
nb_conv = 3
img_rows = 28
img_cols = 28
nb_pool = 2
batch_size = 50

model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                 padding='valid',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid'))
model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten(name="Flatten"))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  

# read data
data, meta = arff.loadarff(filepath_train)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_train = np.delete(dataArray, 784, 1) / 255
X_train = X_train.reshape(-1, img_rows, img_cols, 1)
y_train = dataArray[:,784]
y_train = to_categorical(y_train, 10)

test_data, _ = arff.loadarff(filepath_test)
testdataarray = np.asarray(test_data.tolist(), dtype=np.float)
X_test = np.delete(testdataarray, 784, 1) / 255
X_test = X_test.reshape(-1, img_rows, img_cols, 1)
y_test = testdataarray[:, 784]
y_test = to_categorical(y_test, 10)

#X_50K = X_train[:50000]
#y_50K = y_train[:50000]

#model.fit(X_50K, y_50K, batch_size=100, epochs=20)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=20)
model.save("model_base.h5")

K.clear_session()                    