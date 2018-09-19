from scipy.io import arff
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.activations import relu, softmax


from model_provider import *

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
filepath_test = "test.arff"


nb_filters = 64 # 32
kernel_size = (3, 3)
pool_size = (2, 2)
img_rows = 28
img_cols = 28
batch_size = 100
nb_classes = 10

# Model
model_resnet = make_resnet()
input = Input(shape=(28,28,1))
out = model_resnet(input)
out = Flatten()(out)
out = Dense(units=128)(out)
out = Activation(relu)(out)
out = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(out)
out = Activation(softmax)(out)
model = Model(inputs=input, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_resnet.summary()
# model.summary()
# exit()

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

#model.fit(X_50K, y_50K, batch_size=100, epochs=20)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=20)
model.save("model_base_resnet.h5")
model_resnet.save_weights("model_base_resnet_weights.h5")

K.clear_session()                    