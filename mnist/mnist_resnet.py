from scipy.io import arff
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from resnet import Residual

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
input_var = Input(shape=(img_rows, img_cols, 1))

conv1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(input_var)
# conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
conv2 = Convolution2D(8, kernel_size[0], kernel_size[1],
                      border_mode='same', activation='relu')(conv1)

resnet = conv2
for _ in range(10):
    resnet = Residual(Convolution2D(8, kernel_size[0], kernel_size[1],
                                  border_mode='same'))(resnet)
    resnet = Activation('relu')(resnet)

mxpool = MaxPooling2D(pool_size=pool_size)(resnet)
flat = Flatten()(mxpool)
dropout = Dropout(0.5)(flat)
softmax = Dense(nb_classes, activation='softmax')(dropout)

model = Model(input=[input_var], output=[softmax])
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy']) 


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
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=20)
model.save("model_base_resnet.h5")
model.save_weights("model_base_resnet_weights.h5")

K.clear_session()                    