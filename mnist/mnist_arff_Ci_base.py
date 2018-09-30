from scipy.io import arff
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import sys
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

filepath_train = "train.arff"
filepath_test = "test.arff"


# nb_filters = 64 # 32
# nb_conv = 3
img_rows = 28
img_cols = 28
# nb_pool = 2


# model = Sequential()
# model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
#                  name="layer1",
#                  padding='valid',
#                  input_shape=(img_rows, img_cols, 1)))
# model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', name="layer2"))
# model.add(Activation('relu', name="layer3"))
# model.add(MaxPooling2D(name="layer4", pool_size=(nb_pool, nb_pool)))

# model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer5"))
# model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer6"))
# model.add(Activation('relu', name="layer7"))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name="layer8"))
# model.add(Dropout(0.25, name="layer9"))

# model.add(Flatten(name="Flatten"))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  

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

# read data
data, meta = arff.loadarff(filepath_train)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_train = np.delete(dataArray, 784, 1) / 255
X_train = X_train.reshape(-1, img_rows, img_cols, 1)
y_train = dataArray[:,784]
#y_train = to_categorical(y_train, 10)

if drift_type == "digit0_4":
    x0_4 = []
    y0_4 = []
    for i in range(len(X_train)):
        y = y_train[i]
        if y < 5:
            y0_4.append(y)
            x0_4.extend(X_train[i])
    y_train = to_categorical(y0_4, 10)
    X_train = np.asarray(x0_4).reshape(-1, 28, 28, 1)
elif drift_type == "digit5_9":
    x5_9 = []
    y5_9 = []
    for i in range(len(X_train)):
        y = y_train[i]
        if y > 4:
            y5_9.append(y)
            x5_9.extend(X_train[i])
    y_train = to_categorical(y5_9, 10)
    X_train = np.asarray(x5_9).reshape(-1, 28, 28, 1)
elif drift_type == "flip":
    X_train, y_train, _ = flip_images(X_train, y_train, True)
elif drift_type == "rotate":
    angle = 0
    rotated_X = []
    rotated_y = []
    begin = 0
    batchSize = 100
    for i in range(batchSize):
        angle += 5
        if (angle > 180):
            angle = 5
        stream_size = len(X_train) // batchSize
        X_rotated = X_train[begin : begin + stream_size]
        y = y_train[begin : begin + stream_size]
        x, y, _ = rot(X_rotated, y, angle)
        rotated_X.extend(x)
        rotated_y.extend(y)
        begin += stream_size
        print(i)
    X_train = np.asarray(rotated_X).reshape(-1, 28, 28, 1)
    y_train = np.asarray(rotated_y).reshape(-1, 10)
elif drift_type == "remap":
    X_train, y_train, _ = remap(X_train, y_train, False)
else:
    print("unkown drift type")
    exit()

"""
test_data, _ = arff.loadarff(filepath_test)
testdataarray = np.asarray(test_data.tolist(), dtype=np.float)
X_test = np.delete(testdataarray, 784, 1) / 255
X_test = X_test.reshape(-1, img_rows, img_cols, 1)
y_test = testdataarray[:, 784]
y_test = to_categorical(y_test, 10)
"""


model.fit(X_train, y_train, batch_size=100, epochs=20)
model.save("model_base_resnet_{0}.h5".format(drift_type))
#model.save_weights("model_base_weights.h5")

K.clear_session()                    