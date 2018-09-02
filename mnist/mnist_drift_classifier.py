from scipy.io import arff
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define which classiefier shall be created: Ci or Ei")
    exit()
arg1 = sys.argv[1]


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
filepath_test = "test.arff"


"""
# read data
data, meta = arff.loadarff(filepath_train)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_train = np.delete(dataArray, 784, 1) / 255
X_train = X_train.reshape(-1, 28, 28, 1)
y_train = dataArray[:,784]


X_train = X_train[50000:]
y_train = y_train[50000:]

# exchange 2 and 9
y_train_Ei = [1 if x == 2 or x == 9 else 0 for x in y_train]

X_train_Ci = []
y_train_Ci = []
for x, y in zip(X_train, y_train):
    if y == 2:
        X_train_Ci.append(x)
        y_train_Ci.append(9)
    elif y == 9:
        X_train_Ci.append(x)
        y_train_Ci.append(2)
        
X_train_Ci = np.asarray(X_train_Ci)
y_train_Ci = to_categorical(y_train_Ci, 10)
"""


def load_data(dataPath):
    data, meta = arff.loadarff(dataPath)
    dataArray = np.asarray(data.tolist(), dtype=np.float32)
    X = np.delete(dataArray, 784, 1) / 255
    X = X.reshape(-1, 28, 28, 1)
    y = dataArray[:,784]
    return (X, y)

def flip_images(X, y):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    datagen.fit(X)
    data_it = datagen.flow(X, y, batch_size=1)
    
    data_list = []
    y_list = []
    batch_index = 0
    while batch_index <= data_it.batch_index:
        data = data_it.next()
        #x_data = data[0].reshape((784,))
        data_list.append(data[0])
        y_list.append(data[1])
        batch_index = batch_index + 1

    data_array = np.asarray(data_list)
    y_array = np.asarray(y_list)
    return (data_array, y_array)


X_train, y_train = load_data(filepath_train)
X_train, y_train = flip_images(X_train, y_train)
X_train = X_train.reshape(-1, 28, 28, 1)
print(X_train.shape)
print(y_train.shape)

X_test, y_test = load_data(filepath_test)
X_test, y_test = flip_images(X_test, y_test)
X_test = X_test.reshape(-1, 28, 28, 1)

trainingImagesCount = len(X_train)
testingImagesCount = len(X_test)
print(trainingImagesCount)
print(testingImagesCount)

if arg1 == "Ei":
    # get data from original data
    y_train = np.full(trainingImagesCount, 1.0)
    y_test = np.full(testingImagesCount, 1.0)
    X_train_origin, y_train_origin = load_data(filepath_train)
    y_train_origin = np.zeros(trainingImagesCount)
    X_test_origin, y_test_origin = load_data(filepath_test)
    y_test_origin = np.zeros(testingImagesCount)
    # concatenate original and flipped data
    X_train = np.concatenate((X_train, X_train_origin))
    y_train = np.concatenate((y_train, y_train_origin))
    y_train = y_train.reshape(-1, 1)
    X_test = np.concatenate((X_test, X_test_origin))
    y_test = np.concatenate((y_test, y_test_origin))
    y_test = y_test.reshape(-1, 1)
    print("after concate")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

model = load_model('model_base.h5')
digit_input = model.input
out_flatten = model.get_layer("Flatten")
visual_model = Model(digit_input, out_flatten.output)

class_input = Input(shape=(28,28,1))
out = visual_model(class_input)
out = Dense(128, activation="relu")(out)
out = Dropout(0.5)(out)

if arg1 == "Ei":
    out_Ei = Dense(1, activation="sigmoid")(out)
    model_Ei = Model(class_input, out_Ei)
    model_Ei.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model_Ei.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=5)
    #model_Ei.save("Ei.h5")
else:
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    out_Ci = Dense(10, activation="softmax")(out)
    model_Ci = Model(class_input, out_Ci)
    model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model_Ci.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=5)
    #model_Ci.save("Ci.h5")
