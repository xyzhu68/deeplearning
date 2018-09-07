from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import matplotlib.pyplot as plt

# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

filepath_train = "train.arff"
filepath_test = "test.arff"

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define which classiefier shall be created: Ci or Ei")
    exit()
arg1 = sys.argv[1]
Ei = arg1 == "Ei"

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_data(dataPath):
    data, _ = arff.loadarff(dataPath)
    dataArray = np.asarray(data.tolist(), dtype=np.float)
    X = np.delete(dataArray, 784, 1) / 255
    X = X.reshape(-1, 28, 28, 1)
    y = dataArray[:,784]
    return (X, y)

def data_generator(streamSize): 
    X, y = load_data(filepath_train)
    count = 0
    while True:
        print("count: %d" % count)
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        count += streamSize
        yield X_result, y_result
        
        

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
    data_array = data_array.reshape(-1, 28, 28, 1)
    y_array = np.asarray(y_list)
    return (data_array, y_array)

def make_model(Ei):
    nb_filters = 64
    nb_conv = 3
    img_rows = 28
    img_cols = 28
    nb_pool = 2

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
    if Ei:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    else:
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    

    return model

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches
epochs = 5

# classifiers
model = make_model(Ei)

lossArray_E = []
accArray_E = []
lossArray = []
accArray = []
indices = []

# training in base phase
gen = data_generator(sizeOneBatch)
for i in range(nbBaseBatches):
    print(i)

    X, y = next(gen)
    if Ei:
        y = np.zeros(sizeOneBatch)
    else:
        y = to_categorical(y, 10)
    result = model.fit(X, y, batch_size=50, epochs=epochs)

    lossArray.append(np.mean(result.history["loss"]))
    accArray.append(np.mean(result.history["acc"]))
    indices.append(i)

# adaption: data changed
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X, y = next(gen)
    X, y = flip_images(X, y)
    # evaluate
    if Ei:
        y = np.full(sizeOneBatch, 1.0)
    else:
        y = to_categorical(y, 10)
    loss, acc = model.evaluate(X, y, batch_size=50)
    lossArray.append(loss)
    accArray.append(acc)
    indices.append(i)

    # training
    model.fit(X, y, batch_size=50, epochs=epochs)

result_file = "mnist_drift_clf_results_E.npz" if Ei else "mnist_drift_clf_results_P.npz"
np.savez(result_file, acc=accArray, loss=lossArray, indices=indices)


# result of accuracy
label = "acc error clf" if Ei else "acc patching clf"
plt.plot(indices, accArray, label=label)
#plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch")
plt.show()

# result of loss
label = "losss error clf" if Ei else "loss patching clf"
plt.plot(indices, lossArray, label=label)
#plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.show()