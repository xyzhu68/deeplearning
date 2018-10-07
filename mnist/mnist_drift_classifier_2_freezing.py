from scipy.io import arff
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import sys
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime

from data_provider import *
from model_provider import *

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
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

resnet = True

freeze_add_block = 0
if nbArgs > 2:
    freeze_add_block = int(sys.argv[2])

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
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        count += streamSize
        print("count: %d" % count)
        yield X_result, y_result
        
def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 28, 28, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
            if p[0] > 0.5:
                tp += 1
            else:
                tn += 1
        else:
            if p[0] > 0.5:
                fp += 1
            else:
                fn += 1
        index += 1
    return correct / len(X), tp, tn, fp, fn

# model from scratch
def make_model(Ei):
    nb_filters = 64
    nb_conv = 3
    img_rows = 28
    img_cols = 28
    nb_pool = 2

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    name="layer1",
                    padding='valid',
                    input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', name="layer2"))
    model.add(Activation('relu', name="layer3"))
    model.add(MaxPooling2D(name="layer4", pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer5"))
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer6"))
    model.add(Activation('relu', name="layer7"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name="layer8"))
    model.add(Dropout(0.25, name="layer9"))


    model.add(Flatten(name="Flatten"))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if Ei:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
    else:
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    

    return model

def make_resnet_model(Ei, n):
    print("resnet used")
    model_resnet = make_resnet()
    model_resnet.load_weights("model_base_resnet_weights.h5")
    
    count_add = 0
    for l in model_resnet.layers:
        if l.name.startswith("add"):
            count_add += 1
        if count_add < n and not l.name.startswith("input_"):
            l.trainable = False
            print("Frozen layer {0}".format(l.name))
        else:
            print("Free layer {0}".format(l.name))
                
    
    input = Input(shape=(28,28,1))
    out = model_resnet(input)
    out = Flatten()(out)
    out = Dense(units=128)(out)
    out = Activation(relu)(out)
    if Ei:
        out = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(out)
        out = Activation(sigmoid)(out)
    else:
        out = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(out)
        out = Activation(softmax)(out)
    model = Model(inputs=input, outputs=out)
    lossfunc = "binary_crossentropy" if Ei else "categorical_crossentropy"
    met = "binary_accuracy" if Ei else "categorical_accuracy"
    model.compile(loss=lossfunc, optimizer='adam', metrics=[met])
    
    return model

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
# nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

# build model
model = None
if resnet:
    if drift_type == "appear" or drift_type == "transfer" or drift_type == "remap:
        model = load_model("model_base_resnet_digit5_9.h5")
    else:
        model = load_model("model_base_resnet.h5")

lossArray = []
accArray = []
indices = []

# get data
gen = data_generator(sizeOneBatch)
angle = 0 # for rotate
if drift_type == "appear":
    nbBatches = 80
else:
    nbBatches = 50
for i in range(nbBatches):
    print(i)
    X, y = next(gen)

    if drift_type == "flip":
        X, y, _ = flip_images(X, y, True)
    elif drift_type == "appear":
        X, y, _ = appear(X, y, False)
    elif drift_type == "remap":
        X, y, _ = remap(X, y, False)
    elif drift_type == "rotate":
        if i > 50 and i < 85 and angle <= 180:
            angle += 5
        else:
            angle = 0
        X, y, _ = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        X, y, _ = transfer(X, y, False)
    else:
        print(drift_type + " is unknown")
        exit()

    print(X.shape)
    print(len(X))

    result = model.fit(X, y_E, batch_size=50, epochs=10)
    lossArray.append(np.mean(result.history["loss"]))
    accArray.append(np.mean (result.history["binary_accuracy"]))
    indices.append(i)



endTime = datetime.datetime.now()
print(endTime - beginTime)

fileName = "mnist_{0}_resnet_{1}_freezing".format(drift_type, freeze_add_block)
np.savez(fileName + ".npz", 
         acc=accArray, 
         loss=lossArray,
         indices=indices,
         duration=str(endTime - beginTime))
model.save(fileName + ".h5")

# result of accuracy
# plt.plot(indices, accArray, label="acc patching clf")
# plt.plot(indices, accArray_E, label="acc error clf")
# plt.plot(indices, accChainedArray, label="acc Ei+Ci")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Batch")
# plt.legend()
# plt.show()
# pic_file_name = ""
# if resnet:
#     pic_file_name = "accuracy_{0}_resnet_{1}.png".format(drift_type, freeze_add_block)
# else:
#     pic_file_name = "accuracy_{0}_fs.png".format(drift_type)

# plt.savefig(pic_file_name)

