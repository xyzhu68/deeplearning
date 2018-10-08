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
import os

from data_provider import *
from model_provider import *

# settings for GPU
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

# settings
img_size = 128
gen_batch_size = 700 #20
epochs = 20

# prepare data
train_data_dir = os.path.abspath("../../by_class_2")
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=gen_batch_size,
    color_mode="grayscale",
    class_mode="categorical")

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

        
def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, 128, 128, 1), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, 128, 128, 1), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)



#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

beginTime = datetime.datetime.now()

# settings
totalDataSize = 70000
sizeOneBatch = 700
nbBatches = totalDataSize // sizeOneBatch # devide dataset into batches
nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches


# C0
model = None
model_Ci = None
if drift_type == "flip":
    model = load_model('model_base.h5')
    model_Ci = load_model("model_base_flip.h5")
elif drift_type == "rotate":
    model = load_model('model_base.h5')
    model_Ci = load_model("model_base_rotate.h5")
elif drift_type == "appear":
    model = load_model("model_base_A_Z.h5")
    model_Ci = load_model("model_base_0_9.h5")
elif drift_type == "remap" or drift_type == "transfer":
    model = load_model("model_base_0_9.h5")
    model_Ci = load_model("model_base_A_Z.h5")

model_Ei = make_simple_model(True)



lossArray_E = []
accArray_E = []
lossArray = []
accArray = []
indices = []
accChainedArray = []

# training in base phase
for i in range(nbBaseBatches):
    print(i)
    
    X, y = train_generator.next()

    y_E = None
    if drift_type == "flip":
        X, y, y_E = flip_images(X, y, False)
    elif drift_type == "appear":
        X, y, y_E = appear(X, y, True)
    elif drift_type == "remap":
        X, y, y_E = remap(X, y, True)
    elif drift_type == "rotate":
        X, y, y_E = rot(X, y, 0)
    elif drift_type == "transfer":
        X, y, y_E = transfer(X, y, True)
    else:
        print(drift_type + " is unknown")
        exit()

    print(X.shape)
    print(len(X))
    print(len(y_E))

    result_E = model_Ei.fit(X, y_E, batch_size=10, epochs=10)
    # result_C = model_Ci.fit(X, y, batch_size=20, epochs=10)

    # lossArray.append(np.mean(result_C.history["loss"]))
    # accArray.append(np.mean(result_C.history["categorical_accuracy"]))
    lossArray_E.append(np.mean(result_E.history["loss"]))
    accArray_E.append(np.mean(result_E.history["binary_accuracy"]))
    indices.append(i)
    accChained = calc_accuracy(model, model_Ei, model_Ci, X, y)
    accChainedArray.append(accChained)

# adaption: data changed
angle = 0 # for rotate
for i in range(nbBaseBatches, nbBatches):
    print(i)
    
    X_org, y_org = train_generator.next()
    
    data_changed = True
    X = None
    y = None
    y_E = None
    if drift_type == "flip":
        X, y, y_E = flip_images(X_org, y_org, i >= nbBatches/2)
        data_changed = i >= nbBatches/2
    elif drift_type == "appear":
        X, y, y_E = appear(X_org, y_org, False)
    elif drift_type == "remap":
        X, y, y_E = remap(X_org, y_org, i < nbBatches/2)
        data_changed = i >= nbBatches/2
    elif (drift_type == "rotate"):
        if i > nbBatches // 2:
            angle += 5
            if angle > 180:
                angle = 180
        else:
            angle = 0
            data_changed = False
        X, y, y_E = rot(X_org, y_org, angle)
    elif drift_type == "transfer":
        X, y, y_E = transfer(X_org, y_org, i < nbBatches/2)
        data_changed = i >= nbBatches/2

    X_combine = None
    y_combine = None
    if data_changed: # for Ei: combine original data and changed data and shuffle them to get better result
        X_combine, y_combine = combine_Ei_training_data(drift_type, X_org, y_org, X, y_E)
    else:
        X_combine = X
        y_combine = y_E
        
    # evaluate
    loss_E, acc_E = model_Ei.evaluate(X_combine, y_combine, batch_size=10)
    print("acc_E: {0}, loss: {1}".format(acc_E, loss_E))
    lossArray_E.append(loss_E)
    accArray_E.append(acc_E)
    # loss, acc = model_Ci.evaluate(X, y, batch_size=20)
    # lossArray.append(loss)
    # accArray.append(acc)
    indices.append(i)
    accChained = calc_accuracy(model, model_Ei, model_Ci, X, y)
    accChainedArray.append(accChained)

    # training
    model_Ei.fit(X_combine, y_combine, batch_size=10, epochs=10)
    # model_Ci.fit(X, y, batch_size=20, epochs=10)

endTime = datetime.datetime.now()
print(endTime - beginTime)

npFileName = "nist_drift_{0}_from_scratch_64.npz".format(drift_type)
# if resnet:
#     if freeze_add_block >= 0:
#         npFileName = "mnist_drift_{0}_resnet_{1}.npz".format(drift_type, freeze_add_block)
#     else:
#         npFileName = "mnist_drift_{0}_resnet_fs.npz".format(drift_type)
np.savez(npFileName, acc_E=accArray_E, 
                    loss_E=lossArray_E,
                    accChained=accChainedArray,
                    indices=indices,
                    duration=str(endTime - beginTime))

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

