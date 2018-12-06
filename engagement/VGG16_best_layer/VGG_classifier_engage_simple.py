
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import applications, optimizers
import sys
import os
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime

from data_provider import *


# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# #check arguments
# nbArgs = len(sys.argv)
# if nbArgs < 2:
#     print("Please define drift type")
#     exit()
# drift_type = sys.argv[1]

# if nbArgs < 3:
#     print("Please define the block number for engagement")
#     exit()
# blockNumber = int(sys.argv[2])



# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def calc_accuracy(modelC0, modelEi, modelCi, X, y):
    predictEi = modelEi.predict(X)
    index = 0
    correct = 0
    predict = None
    for p in predictEi:
        if p[0] > 0.5:
            predict = modelCi.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
        else:
            predict = modelC0.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
        if (np.argmax(predict) == np.argmax(y[index])):
            correct += 1
        index += 1
    return correct / len(X)

def make_C0_model(nbClasses):
    input_tensor = Input(shape=(img_size,img_size,3))
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    fc_model.add(Dense(256, activation='relu'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(nbClasses, activation='softmax'))

    # add the model on the convolutional base
    model = Model(inputs=base_model.input, outputs=fc_model(base_model.output))
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['categorical_accuracy'])
        

    return model

def make_vgg_model(Ei, img_size, nbClasses, layersToEngage):
    input_tensor = Input(shape=(img_size,img_size,3))
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = False
    # popDict = {1: 15, 2: 12, 3: 8, 4: 4, 5: 0}
    # layersToPop = popDict[blockNumber]
    for i in range(layersToEngage):
        base_model.layers.pop()

    nbFilter = base_model.layers[-1].output_shape[1:][2]
    patch_model = Sequential()
    # add conv-block in patching-network
    patch_model.add(Conv2D(nbFilter, (3, 3), activation="relu", 
                            input_shape=base_model.layers[-1].output_shape[1:]))

    patch_model.add(MaxPooling2D(pool_size=(2, 2)))
    # conv-block end
    patch_model.add(Flatten())
    patch_model.add(Dense(256, activation='relu'))
    patch_model.add(Dropout(0.5))
    if Ei:
        patch_model.add(Dense(1, activation='sigmoid'))
    else:
        patch_model.add(Dense(nbClasses, activation='softmax'))

    # add the model on the convolutional base
    model = Model(inputs=base_model.input, outputs=patch_model(base_model.layers[-1].output))

    if Ei:
        model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['binary_accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                        metrics=['categorical_accuracy'])

    return model
                
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Run_one_engagement(drift_type, layersToEngage):
    # settings
    img_size = 150
    gen_batch_size = 300
    epochs = 10
    nbClasses = 20
    bz = 20 # batch_size for fit & evaluate

    nbBatches = 100
    nbBaseBatches = nbBatches // 5 # size of base dataset: 20% of total batches
    beginTime = datetime.datetime.now()

    # prepare data
    train_data_dir = os.path.abspath("../../data/dog_monkey") 
    train_datagen = ImageDataGenerator(rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_size, img_size),
        batch_size=gen_batch_size,
        class_mode="categorical")

    model_C0 = make_C0_model(nbClasses)

    # accArray_E = []
    accArray_MS = []
    accArray_P = []
    indices = []
    # accArray_Base = []
    # accEiPi = []
    accMSPi = []


    # training in base phase
    for i in range(nbBaseBatches):
    #for i in range(1): # !!!!!!!!!!!!!!!!
        print(i)
        X, y = train_generator.next()
        
        if drift_type == "appear":
            X, y, _ = appear(X, y, True)
        elif drift_type == "remap":
            X, y, _ = remap(X, y, True)
        elif drift_type == "transfer":
            X, y, _ = transfer(X, y, True)
        else:
            print("Unknown drift type")
            exit()

        model_C0.fit(X, y, batch_size=bz, epochs = epochs)


    model_P = make_vgg_model(False, img_size, nbClasses, layersToEngage)
    # model_E = make_vgg_model(True, img_size, nbClasses, layersToEngage)
    model_ms = make_vgg_model(True, img_size, nbClasses, layersToEngage)

    # adaption: data changed
    angle = 0 # for rotate
    for i in range(nbBaseBatches, nbBatches):
        print("batch number {0}".format(i))
        
        X_org, y_org = train_generator.next()
        
        X = None
        y = None
        if drift_type == "appear":
            X, y, _ = appear(X_org, y_org, False)
        elif drift_type == "remap":
            X, y, _ = remap(X_org, y_org, i < nbBatches/2)
        elif drift_type == "transfer":
            X, y, _ = transfer(X_org, y_org, i < nbBatches/2)

        # accEiPi.append(calc_accuracy(model_C0, model_E, model_P, X, y))
        accMSPi.append(calc_accuracy(model_C0, model_ms, model_P, X, y))
        # x_Ei = []
        # y_Ei = []
        x_ms = []
        y_ms = []
        # base_correct = 0
        for index in range(len(X)):
            predictC0 = model_C0.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
            predictP = model_P.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
            # if np.argmax(predictC0) == np.argmax(y[index]):
            #     x_Ei.append(X[index])
            #     y_Ei.append(0)
            #     base_correct += 1
            # else:
            #     x_Ei.append(X[index])
            #     y_Ei.append(1)

            yLabel = np.argmax(y[index])
            if predictC0[0][yLabel] > predictP[0][yLabel]:
                x_ms.append(X[index])
                y_ms.append(0)
            else:
                x_ms.append(X[index])
                y_ms.append(1)


        # x_Ei = np.asarray(x_Ei)
        # x_Ei = x_Ei.reshape(-1, img_size, img_size, 3)
        # loss_Ei, acc_Ei = model_E.evaluate(x_Ei, y_Ei, batch_size=bz)
        # accArray_E.append(acc_Ei)
        # model_E.fit(x_Ei, y_Ei, batch_size = bz, epochs = epochs)

        x_ms = np.asarray(x_ms)
        x_ms = x_ms.reshape(-1, img_size, img_size, 3)
        loss_ms, acc_ms = model_ms.evaluate(x_ms, y_ms, batch_size=bz)
        accArray_MS.append(acc_ms)
        model_ms.fit(x_ms, y_ms, batch_size = bz, epochs = epochs)

        loss_Pi, acc_Pi = model_P.evaluate(X, y, batch_size=bz)
        accArray_P.append(acc_Pi)
        model_P.fit(X, y, batch_size=bz, epochs=epochs)

        
        indices.append(i)

        # accArray_Base.append(base_correct / len(X))
        
    
    endTime = datetime.datetime.now()
    print(endTime - beginTime)

    npFileName = "vgg_{0}_{1}_simple.npz".format(drift_type, layersToEngage)
    np.savez(npFileName, #  accE = accArray_E,
                        accP = accArray_P,
                        accMS = accArray_MS,
                        #  accEiPi = accEiPi,
                        accMSPi = accMSPi,
                        indices=indices,
                        duration=str(endTime - beginTime))

    finalAcc = np.mean(accMSPi[-5:])
    return finalAcc


