
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import applications, optimizers
from keras.layers.normalization import BatchNormalization
import sys
import os
import matplotlib.pyplot as plt
from keras.activations import relu, softmax, sigmoid
import datetime

from data_provider import *
from evolutionary import *


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

# if nbArgs < 3:
#     print("Please define the block number for engagement")
#     exit()
# blockNumber = int(sys.argv[2])



# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def calc_accuracy(modelC0, modelEi, modelCi, X, y):
#     predictEi = modelEi.predict(X)
#     index = 0
#     correct = 0
#     predict = None
#     for p in predictEi:
#         if p[0] > 0.5:
#             predict = modelCi.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
#         else:
#             predict = modelC0.predict(X[index].reshape(1, img_size, img_size, 3), batch_size=1)
#         if (np.argmax(predict) == np.argmax(y[index])):
#             correct += 1
#         index += 1
#     return correct / len(X)

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

# def make_vgg_model(Ei, img_size, nbClasses, layersToEngage):
#     input_tensor = Input(shape=(img_size,img_size,3))
#     base_model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

#     for layer in base_model.layers:
#         layer.trainable = False

#     for i in range(18-layersToEngage):
#         base_model.layers.pop()

#     nbFilter = base_model.layers[-1].output_shape[1:][2]
#     patch_model = Sequential()
    
#     # add conv-block in patching-network
#     patch_model.add(Conv2D(nbFilter, (3, 3), activation="relu", 
#                             input_shape=base_model.layers[-1].output_shape[1:]))

#     patch_model.add(MaxPooling2D(pool_size=(2, 2)))
#     # conv-block end
#     patch_model.add(Dropout(0.5))
#     patch_model.add(Flatten())
#     patch_model.add(Dense(256))
#     patch_model.add(BatchNormalization(momentum=0.9))
#     patch_model.add(Activation('relu'))
#     patch_model.add(Dropout(0.25))

#     if Ei:
#         patch_model.add(Dense(1, activation='sigmoid'))
#     else:
#         patch_model.add(Dense(nbClasses, activation='softmax'))

#     # add the model on the convolutional base
#     model = Model(inputs=base_model.input, outputs=patch_model(base_model.layers[-1].output))

#     if Ei:
#         model.compile(loss='binary_crossentropy',
#                         optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#                         metrics=['binary_accuracy'])
#     else:
#         model.compile(loss='categorical_crossentropy',
#                         optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#                         metrics=['categorical_accuracy'])

#     return model
                
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Run_one_engagement(drift_type):
    # settings
    img_size = 150
    gen_batch_size = 300
    epochs = 10
    nbClasses = 20
    bz = 20 # batch_size for fit & evaluate

    bundleSize = 1 #10 # we bundle batches

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
        batch_size=gen_batch_size * bundleSize,
        class_mode="categorical")

    model_C0 = make_C0_model(nbClasses)

    # training in base phase
    for i in range(nbBaseBatches//bundleSize):
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

        model_C0.fit(X, y, batch_size=bz, epochs = epochs, verbose=0)

    model_C0.save("C0_weigths_{0}.h5".format(drift_type))
    del model_C0


    # adaption: data changed
    angle = 0 # for rotate
    engageResults = []
    PAResults = []
    for b in range(nbBaseBatches//bundleSize, nbBatches//bundleSize):
        print("bundle number: {0}".format(b))
        i = b * bundleSize

        loopbegin = datetime.datetime.now()
        
        X_org, y_org = train_generator.next()
        
        X = None
        y = None
        if drift_type == "appear":
            X, y, _ = appear(X_org, y_org, False)
        elif drift_type == "remap":
            X, y, _ = remap(X_org, y_org, i < nbBatches/2)
        elif drift_type == "transfer":
            X, y, _ = transfer(X_org, y_org, i < nbBatches/2)

        results = Evolutionary(18, # total
                            2, # nbOfFitnessToCompare,
                            5, # initPopulation,
                            2, # nextPopulation,
                            #  model_C0,
                            #  models_P,
                            #  models_ms,
                            img_size,
                            b,
                            X,
                            y,
                            epochs,
                            drift_type,
                            "engage")
        engageResults.append(results)
        results = Evolutionary(10, 2, 3, 2, img_size, b, X, y, epochs, drift_type, "PA", results[0])
        PAResults.append(results)
   
        loopend = datetime.datetime.now()
        print("loop time: ", loopend - loopbegin)
        # break # !!!!!!!! TEST
        
    
    endTime = datetime.datetime.now()
    print(endTime - beginTime)

    fileName = "dm_best_layer_{0}.npz".format(drift_type)
    np.savez(fileName, engageResults = engageResults, PAResults = PAResults, duration=str(endTime - beginTime))


Run_one_engagement(drift_type)