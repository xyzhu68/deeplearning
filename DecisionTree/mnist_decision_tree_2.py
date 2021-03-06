from scipy.io import arff
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
import sys
from vfdt import *
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

filepath_train = Path("../mnist/train.arff")
filepath_test = Path("../mnist/test.arff")

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_data(dataPath):
    data, meta = arff.loadarff(dataPath)
    dataArray = np.asarray(data.tolist(), dtype=np.float)
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
        x_data = data[0].reshape((784,))
        data_list.append(x_data)
        y_list.append(data[1])
        batch_index = batch_index + 1

    data_array = np.asarray(data_list)
    y_array = np.asarray(y_list)
    return (data_array, y_array)

def data_generator(streamSize, X, y): 
    #X, y = load_data(filepath_train)
    count = 0
    while True:
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        count += streamSize
        yield X_result, y_result
        
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

# get data
X_train, y_train = load_data(filepath_train)
#X_train, y_train = flip_images(X_train, y_train)

# build trees
features = []
for i in range(784):
    f = "f" + str(i)
    features.append(f)

tree_e = Vfdt(features, delta=0.01, nmin=2000, tau=1.2)
tree_p = Vfdt(features, delta=0.01, nmin=2000, tau=1.2)

accArray_E = []
accArray = []
indices = []
# training in base phase
gen = data_generator(sizeOneBatch, X_train, y_train)
for i in range(nbBaseBatches):
    print(i)
    
    X_gen, y_gen = next(gen)
    X_gen = X_gen.reshape(-1, 784)
    y_gen_E = np.zeros(sizeOneBatch)
    
    for X, y in zip(X_gen, y_gen_E):
        tree_e.update(X, y)
    for X, y in zip(X_gen, y_gen):
        tree_p.update(X, y)

    # c, r = y_E.shape
    # y_E = y_E.reshape(c,)
    # scores = cross_val_score(tree_e, X, y_E, cv=5)
    # print(scores)
    # accArray_E.append(scores.mean())

    # c, r = y.shape
    # y = y.reshape(c,)
    # scores = cross_val_score(tree_p, X, y, cv=5)
    # print(scores)
    # accArray.append(scores.mean())

# adaption: data changed
for i in range(nbBaseBatches, nbBatches):
    print(i)

    X_gen_org, y_gen_org = next(gen)
    X_gen = None
    y_gen = None
    doFlip = i >= nbBatches / 2
    if doFlip:
        X_gen, y_gen = flip_images(X_gen_org, y_gen_org)
        y_gen = y_gen.reshape(-1,)
        y_gen_E = np.full(len(y_gen), 1.0)
    else:
        X_gen = X_gen_org.reshape(-1, 784)
        y_gen = y_gen_org
        y_gen_E = np.zeros(len(y_gen_org))
    # evaluate
    predict_E = tree_e.predict(X_gen)
    acc_E = accuracy_score(predict_E, y_gen_E)
    print(acc_E)
    accArray_E.append(acc_E)
    predict = tree_p.predict(X_gen)
    acc = accuracy_score(predict, y_gen)
    print(acc)
    accArray.append(acc)
    indices.append(i)

    # training
    for X, y in zip(X_gen, y_gen):
        tree_p.update(X, y)

    if doFlip:
        X_combine = np.concatenate((X_gen_org.reshape(-1, 784), X_gen))
        y_E_org = np.zeros(len(X_gen))
        y_combine = np.concatenate((y_E_org, y_gen_E))
        X_shuffled, y_shuffled = shuffle(X_combine, y_combine)
        for X, y in zip(X_shuffled, y_shuffled):
            tree_e.update(X, y)
    else:
        for X, y in zip(X_gen, y_gen_E):
            tree_e.update(X, y)


np.savez("mnist_hoeffding.npz", acc=accArray, acc_E=accArray_E, indices=indices)
# https://github.com/doubleplusplus/incremental-decision-tree-CART-python/blob/master/vfdt.py