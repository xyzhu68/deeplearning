from scipy.io import arff
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
import sys

filepath_train = Path("../mnist/train.arff")
filepath_test = Path("../mnist/test.arff")

# Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        x_data = data[0].reshape((784,))
        data_list.append(x_data)
        y_list.append(data[1])
        batch_index = batch_index + 1

    data_array = np.asarray(data_list)
    y_array = np.asarray(y_list)
    return (data_array, y_array)

def data_generator(streamSize): 
    X, y = load_data(filepath_train)
    count = 0
    while True:
        X_result = X[count : count + streamSize]
        y_result = y[count : count + streamSize]
        yield X_result, y_result
        count += streamSize
#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

# settings
totalDataSize = 60000
nbBatches = 100 # devide dataset into 100 batches
nbBaseBatches = 20 # size of base dataset
sizeOneBatch = totalDataSize // nbBatches

clf_e = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
clf_p = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
accArray_E = []
accArray = []
indices = []
# training in base phase
for i in range(nbBaseBatches):
    print(i)
    gen = data_generator(sizeOneBatch)
    X, y = next(gen)
    X = X.reshape(-1, 784)
    y_E = np.zeros(sizeOneBatch)

    clf_e = clf_e.fit(X, y_E)
    clf_p = clf_p.fit(X, y)

    scores_e = cross_val_score(clf_e, X, y_E, cv=5)
    print (scores_e)
    scores_p = cross_val_score(clf_p, X, y, cv=5)
    print (scores_p)


"""predictionRes_e = clf_e.predict(X_test)
predictionRes_p = clf_p.predict()

print (metrics.classification_report(y_test, predictionRes, digits=4))

c, r = y_train.shape
y_train = y_train.reshape(c,)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print (scores)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
"""

# https://github.com/doubleplusplus/incremental-decision-tree-CART-python/blob/master/vfdt.py