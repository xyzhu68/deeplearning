from scipy.io import arff
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator

filepath_train = Path("../mnist/train.arff")
filepath_test = Path("../mnist/test.arff")

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


X_train, y_train = load_data(filepath_train)
X_train, y_train = flip_images(X_train, y_train)
print(X_train.shape)
print(y_train.shape)

X_test, y_test = load_data(filepath_test)
X_test, y_test = flip_images(X_test, y_test)

trainingImagesCount = len(X_train)
testingImagesCount = len(X_test)
print(trainingImagesCount)
print(testingImagesCount)

clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
clf = clf.fit(X_train, y_train)

predictionRes = clf.predict(X_test)

print (metrics.classification_report(y_test, predictionRes, digits=4))

c, r = y_train.shape
print(c)
y_train = y_train.reshape(c,)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print (scores)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))