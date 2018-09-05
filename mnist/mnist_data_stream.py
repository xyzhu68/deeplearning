from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K


filepath_train = "./train.arff"
filepath_test = "./test.arff"

nb_filters = 32
nb_conv = 3
img_rows = 28
img_cols = 28
nb_pool = 2

def read_arff(fileName):
    data, meta = arff.loadarff(fileName)
    dataArray = np.asarray(data.tolist(), dtype=np.float32)
    X_train = np.delete(dataArray, 784, 1) / 255
    X_train = X_train.reshape(-1, img_rows, img_cols, 1)
    y_train = dataArray[:,784]
    y_train = to_categorical(y_train, 10)
    return X_train, y_train

def data_generator(streamSize): 
    X_train, y_train = read_arff(filepath_train)
    X_test, y_test = read_arff(filepath_test)
    print(X_train.shape)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    X = X[30000:]
    y = y[30000:]
    print(y.shape)
    count = 0
    while True:
        if count >= 40000:
            yield None, None
        print("Count: ", count)
        X_result = X[count : count + streamSize : 1]
        y_result = y[count : count + streamSize : 1]
        yield X_result, y_result
        count += streamSize


gen = data_generator(1000)
lossArray = []
accArray = []
indices = []
i = 0
savedFileName = "model_30K.h5"

while True:
    model = load_model(savedFileName)
    X, y = next(gen)
    if X is None:
        break
    # Test the data
    loss, acc = model.evaluate(X, y)
    # save the loss/accuracy
    lossArray.append(loss)
    accArray.append(acc)
    indices.append(i)
    
    # train the data
    model.fit(X, y, batch_size=32, epochs=5)
    
    model.save(savedFileName)

    i += 1

plt.plot(indices, lossArray, label="loss")
plt.legend()
plt.show()

plt.plot(indices, accArray, label="accuracy")
plt.legend()
plt.show()

print("last loss: ", lossArray[-1])
print("last accuracy: ", accArray[-1])

#K.clear_session()                    