from scipy.io import arff
import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Dense, Activation, Dropout
from keras.utils import to_categorical
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
    model_Ei.fit(X_train, y_train_Ei, batch_size=100, epochs=20)
    model_Ei.save("Ei.h5")
else:
    out_Ci = Dense(10, activation="softmax")(out)
    model_Ci = Model(class_input, out_Ci)
    model_Ci.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model_Ci.fit(X_train_Ci, y_train_Ci, batch_size=100, epochs=20)
    model_Ci.save("Ci.h5")
