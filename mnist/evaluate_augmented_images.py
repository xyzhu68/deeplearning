from scipy.io import arff
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from random import getrandbits

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


filepath_test = "test.arff"

# read data
data, meta = arff.loadarff(filepath_test)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_test = np.delete(dataArray, 784, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1)
y_test = dataArray[:,784]

# augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,)

datagen.fit(X_test)

model_Ci = load_model("Ci_Aug.h5")
count_correct = 0

for x, y in zip(X_test, y_test):
    x = np.asarray([x])
    if getrandbits(1):
        x = datagen.flow(x, batch_size=1)[0]

    predict = model_Ci.predict(x)
    pred_num = np.argmax(predict[0])
    if pred_num == y:
        count_correct += 1

print(f"accuracy {count_correct / len(y_test)}")
