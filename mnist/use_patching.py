from scipy.io import arff
import numpy as np
from keras.models import load_model


filepath_test = "test.arff"

# read data
data, meta = arff.loadarff(filepath_test)
dataArray = np.asarray(data.tolist(), dtype=np.float32)
X_test = np.delete(dataArray, 784, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1)
y_test = dataArray[:,784]
for i in range(len(y_test)):
    if (y_test[i] == 2):
        y_test[i] = 9
    elif (y_test[i] == 9):
        y_test[i] = 2

model_base = load_model("model_base.h5")
model_Ci = load_model("Ci.h5")
model_Ei = load_model("Ei.h5")
count_correct = 0

for x, y in zip(X_test, y_test):
    x = np.asarray([x])
    e = model_Ei.predict(x)
    e = e[0, 0]
    predict = None
    if e > 0.5: # patching
        predict = model_Ci.predict(x)
    else:
        predict = model_base.predict(x)
    pred_num = np.argmax(predict[0])
    if pred_num == y:
        count_correct += 1

print(f"accuracy {count_correct / len(y_test)}")
