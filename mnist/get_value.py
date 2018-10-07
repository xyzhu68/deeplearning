import numpy as np

def get_y_value(fileName, name, x):
    data = np.load(fileName)
    acc = data[name]
    return acc[x]

# example
yValue = get_y_value("resnet_flip_2/mnist_drift_flip_resnet_6.npz", "acc_E", 70)
print(yValue)
