import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def plot_one_npz(fileName):
    data = np.load(fileName)
    indices = data["indices"]
    accArray_Base = data["accBase"]
    accArray_E = data["accE"]
    accArray_P = data["accP"]
    print(indices)
    plt.hold(True)

    plt.plot(indices, accArray_Base, label="acc base")
    plt.plot(indices, accArray_E, label="acc Ei")
    plt.plot(indices, accArray_P, label="acc Patching")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

plot_one_npz("mnist_drift_flip_simple_64.npz")