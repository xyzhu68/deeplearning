import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def plot_Ei_n(drift_type):
    d0 = np.load("mnist_drift_{0}_resnet_fs.npz".format(drift_type))
    d1 = np.load("mnist_drift_{0}_resnet_1.npz".format(drift_type))
    d2 = np.load("mnist_drift_{0}_resnet_2.npz".format(drift_type))
    d3 = np.load("mnist_drift_{0}_resnet_3.npz".format(drift_type))
    d4 = np.load("mnist_drift_{0}_resnet_4.npz".format(drift_type))
    d5 = np.load("mnist_drift_{0}_resnet_5.npz".format(drift_type))
    d6 = np.load("mnist_drift_{0}_resnet_6.npz".format(drift_type))

    # d0 = d0["accE"]
    # d1 = d1["accE"]
    # d2 = d2["accE"]
    # d3 = d3["accE"]
    # d4 = d4["accE"]
    # d5 = d5["accE"]
    # d6 = d6["accE"]

    # plt.plot(d0, label="E0")
    # plt.plot(d1, label="E1")
    # plt.plot(d2, label="E2")
    # plt.plot(d3, label="E3")
    # plt.plot(d4, label="E4")
    # plt.plot(d5, label="E5")
    # plt.plot(d6, label="E6")

    d0 = d0["accP"]
    d1 = d1["accP"]
    d2 = d2["accP"]
    d3 = d3["accP"]
    d4 = d4["accP"]
    d5 = d5["accP"]
    d6 = d6["accP"]

    plt.plot(d0, label="P0")
    plt.plot(d1, label="P1")
    plt.plot(d2, label="P2")
    plt.plot(d3, label="P3")
    plt.plot(d4, label="P4")
    plt.plot(d5, label="P5")
    plt.plot(d6, label="P6")

    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

#plot_Ei_n("flip")

def plot_one_npz(fileName):
    data = np.load(fileName)
    indices = data["indices"]
    # accArray_Base = data["accBase"]
    # accArray_E = data["accE"]
    # accArray_P = data["accP"]
    # accEiPi = data["accEiPi"]


    plt.plot(indices, data["accBaseUpdated"], label="Base update")
    plt.plot(indices, data["accFreezing"], label="Freezing")
    plt.plot(indices, data["accBase"], label="Base line")
    plt.plot(indices, data["accEiPi"], label = "NN-Patching")
    plt.plot(indices, data["accMSPi"], label = "NN-Patching ms")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

plot_one_npz("mnist_engage_flip_7.npz")

def plot_engagement():
    acc_list = []
    for i in range(7):
        file = "mnist_engage_flip_{0}.npz".format(i+1)
        data = np.load(file)
        final_acc = data["accEiPi"][-5]
        acc_list.append(np.mean(final_acc))

    plt.plot([1, 2, 3, 4, 5, 6, 7], acc_list, marker="s")
    plt.title("Engagement")
    plt.ylabel("Accuracy")
    plt.xlabel("Engagement Layer")
    plt.legend()
    plt.show()
    
#plot_engagement()