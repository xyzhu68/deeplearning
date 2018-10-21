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

plot_Ei_n("flip")

def plot_one_npz(fileName):
    data = np.load(fileName)
    indices = data["indices"]
    accArray_Base = data["accBase"]
    accArray_E = data["accE"]
    accArray_P = data["accP"]
    accEiPi = data["accEiPi"]


    plt.plot(indices, accArray_Base, label="acc base")
    plt.plot(indices, accArray_E, label="acc Ei")
    plt.plot(indices, accArray_P, label="acc Pi")
    plt.plot(indices, accEiPi, label = "Ei + Pi")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

#plot_one_npz("mnist_drift_transfer_resnet_fs.npz")

def plot_filters(drift_type):
    data16 = np.load("mnist_drift_{0}_simple_16.npz".format(drift_type))
    data32 = np.load("mnist_drift_{0}_simple_32.npz".format(drift_type))
    data64 = np.load("mnist_drift_{0}_simple_64.npz".format(drift_type))
    data128 = np.load("mnist_drift_{0}_simple_128.npz".format(drift_type))

    acc16 = np.mean(data16["accEiPi"][-20])
    acc32 = np.mean(data32["accEiPi"][-20])
    acc64 = np.mean(data64["accEiPi"][-20])
    acc128 = np.mean(data128["accEiPi"][-20])

    plt.plot([16, 32, 64, 128], [acc16, acc32, acc64, acc128], marker="s")
    plt.ylabel("Accuracy")
    plt.xlabel("Filters")
    plt.legend()
    plt.show()

#plot_filters("transfer")

def plot_freezing_acc(drift_type):
    data0 = np.load("mnist_drift_{0}_resnet_fs.npz".format(drift_type))
    data1 = np.load("mnist_drift_{0}_resnet_1.npz".format(drift_type))
    data2 = np.load("mnist_drift_{0}_resnet_2.npz".format(drift_type))
    data3 = np.load("mnist_drift_{0}_resnet_3.npz".format(drift_type))
    data4 = np.load("mnist_drift_{0}_resnet_4.npz".format(drift_type))
    data5 = np.load("mnist_drift_{0}_resnet_5.npz".format(drift_type))
    data6 = np.load("mnist_drift_{0}_resnet_6.npz".format(drift_type))

    acc0 = np.mean(data0["accEiPi"][80:])
    acc1 = np.mean(data1["accEiPi"][80:])
    acc2 = np.mean(data2["accEiPi"][80:])
    acc3 = np.mean(data3["accEiPi"][80:])
    acc4 = np.mean(data4["accEiPi"][80:])
    acc5 = np.mean(data5["accEiPi"][80:])
    acc6 = np.mean(data6["accEiPi"][80:])



    plt.plot([0, 1, 2, 3, 4, 5, 6], [acc0, acc1, acc2, acc3, acc4, acc5, acc6], marker="s")
    plt.ylabel("Accuracy")
    plt.xlabel("Layers frozen")
    plt.legend()
    plt.show()

#plot_freezing_acc("transfer")

def compare_acc(drift_type):
    accList_no_patching = []
    accList = []
    for n in [-1, 1, 2, 3, 4, 5, 6]:
        data = np.load("mnist_{0}_resnet_{1}_freezing_no_patching.npz".format(drift_type, n))
        acc_no_patching = np.mean(data["acc"][-20])
        accList_no_patching.append(acc_no_patching)
        
        layer = str(n) if n > 0 else "fs"
        data2 = np.load("mnist_drift_{0}_resnet_{1}.npz".format(drift_type, layer))
        acc = np.mean(data2["accEiPi"][-20])
        accList.append(acc)

    plt.plot([0, 1, 2, 3, 4, 5, 6], accList, label="acc patching")
    plt.plot([0, 1, 2, 3, 4, 5, 6], accList_no_patching, label="acc no patching")
    plt.ylabel("Accuracy")
    plt.xlabel("Layers frozen")
    plt.legend()
    plt.show()

#compare_acc("transfer")