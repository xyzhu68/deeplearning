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
    accEiPi = data["accEiPi"]
    # print(indices)
    # plt.hold(True)


    plt.plot(indices, accArray_Base, label="acc base")
    plt.plot(indices, accArray_E, label="acc Ei")
    plt.plot(indices, accArray_P, label="acc Pi")
    plt.plot(indices, accEiPi, label = "Ei + Pi")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

# plot_one_npz("nist_drift_remap_simple_fs.npz")

def plot_freezing(drift_type):
    data0 = np.load("nist_drift_{0}_simple_fs.npz".format(drift_type))
    data1 = np.load("nist_drift_{0}_simple_1.npz".format(drift_type))
    data2 = np.load("nist_drift_{0}_simple_2.npz".format(drift_type))
    data3 = np.load("nist_drift_{0}_simple_5.npz".format(drift_type))

    acc0 = data0["accEiPi"]
    acc1 = data1["accEiPi"]
    acc2 = data2["accEiPi"]
    acc3 = data3["accEiPi"]

    plt.plot(acc0, label="no freezing")
    plt.plot(acc1, label="freeze 1 layer")
    plt.plot(acc2, label="freeze 2 layessr")
    plt.plot(acc3, label="freeze 3 layesrs")

    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

#plot_freezing("flip")

def plot_freezing_acc(drift_type):
    data0 = np.load("nist_drift_{0}_simple_fs.npz".format(drift_type))
    data1 = np.load("nist_drift_{0}_simple_1.npz".format(drift_type))
    data2 = np.load("nist_drift_{0}_simple_2.npz".format(drift_type))
    data3 = np.load("nist_drift_{0}_simple_5.npz".format(drift_type))

    # acc0 = np.mean(data0["accEiPi"][80:])
    # acc1 = np.mean(data1["accEiPi"][80:])
    # acc2 = np.mean(data2["accEiPi"][80:])
    # acc3 = np.mean(data3["accEiPi"][80:])

    acc0 = np.mean(data0["accP"][80:])
    acc1 = np.mean(data1["accP"][80:])
    acc2 = np.mean(data2["accP"][80:])
    acc3 = np.mean(data3["accP"][80:])

    plt.plot([0, 1, 2, 3], [acc0, acc1, acc2, acc3], marker="s")
    plt.ylabel("Accuracy")
    plt.xlabel("Layers frozen")
    plt.legend()
    plt.show()

#plot_freezing_acc("appear")

def compare_acc(drift_type):
    accList_no_patching = []
    accList = []
    for n in [0, 1, 2, 5]:
        data = np.load("model_{0}_freeze_{1}_no_patching.npz".format(drift_type, n))
        acc_no_patching = np.mean(data["acc"][-20])
        accList_no_patching.append(acc_no_patching)
        
        layer = str(n) if n > 0 else "fs"
        data2 = np.load("nist_drift_{0}_simple_{1}.npz".format(drift_type, layer))
        acc = np.mean(data2["accEiPi"][-20])
        accList.append(acc)

    plt.plot([0, 1, 2, 3], accList, label="acc patching")
    plt.plot([0, 1, 2, 3], accList_no_patching, label="acc no patching")
    plt.ylabel("Accuracy")
    plt.xlabel("Layers frozen")
    plt.legend()
    plt.show()

compare_acc("flip")