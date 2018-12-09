import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import rankdata

def plot_one_npz(fileName, drift_type):
    data = np.load(fileName)
    begin = 0
    indices = data["indices"][begin:]
    # accArray_Base = data["accBase"]
    # accArray_E = data["accE"]
    # accArray_P = data["accP"]
    # accEiPi = data["accEiPi"]
    print("duration: {0}".format(data["duration"]))

    # #plt.plot(indices, data["accBaseUpdated"][begin:], label="Base update")
    # #plt.plot(indices, data["accFreezing"][begin:], label="Freezing")
    # plt.plot(indices, data["accBase"][begin:], label="Base line")
    # plt.plot(indices, data["accEiPi"][begin:], label = "NN-Patching")
    # plt.plot(indices, data["accMSPi"][begin:], label = "NN-Patching ms")
    # # plt.plot(indices, data["accE"][begin:], label="Ei")
    # # plt.plot(indices, data["accMS"][begin:], label="MS")
    # # plt.plot(indices, data["accP"][begin:], label="Pi")
    # plt.title("Dog-Monkey (VGG16) - {0}".format(drift_type))
    # plt.ylabel("Accuracy")
    # plt.xlabel("Batch")
    # plt.legend()
    # plt.show()

drift_type = "remap"
layers = [1,2,3,4,5]
for layer in layers:
    plot_one_npz("vgg_{0}_{1}_simple.npz".format(drift_type, layer), drift_type)


def plot_engagement(drift_type):
    acc_list = []
    for i in range(4):
        file = "vgg_{0}_{1}.npz".format(drift_type, i+1)
        data = np.load(file)
        final_acc = data["accMSPi"][-5]
        acc_list.append(np.mean(final_acc))

    ft_data = np.load("vgg_{0}_base.npz".format(drift_type))
    ft_acc = ft_data["accMSPi"][-5]
    acc_list.append(np.mean(ft_acc))

    plt.plot(["1", "2", "3", "4", "fine-tuning"], acc_list, marker="s")
    plt.title("Dog-Monkdy (VGG) - Engagement (ms) - {0}".format(drift_type))
    plt.ylabel("Accuracy")
    plt.xlabel("Engagement Block")
    #plt.legend()
    plt.show()
    
#plot_engagement("transfer")