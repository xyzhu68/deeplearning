import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import rankdata

def plot_one_npz(fileName, drift_type):
    data = np.load(fileName)
    begin = 30
    indices = data["indices"][begin:]
    i2 = data["indices"][10:]
    # accArray_Base = data["accBase"]
    # accArray_E = data["accE"]
    # accArray_P = data["accP"]
    # accEiPi = data["accEiPi"]
    print("duration: {0}".format(data["duration"]))

    #plt.plot(indices, data["accBaseUpdated"][begin:], label="Base update")
    #plt.plot(indices, data["accFreezing"][begin:], label="Freezing")
    plt.plot(i2, data["accBase"][10:], label="Base line")
    plt.plot(indices, data["accEiPi"][begin:], label = "NN-Patching")
    plt.plot(indices, data["accMSPi"][begin:], label = "NN-Patching ms")
    # plt.plot(indices, data["accE"][begin:], label="Ei")
    # plt.plot(indices, data["accMS"][begin:], label="MS")
    # plt.plot(indices, data["accP"][begin:], label="Pi")
    plt.title("Dog-Monkey (VGG16) - {0}".format(drift_type))
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

drift_type = "transfer"
plot_one_npz("vgg_{0}_base.npz".format(drift_type), drift_type)
# layers = [1,2,3,4,5]
# for layer in layers:
#     plot_one_npz("vgg_{0}_{1}_simple.npz".format(drift_type, layer), drift_type)


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

def give_engagement_data(drift_type):
    acc_list_ei = []
    acc_list_ms = []
    for i in range(4):
        file = "vgg_{0}_{1}.npz".format(drift_type, i+1)
        data = np.load(file)
        final_acc_ms = data["accMSPi"][-5]
        acc_list_ms.append(np.mean(final_acc_ms))
        final_acc_ei = data["accEiPi"][-5]
        acc_list_ei.append(np.mean(final_acc_ei))

    ft_data = np.load("vgg_{0}_base.npz".format(drift_type))
    ft_acc_ms = ft_data["accMSPi"][-5]
    acc_list_ms.append(np.mean(ft_acc_ms))
    ft_acc_ei = ft_data["accEiPi"][-5]
    acc_list_ei.append(np.mean(ft_acc_ei))

    # print(drift_type + ": Error detector")
    # print(acc_list_ei)
    print(drift_type + ": Model selector")
    print(acc_list_ms)

# give_engagement_data("transfer")

def calculate_metrics(input_file, cp):
    data = np.load(input_file)
    acc_ei = data["accEiPi"]
    acc_ms = data["accMSPi"]

    finalAcc = np.mean(acc_ei[-5:])
    print("final accuracy ei: {0}".format(finalAcc))

    finalAcc_ms = np.mean(acc_ms[-5:])
    print("final accuracy ms: {0}".format(finalAcc_ms))

    begin = 100 - cp
    acAcc = np.mean(acc_ei[-begin:])
    print("average accuracy ei: ", acAcc)
    acAcc = np.mean(acc_ms[-begin:])
    print("average accuracy ms: ", acAcc)

    index_ei = 0
    for i in range(cp - 20, len(acc_ei)):
        #print(acc_ei[i])
        if acc_ei[i] >= 0.9 * finalAcc:
            index_ei = i
            break
    print("recovery speed ei: {0}".format(index_ei - (cp - 20)))

    index_ms = 0
    for i in range(cp - 20, len(acc_ms)):
        if acc_ms[i] >= 0.9 * finalAcc_ms:
            index_ms = i
            break
    print("recovery speed ms: {0}".format(index_ms - (cp - 20)))

    print("duration of {0}: {1}".format(input_file, data["duration"]))

# drift_type = "transfer"
# cp = 50
# calculate_metrics("vgg_{0}_base.npz".format(drift_type), cp)