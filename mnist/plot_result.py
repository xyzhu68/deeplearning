import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
data = np.load("mnist_drift_clf_results.npz")

indices = data["indices"]
accArray = data["acc"]
accArray_E = data["acc_E"]
lossArray = data["loss"]
lossArray_E = data["loss_E"]

# result of accuracy
plt.plot(indices, accArray, label="acc patching clf")
plt.plot(indices, accArray_E, label="acc error clf")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch")
plt.legend()
plt.show()

# result of loss
plt.plot(indices, lossArray, label="loss patching clf")
plt.plot(indices, lossArray_E, label="loss error clf")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Batch")
plt.legend()
plt.show()
"""

# data = np.load(Path("../DecisionTree/mnist_hoeffding.npz"))

# indices = data["indices"]
# accArray = data["acc"]
# accArray_E = data["acc_E"]


# # result of accuracy
# plt.plot(indices, accArray, label="acc patching clf")
# plt.plot(indices, accArray_E, label="acc error clf")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Batch")
# plt.legend()
# plt.show()

def get_average_acc(file, acc_key, cp):
    data = np.load(file)
    accArray = data[acc_key]
    accArray = accArray[cp:]
    return np.mean(accArray)

def show_attach_result(clf, label):
    accList = []
    #accList2 = []
    for i in range(7):
        fileName = "mnist_drift_appear_resnet_{0}.npz".format(i)
        accList.append(get_average_acc(fileName, clf, 20))
        #accList2.append(get_average_acc(fileName, clf2, 20))

    plt.plot(range(7), accList, label=label)
    #plt.plot(range(7), accList2, label=label2)
    plt.xlabel("frozen layers")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

#show_attach_result("acc", "Patching Clf Accuracy")
#show_attach_result("acc_E", "Error Clf Accuracy")
#show_attach_result("acc", "Patching Clf Accuracy", "acc_E", "Error Clf Accuracy")

def show_flip_result(clf, label, clf2, label2):
    accList = []
    accList2 = []
    filters = [16, 32, 64, 128]
    for i in range(4):
        fileName = "mnist_drift_flip_from_scratch_{0}.npz".format(filters[i])
        accList.append(get_average_acc(fileName, clf, 50))
        accList2.append(get_average_acc(fileName, clf2, 50))

    plt.plot(filters, accList, label=label)
    plt.plot(filters, accList2, label=label2)
    plt.xlabel("filters")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

#show_flip_result("acc", "Patching Clf Accuracy")
#show_flip_result("acc_E", "Error Clf Accuracy")
show_flip_result("acc", "Patching Clf", "acc_E", "Error Clf")