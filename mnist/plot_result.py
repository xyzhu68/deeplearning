import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def draw_batch_accuracy(dataFile, pngFile):
    #data = np.load(Path("mnist_drift_remap_resnet_3.npz"))
    data = np.load(dataFile)

    indices = data["indices"]
    accArray = data["acc"]
    accArray_E = data["acc_E"]
    accChainedArray = data["accChained"]


    # result of accuracy
    fig = plt.figure()
    plt.plot(indices, accArray, label="acc patching clf")
    plt.plot(indices, accArray_E, label="acc error clf")
    plt.plot(indices, accChainedArray, label="acc Ei+Ci")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    #plt.show()
    plt.savefig(pngFile)
    plt.close(fig)

# for i in range(7):
#     dataPath = "resnet_{0}/mnist_drift_{0}_resnet_{1}.npz".format("transfer", i)
#     pngPath = "resnet_{0}/mnist_drift_{0}_resnet_{1}".format("transfer", i)
#     draw_batch_accuracy(Path(dataPath), pngPath)


"""
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

def show_attach_result(clf, label, clf2, label2):
    accList = []
    accList2 = []
    for i in range(7):
        fileName = "resnet_{0}/mnist_drift_{0}_resnet_{1}.npz".format("transfer", i)
        accList.append(get_average_acc(fileName, clf, 20))
        accList2.append(get_average_acc(fileName, clf2, 20))

    fig = plt.figure()
    plt.plot(range(7), accList, label=label)
    plt.plot(range(7), accList2, label=label2)
    plt.xlabel("frozen layers")
    plt.ylabel("accuracy")
    plt.legend()
    #plt.show()
    plt.savefig("resnet_{0}/result_Ei_Ci.png".format("transfer"))
    plt.close(fig)

#show_attach_result("acc", "Patching Clf Accuracy")
#show_attach_result("acc_E", "Error Clf Accuracy")
#show_attach_result("acc", "Patching Clf Accuracy", "acc_E", "Error Clf Accuracy")

def show_flip_result(clf, label, clf2, label2):
    drift_type = "appear"
    accList = []
    accList2 = []
    filters = [16, 32, 64, 128]
    for i in range(4):
        fileName = "simple_{0}_filters/mnist_drift_{0}_fs_filters_{1}.npz".format(drift_type, filters[i])
        accList.append(get_average_acc(fileName, clf, 50))
        accList2.append(get_average_acc(fileName, clf2, 50))

    fig = plt.figure()

    plt.plot(filters, accList, label=label)
    plt.plot(filters, accList2, label=label2)
    plt.xlabel("filters")
    plt.ylabel("accuracy")
    plt.legend()
    #plt.show()
    plt.savefig("simple_{0}_filters/result_filters.png".format(drift_type))

#show_flip_result("acc", "Patching Clf Accuracy")
#show_flip_result("acc_E", "Error Clf Accuracy")
show_flip_result("acc", "Patching Clf", "acc_E", "Error Clf")