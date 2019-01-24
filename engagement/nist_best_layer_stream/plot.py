import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_engagement(drift_type):
    data = np.load("nist_best_layer_{0}.npz".format(drift_type))
    engageResults = data["engageResults"]
    PAResults = data["PAResults"]
    x = []
    y = []
    for result in engageResults:
        topResult = result[0]
        x.append(topResult[0])
        y.append(topResult[1]+1)

    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if drift_type == "appear":
        ax.set_ylim([0, 13])
        plt.axvline(x=30, color="red")
    else:
        plt.axvline(x=50, color="red")

    plt.plot(x, y, marker="s", markersize=2, linestyle="None")
    
    plt.title("NIST - Best Engagement Layer on Batches ({0})".format(drift_type))
    plt.ylabel("Layer")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()
    print("Duration: ", data["duration"])
    topOne = PAResults[0][0]
    print("batch: {0}, var: {1}".format(topOne[0], topOne[1]))

#plot_engagement("remap")

def plot_PA(drift_type):
    dict = {1 : "A", 2 : "B", 10 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G", 7 : "H", 8 : "I", 9 : "J"}
    data = np.load("nist_best_layer_{0}.npz".format(drift_type))
    PAResults = data["PAResults"]
    x = []
    y = []
    for result in PAResults:
        topResult = result[1]
        x.append(topResult[0])
        y.append(dict[topResult[1]+1])

    yticks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    ax = plt.figure().gca()
    ax.set_yticklabels(yticks)

    if drift_type == "appear":
        plt.axvline(x=30, color="red")
    else:
        plt.axvline(x=50, color="red")

    plt.plot(x, y, marker="D", markersize=2, linestyle="None", color="blue")
    # plt.axvline(x=50, color="red")
    plt.title("NIST - Best Patch on Batches ({0})".format(drift_type))
    plt.ylabel("Architecture")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()
    print("Duration: ", data["duration"])

#plot_PA("flip")

def give_sum_for_engage(drift_type):
    data = np.load("nist_best_layer_{0}.npz".format(drift_type))
    engageResults = data["engageResults"]
    
    changePoint = 50
    if drift_type == "appear":
        changePoint = 30
    dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
    for result in engageResults:
        topResult = result[0]
        if topResult[0] < changePoint: 
            continue
        #print(topResult[0])
        # x.append(topResult[0]) # batch
        # y.append(topResult[1]+1) # best layer
        layer = topResult[1] + 1
        dict[layer] += 1

    print (dict)

#give_sum_for_engage("appear")

def give_sum_for_architecture(drift_type):
    changePoint = 50
    if drift_type == "appear":
        changePoint = 30
    mapping = {1 : "A", 2 : "B", 10 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G", 7 : "H", 8 : "I", 9 : "J"}
    index = 1
    dataFile = "nist_best_layer_{0}.npz".format(drift_type)
    if drift_type == "flip" or drift_type == "remap":
        dataFile = "nist_best_layer_{0}_PA.npz".format(drift_type)
        index = 0
    data = np.load(dataFile)
    PAResults = data["PAResults"]

    resultDict = {"A":0, "B":0, "C":0, "D":0, "E":0, "F":0, "G":0, "H":0, "I":0, "J":0}
    for result in PAResults:
        topResult = result[index]
        if topResult[0] < changePoint:
            continue
        #x.append(topResult[0]) # batch
        #y.append(mapping[topResult[1]+1]) # arch name
        arch = mapping[topResult[1]+1]
        resultDict[arch] += 1

    print(resultDict)

give_sum_for_architecture("transfer")