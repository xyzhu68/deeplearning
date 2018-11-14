import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import rankdata

def plot_Ei_n(drift_type):
    d0 = np.load("mnist_engage_{0}_1.npz".format(drift_type))
    d1 = np.load("mnist_engage_{0}_2.npz".format(drift_type))
    d2 = np.load("mnist_engage_{0}_3.npz".format(drift_type))
    d3 = np.load("mnist_engage_{0}_4.npz".format(drift_type))
    d4 = np.load("mnist_engage_{0}_5.npz".format(drift_type))
    d5 = np.load("mnist_engage_{0}_6.npz".format(drift_type))
    d6 = np.load("mnist_engage_{0}_7.npz".format(drift_type))


    # d0 = d0["accE"]
    # d1 = d1["accE"]
    # d2 = d2["accE"]
    # d3 = d3["accE"]
    # d4 = d4["accE"]
    # d5 = d5["accE"]
    # d6 = d6["accE"]

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

def plot_one_npz(fileName, drift_type):
    data = np.load(fileName)
    begin = 10
    indices = data["indices"][begin:]
    # accArray_Base = data["accBase"]
    # accArray_E = data["accE"]
    # accArray_P = data["accP"]
    # accEiPi = data["accEiPi"]


    plt.plot(indices, data["accBaseUpdated"][begin:], label="Base update")
    plt.plot(indices, data["accFreezing"][begin:], label="Freezing")
    plt.plot(indices, data["accBase"][begin:], label="Base line")
    plt.plot(indices, data["accEiPi"][begin:], label = "NN-Patching")
    plt.plot(indices, data["accMSPi"][begin:], label = "NN-Patching ms")
    plt.title("MNIST - {0}".format(drift_type))
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()

# drift_type = "remap"
# layer = 7
# plot_one_npz("mnist_engage_{0}_{1}.npz".format(drift_type, layer), drift_type)

def plot_engagement(drift_type):
    acc_list = []
    for i in range(7):
        file = "mnist_engage_{0}_{1}.npz".format(drift_type, i+1)
        data = np.load(file)
        final_acc = data["accMSPi"][-5]
        acc_list.append(np.mean(final_acc))

    plt.plot([1, 2, 3, 4, 5, 6, 7], acc_list, marker="s")
    plt.title("MNIST Engagement (ms) - {0}".format(drift_type))
    plt.ylabel("Accuracy")
    plt.xlabel("Engagement Layer")
    plt.legend()
    plt.show()
    
#plot_engagement("rotate")

def calculate_metrics(inputFile, outputFile, cp):
    data = np.load(inputFile)
    acc = []
    acc.append(data["accBase"])
    acc.append(data["accBaseUpdated"])
    acc.append(data["accFreezing"])
    acc.append(data["accEiPi"])
    acc.append(data["accMSPi"])

    finalBatch = 5

    titles = ["Base", "Base update", "Freezing", "NN-Patching", "NN-Patching-MS"]
    of = open(outputFile, "w")
    # final acc
    finalAcc = []
    of.write("Final accuracy\n")
    for i in range(5):
        item = acc[i]
        avAcc = np.mean(item[-finalBatch:])
        line = "{0}: {1}".format(titles[i], avAcc)
        of.write(line + "\n")
        finalAcc.append(avAcc)
    of.write("\n\n")
    # average acc
    of.write("Average accuracy\n")
    for i in range(5):
        item = acc[i]
        avAcc = np.mean(item[-cp:])
        line = "{0}: {1}".format(titles[i], avAcc)
        of.write(line + "\n")
    of.write("\n\n")
    # recovery speed
    of.write("Recovery Speed\n")
    indices = []
    for i in range(5):
        item = acc[i]
        index = 0
        targetAcc = finalAcc[i]
        for j in range(cp, len(item)):
            if item[j] >= 0.9 and np.mean(item[j:j+5]) >= 0.9: #targetAcc:
                index = j
                break
        line = "{0}: {1}".format(titles[i], index - cp)
        of.write(line + "\n")
        indices.append(index)
    of.write("\n\n")
    # adaption rank
    of.write("Adaption Rank\n")
    maxIndex = max(indices)
    if maxIndex == cp:
        maxIndex = cp + 10
    ranks = [0, 0, 0, 0, 0]
    for i in range(cp, maxIndex):
        acc_at_i = []
        for j in range(5):
            acc_at_i.append(acc[j][i])
        curr_rank = rankdata([-1 * k for k in acc_at_i])
        ranks = [a+b for a, b in zip(ranks, curr_rank)]
    ranks = [r / (maxIndex - cp) for r in ranks]
    for i in range(5):
        line = "{0}: {1}".format(titles[i], ranks[i])
        of.write(line + "\n")
    of.write("\n\n")
    # final rank
    of.write("Final rank\n")
    ranks = [0, 0, 0, 0, 0]
    for i in range(80-finalBatch, 80):
        acc_at_i = []
        for j in range(5):
            acc_at_i.append(acc[j][i])
        curr_rank = rankdata([-1 * k for k in acc_at_i])
        ranks = [a+b for a, b in zip(ranks, curr_rank)]
    ranks = [r / finalBatch for r in ranks]
    for i in range(5):
        line = "{0}: {1}".format(titles[i], ranks[i])
        of.write(line + "\n")
    of.write("\n\n")
    #final oscillation
    of.write("Average oscillation\n")
    # for i in range(5):
    #     item = acc[i]
    #     minimum = min(item[-finalBatch:])
    #     maximum = max(item[-finalBatch:])
    #     line = "{0}: {1}".format(titles[i], maximum - minimum)
    #     of.write(line + "\n")
    for i in range(5):
        item = acc[i]
        avgAcc = np.mean(item[-30:])
        total = 0
        for j in range(50, 80):
            total += abs(item[j] - avgAcc)
        line = "{0}: {1}".format(titles[i], total / 30)
        of.write(line + "\n")
    

    of.close()

#calculate_metrics("mnist_engage_rotate_7.npz", "rotate_metrics.txt", 50-20)  # 50 - 20 

def plot_filters(drift_type):
    data16 = np.load("filters/mnist_filters_{0}_16.npz".format(drift_type))
    data32 = np.load("filters/mnist_filters_{0}_32.npz".format(drift_type))
    data64 = np.load("filters/mnist_filters_{0}_64.npz".format(drift_type))
    data128 = np.load("filters/mnist_filters_{0}_128.npz".format(drift_type))
    
    data = [data16, data32, data64, data128]
    accs = []
    for i in range(4):
        accs.append(np.mean(data[i]["accMSPi"][-40:]))

    x = np.array([0,1,2,3])
    xticks = ["16", "32", "64", "128"]
    plt.xticks(x, xticks)
    plt.plot(x, accs, marker="s")
    plt.ylabel("Accuracy")
    plt.xlabel("Filters")
    #plt.legend()
    plt.show()

#plot_filters("rotate")

def plot_filter_time(drift_type):
    data16 = np.load("filters/mnist_filters_{0}_16.npz".format(drift_type))
    data32 = np.load("filters/mnist_filters_{0}_32.npz".format(drift_type))
    data64 = np.load("filters/mnist_filters_{0}_64.npz".format(drift_type))
    data128 = np.load("filters/mnist_filters_{0}_128.npz".format(drift_type))
    
    data = [data16, data32, data64, data128]
    times = []
    for i in range(4):
        d_str = data[i]["duration"]
        t = datetime.strptime(str(d_str).split(".")[0],"%H:%M:%S")
        delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        times.append(delta.total_seconds())

    x = np.array([0,1,2,3])
    xticks = ["16", "32", "64", "128"]
    plt.xticks(x, xticks)
    plt.plot(x, times, marker="s")
    plt.xlabel("Filters")
    plt.ylabel("Running time")
    plt.show()

plot_filter_time("rotate")