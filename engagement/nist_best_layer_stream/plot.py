import matplotlib.pyplot as plt
import numpy as np

def plot_engagement():
    data = np.load("nist_best_layer_flip.npz")
    engageResults = data["engageResults"]
    PAResults = data["PAResults"]
    x = []
    y = []
    for result in engageResults:
        topResult = result[0]
        x.append(topResult[0])
        y.append(topResult[1]+1)

    plt.plot(x, y, marker="s", markersize=2, linestyle="None")
    plt.axvline(x=50, color="red")
    plt.title("NIST - Best Engagement Layer on Batches")
    plt.ylabel("Layer")
    plt.xlabel("Batch")
    plt.legend()
    plt.show()
    print("Duration: ", data["duration"])
    topOne = PAResults[0][0]
    print("batch: {0}, var: {1}".format(topOne[0], topOne[1]))

plot_engagement()