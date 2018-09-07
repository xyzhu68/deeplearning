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

data = np.load(Path("../DecisionTree/mnist_horffding.npz"))

indices = data["indices"]
accArray = data["acc"]
accArray_E = data["acc_E"]


# result of accuracy
plt.plot(indices, accArray, label="acc patching clf")
plt.plot(indices, accArray_E, label="acc error clf")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch")
plt.legend()
plt.show()

