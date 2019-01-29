from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

def plot_model():
    MODEL_DIR = "autokeras_mnist_Ci_flip.h5"
    model = load_model(MODEL_DIR)
    plot_model(model, to_file="flip_ci.png")

def plot_found_model_info():
    fig, ax1 = plt.subplots()
    model_id = np.arange(1, 21)
    plt.xticks(model_id)
    loss = [
            2.0389631032943725,
            0.8699543207883835,
            0.6558810085058212,
            0.6918977051973343,
            0.654238361120224,
            0.6275521427392959,
            0.6637942597270012,
            0.6114598050713539,
            0.6675792306661605,
            0.5916307374835015,
            0.6433073043823242,
            0.5769579187035561,
            0.5725218251347541,
            0.5803564831614494,
            0.6010283149778843,
            0.6218903139233589,
            0.5618297100067139,
            0.5875071451067925,
            0.5794000700116158,
            0.5859546527266503 ]
    acc = [
            0.8384,
            0.9292000000000001,
            0.9480000000000001,
            0.95444,
            0.9483999999999998,
            0.958,
            0.9551999999999999,
            0.9583999999999999,
            0.9516,
            0.9596,
            0.9591999999999998,
            0.9616,
            0.9632,
            0.9640000000000001,
            0.9656,
            0.9583999999999999,
            0.9635999999999999,
            0.9632,
            0.9628,
            0.9632
    ]
    ax1.plot(model_id, loss, 'b-s')
    ax1.set_xlabel('Model ID')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(model_id, acc, 'g-*')
    ax2.set_ylabel('Accuracy', color='g')
    ax2.tick_params('y', colors='g')

    fig.tight_layout()
    plt.show()

#plot_found_model_info()

def plot_time_acc():
    x = [1, 6, 12]
    acc_ci_flip = [0.9635833333333333, 0.96975, 0.974]
    acc_ei_flip = [0.6957, 0.7947, 0.7879]
    acc_ci_remap = [0.991801878736123, 0.9959009393680615, 0.9970964987190436]
    acc_ei_remap = [0.9475878499106611, 0.95096287472702, 0.9483819733968633]

    plt.xticks(x)
    plt.plot(x, acc_ci_flip, label='Ci flip', marker='s')
    plt.plot(x, acc_ei_flip, label='Ei flip', marker='o')
    plt.plot(x, acc_ci_remap, label='Ci remap', marker='s')
    plt.plot(x, acc_ei_remap, label='Ei remap', marker='o')
    plt.title('Accuracy vs Time')
    plt.ylabel('Accuracy')
    plt.xlabel('Running time (hours)')
    plt.legend()
    plt.show()

#plot_time_acc()

def show_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image=x_train[6781]
    plt.imshow(image, cmap='gray')
    plt.show()

#show_mnist()

def plot_one_npz(fileName, drift_type):
    data = np.load(fileName)
    begin = 10
    indices = data["indices"][begin:]
    # accArray_Base = data["accBase"]
    # accArray_E = data["accE"]
    # accArray_P = data["accP"]
    # accEiPi = data["accEiPi"]



    plt.plot(indices, data["accEiPi"][begin:], label = "Patching with AK models")

    plt.title("MNIST - {0}".format(drift_type))
    plt.ylabel("Accuracy")
    plt.xlabel("Batch")
    plt.legend(loc = "lower right")
    plt.show()

#plot_one_npz("mnist_ak_remap_1_sgd.npz", "remap")

def calculate_metrics(input_file):
    data = np.load(input_file)
    acc = data["accEiPi"]

    finalAcc = np.mean(acc[-5:])
    print("final accuracy: {0}".format(finalAcc))

    acAcc = np.mean(acc[-50:])
    print("average accuracy: ", acAcc)

    index = 0
    for i in range(30, len(acc)):
        print(acc[i])
        if acc[i] >= 0.9 * finalAcc:
            index = i
            break
    print("recovery speed: {0}".format(index - 30))
    print("duration of {0}: {1}".format(input_file, data["duration"]))


calculate_metrics("mnist_ak_remap_1_sgd.npz")