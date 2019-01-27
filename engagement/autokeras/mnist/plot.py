from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

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

plot_found_model_info()