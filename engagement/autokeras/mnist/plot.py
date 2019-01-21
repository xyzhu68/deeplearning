from keras.models import load_model
from keras.utils import plot_model

MODEL_DIR = "autokeras_mnist_Ci_flip.h5"
model = load_model(MODEL_DIR)
plot_model(model, to_file="flip_ci.png")