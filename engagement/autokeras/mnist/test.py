from keras.datasets import mnist
#from autokeras.image.image_supervised import ImageClassifier
from keras.models import load_model, Model
from keras.utils import to_categorical
from keras.activations import softmax
from keras.layers import Activation

# settings for GPU
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# clf = ImageClassifier(verbose=True)
# clf.fit(x_train, y_train, time_limit=8 * 60 )
# clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
# y = clf.evaluate(x_test, y_test)
# print(y)

model = load_model('autokeras_mnist_Ei_flip_12.h5')
# model.layers[-1].activation = softmax
x = model.output
x = Activation('sigmoid', name='activation_add')(x)
model = Model(model.input, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # categorical_crossentropy


model.fit(x_train, y_train,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_data=(x_test, y_test))



# https://github.com/jhfjhfj1/autokeras/issues/186