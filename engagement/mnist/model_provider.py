from keras.activations import relu, softmax
from keras.layers.merge import add
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model, Sequential


def make_conv_model(nbFilters, noTop):
    nb_filters = nbFilters # 64
    nb_conv = 3
    img_rows = 28
    img_cols = 28
    nb_pool = 2

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    name="layer1",
                    padding='valid',
                    activation='relu',
                    input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', activation='relu', name="layer2"))
    model.add(MaxPooling2D(name="layer3", pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', activation='relu', name="layer4"))
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', activation='relu', name="layer5"))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name="layer6"))
    model.add(Dropout(0.25, name="layer7"))

    if noTop:
        return model


    model.add(Flatten(name="Flatten"))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])

    return model