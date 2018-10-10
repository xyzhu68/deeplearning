from keras.activations import relu, softmax
from keras.layers.merge import add
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model, Sequential

# https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        
        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f

def make_resnet():
    # input tensor
    input_tensor = Input((128, 128, 1))

    # first conv2d with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # F_1
    x = block(16)(x)
    # F_2
    x = block(16)(x)

    # F_3
    # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
    # and we can't add together tensors of inconsistent sizes, so we use upscale=True
    x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_4
    x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
    # F_5
    x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

    # F_6
    x = block(64, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_7   
    x = block(64)(x)                     # !!! <------- Uncomment for local evaluation

    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)

    # dropout for more robust learning
    x = Dropout(0.2)(x)

    # # last softmax layer
    x = Dense(units=36, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation(softmax)(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #return Model(inputs=input_tensor, outputs=x)
    return model

def make_simple_model(Ei, suffix):
    nb_filters = 64
    nb_conv = 3
    img_rows = 128
    img_cols = 128
    nb_pool = 4

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    name="layer1",
                    padding='valid',
                    input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', name="layer2"))
    model.add(Activation('relu', name="layer3"))
    model.add(MaxPooling2D(name="layer4", pool_size=(nb_pool, nb_pool)))

    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer5"+suffix))
    model.add(Conv2D(nb_filters * 2, (nb_conv, nb_conv), padding='valid', name="layer6"+suffix))
    model.add(Activation('relu', name="layer7"+suffix))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name="layer8"+suffix))
    model.add(Dropout(0.25, name="layer9"+suffix))


    model.add(Flatten(name="Flatten"+suffix))
    model.add(Dense(128, name="dense1"+suffix))
    model.add(Activation('relu', name="act"+suffix))
    model.add(Dropout(0.5, name="dropout"+suffix))
    if Ei:
        model.add(Dense(1, name="dropout2"+suffix))
        model.add(Activation('sigmoid', name="sigmoid"+suffix))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['binary_accuracy'])
    else:
        model.add(Dense(36, name="dense"+suffix))
        model.add(Activation('softmax', name="softmax"+suffix))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
    

    return model