from keras.activations import relu, softmax
from keras.layers.merge import add
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model

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
    # input tensor is the 28x28 grayscale image
    input_tensor = Input((28, 28, 1))

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
    # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_4
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
    # F_5
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

    # F_6
    # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_7   
    # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)

    # dropout for more robust learning
    x = Dropout(0.2)(x)

    # last softmax layer
    x = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation(softmax)(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model