from keras.layers import Lambda, Dense
from keras.layers import Conv2D
from keras.layers import Add, Multiply
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute
from keras import backend as K


def channel_se_block(ratio=16):
    ''' Create a squeeze-excite block

    Squeeze and Excitation Networks in Keras
        https://github.com/titu1994/keras-squeeze-excite-network

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    def layer(input_tensor):
        init = input_tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init._keras_shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = Multiply()([init, se])
        return x
    return layer


def spatial_se_block():
    def layer(input_tensor):
        x = Conv2D(1, (1, 1), kernel_initializer="he_normal", activation='sigmoid', use_bias=False)(input_tensor)
        x = Multiply()([input_tensor, x])
        return x
    return layer


def csse_block(ratio=2):
    '''
    Spatial and Channel Squeeze & Excitation Block (scSE)
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568

    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
    '''
    def layer(input_tensor):
        cse = channel_se_block(ratio)(input_tensor)
        sse = spatial_se_block()(input_tensor)
        x = Add()([cse, sse])

        return x
    return layer
