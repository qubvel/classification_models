import keras.backend as K
import keras.layers as kl

from keras.utils.generic_utils import get_custom_objects


class Slice(kl.Layer):

    def __init__(self, start, stop, **kwargs):
        self.start = start
        self.stop = stop
        super(Slice, self).__init__(**kwargs)

    def call(self, x):
        return x[..., self.start:self.stop]

    def compute_output_shape(self, input_shape):
        bs, h, w, ch = input_shape
        new_ch = self.stop - self.start
        return (bs, h, w, new_ch)

    def get_config(self):
        config = super(Slice, self).get_config()
        config['start'] = self.start
        config['stop'] = self.stop
        return config


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation='linear',
                padding='valid',
                **kwargs):
    """
    Grouped Convolution Layer implemented as a Slice,
    Conv2D and Concatenate layers. Split filters to groups, apply Conv2D and concatenate back.

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the stride
            length of the convolution.
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use (see activations).
            If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
        padding: one of "valid" or "same" (case-insensitive).

    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".

    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.

    """

    def layer(input_tensor):
        inp_ch = int(K.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            start = c * inp_ch
            stop = (c + 1) * inp_ch
            x = Slice(start, stop)(input_tensor)
            x = kl.Conv2D(out_ch,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          use_bias=use_bias,
                          activation=activation,
                          padding=padding,
                          **kwargs)(x)
            blocks.append(x)

        x = kl.Concatenate(axis=-1)(blocks)
        return x

    return layer


def ChannelSE(reduction=16):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """

    def layer(input_tensor):
        # get number of channels/filters
        channels = K.int_shape(input_tensor)[-1]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        # custom global average pooling where keepdims=True
        x = kl.Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(x)
        x = kl.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('sigmoid')(x)

        # apply attention
        x = kl.Multiply()([input_tensor, x])

        return x

    return layer


def SpatialSE():
    """
    Spatial squeeze and excitation block (applied across spatial dimensions)
    """

    def layer(input_tensor):
        x = kl.Conv2D(1, (1, 1), kernel_initializer="he_normal", activation='sigmoid', use_bias=False)(input_tensor)
        x = kl.Multiply()([input_tensor, x])
        return x

    return layer


def ChannelSpatialSE(reduction=2):
    """
    Spatial and Channel Squeeze & Excitation Block (scSE)
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568

    Implementation of Concurrent Spatial and Channel `Squeeze & Excitation` in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
    """

    def layer(input_tensor):
        cse = ChannelSE(reduction=reduction)(input_tensor)
        sse = SpatialSE()(input_tensor)
        x = kl.Add()([cse, sse])

        return x

    return layer


get_custom_objects().update({
    'Slice': Slice,
})

