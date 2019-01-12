import keras.backend as K
import keras.layers as kl

from ..common.blocks import ChannelSE
from ..common.blocks import GroupConv2D


bn_params = {
    'epsilon': 9.999999747378752e-06,
}


def SEResNetBottleneck(filters, reduction=16, strides=1, **kwargs):

    def layer(input):

        x = input
        residual = input

        # bottleneck
        x = kl.Conv2D(filters // 4, (1, 1), kernel_initializer='he_uniform', strides=strides, use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(filters // 4, (3, 3),
                      kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = K.int_shape(x)[-1]
        r_channels = K.int_shape(residual)[-1]

        if strides != 1 or x_channels != r_channels:

            residual = kl.Conv2D(x_channels, (1, 1), strides=strides,
                                 kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = kl.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction)(x)

        # add residual connection
        x = kl.Add()([x, residual])

        x = kl.Activation('relu')(x)

        return x
    return layer


def SEResNeXtBottleneck(filters, reduction=16, strides=1, groups=32, base_width=4, **kwargs):

    def layer(input):

        x = input
        residual = input

        width = (filters // 4) * base_width * groups // 64

        # bottleneck
        x = kl.Conv2D(width, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = GroupConv2D(width, (3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = K.int_shape(x)[-1]
        r_channels = K.int_shape(residual)[-1]

        if strides != 1 or x_channels != r_channels:

            residual = kl.Conv2D(x_channels, (1, 1), strides=strides,
                                 kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = kl.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction)(x)

        # add residual connection
        x = kl.Add()([x, residual])

        x = kl.Activation('relu')(x)

        return x
    return layer


def SEBottleneck(filters, reduction=16, strides=1, groups=64, is_first=False):

    if is_first:
        downsample_kernel_size = (1, 1)
        padding = False
    else:
        downsample_kernel_size = (3, 3)
        padding = True

    def layer(input):

        x = input
        residual = input

        # bottleneck
        x = kl.Conv2D(filters // 2, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = GroupConv2D(filters, (3, 3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = K.int_shape(x)[-1]
        r_channels = K.int_shape(residual)[-1]

        if strides != 1 or x_channels != r_channels:
            if padding:
                residual = kl.ZeroPadding2D(1)(residual)
            residual = kl.Conv2D(x_channels, downsample_kernel_size, strides=strides,
                                 kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = kl.BatchNormalization(**bn_params)(residual)

        # apply attention module
        x = ChannelSE(reduction=reduction)(x)

        # add residual connection
        x = kl.Add()([x, residual])

        x = kl.Activation('relu')(x)

        return x
    return layer
