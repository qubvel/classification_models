import keras.backend as K
import keras.layers as kl
import keras.models as km


bn_params = {
    'epsilon': 9.999999747378752e-06,
}


def GroupConv2D(filters,
                kernel_size,
                strides=(1, 1),
                groups=32,
                kernel_initializer='he_uniform',
                use_bias=True,
                activation=None):

    def layer(x):

        inp_ch = int(K.int_shape(x)[-1] // groups) # input grouped channels
        out_ch = int(filters // groups) # output grouped channels

        blocks = []
        for c in range(groups):
            x = kl.Lambda(lambda z: z[..., c*inp_ch:(c + 1)*inp_ch])(x)
            x = kl.Conv2D(out_ch,
                          kernel_size,
                          strides=strides,
                          kernel_initializer=kernel_initializer,
                          use_bias=use_bias,
                          activation=activation)(x)
            blocks.append(x)

        x = kl.Concatenate(axis=-1)(blocks)
        return x
    return layer


def SE(reduction):

    def layer(input):

        # get number of channels/filters
        channels = K.int_shape(input)[-1]

        pool_size = {
            256: (56, 56),
            512: (28, 28),
            1024: (14, 14),
            2048: (7, 7),
        }

        # squeeze and excite
        x = input
        x = kl.AveragePooling2D(pool_size[channels])(x)
        x = kl.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('relu')(x)
        x = kl.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = kl.Activation('sigmoid')(x)

        # apply attention
        x = kl.Multiply()([input, x])

        return x

    return layer


def SEResNetBottleneck(filters, reduction=16, strides=1):

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

        if strides != 1 or  x_channels != r_channels:

            residual = kl.Conv2D(x_channels, (1, 1), strides=strides,
                                 kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = kl.BatchNormalization(**bn_params)(residual)


        # apply attention module
        x = SE(reduction)(x)

        # add residual connection
        x = kl.Add()([x, residual])

        x = kl.Activation('relu')(x)

        return x
    return layer


def SEResNeXtBottleneck(filters, reduction=16, strides=1):

    def layer(input):

        x = input
        residual = input

        # bottleneck
        x = kl.Conv2D(filters // 4, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = GroupConv2D(filters // 4, (3, 3), strides=strides,
                      kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.Conv2D(filters, (1, 1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)

        #  if number of filters or spatial dimensions changed
        #  make same manipulations with residual connection
        x_channels = K.int_shape(x)[-1]
        r_channels = K.int_shape(residual)[-1]

        if strides != 1 or  x_channels != r_channels:

            residual = kl.Conv2D(x_channels, (1, 1), strides=strides,
                                 kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = kl.BatchNormalization(**bn_params)(residual)


        # apply attention module
        x = SE(reduction)(x)

        # add residual connection
        x = kl.Add()([x, residual])

        x = kl.Activation('relu')(x)

        return x
    return layer


def SENet(repetitions,
          input_tensor=None,
          input_shape=(None, None, 3),
          groups=1,
          block_type='resnet',
          reduction=16,
          init_filters=64,
          input_3x3=True,
          classes=1000,
          include_top=True,
          activation='softmax',
          ):
    """
    Parameters
    ----------
    block (str):
        - For SENet154: 'senet'
        - For SE-ResNet models: 'resnet'
        - For SE-ResNeXt models:  'resnext'
    layers (list of ints): Number of residual blocks for 4 layers of the
        network (layer1...layer4).
    groups (int): Number of groups for the 3x3 convolution in each
        bottleneck block.
        - For SENet154: 64
        - For SE-ResNet models: 1
        - For SE-ResNeXt models:  32
    reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
        - For all models: 16
    dropout_p (float or None): Drop probability for the Dropout layer.
        If `None` the Dropout layer is not used.
        - For SENet154: 0.2
        - For SE-ResNet models: None
        - For SE-ResNeXt models: None
    init_filters (int):  Number of input channels for layer1.
        - For SENet154: 128
        - For SE-ResNet models: 64
        - For SE-ResNeXt models: 64
    input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
        a single 7x7 convolution in layer0.
        - For SENet154: True
        - For SE-ResNet models: False
        - For SE-ResNeXt models: False
    num_classes (int): Number of outputs in `last_linear` layer.
        - For all models: 1000
    """

    # choose block type
    if block_type == 'resnet':
        ResidualBlock = SEResNetBottleneck
    elif block_type == 'resnext':
        ResidualBlock = SEResNeXtBottleneck
    else:
        raise ValueError('Block type not in ["resnet", "resnext", "senet"]')

    # define input
    if input_tensor is None:
        input = kl.Input(shape=input_shape, name='input')
    else:
        if not K.is_keras_tensor(input_tensor):
            input = kl.Input(tensor=input_tensor, shape=input_shape)
        else:
            input = kl.input_tensor

    x = input

    if input_3x3:

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(64,  (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(64,  (3, 3), padding='same', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(init_filters, (3, 3), padding='same', use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

    else:
        x = kl.ZeroPadding2D(3)(x)
        x = kl.Conv2D(init_filters, (7, 7), strides=2, use_bias=False)(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

    x = kl.ZeroPadding2D(1)(x)
    x = kl.MaxPooling2D((3, 3), strides=2)(x)


    # body of resnet
    filters = init_filters * 2
    for i, stage in enumerate(repetitions):

        # increase number of filters with each stage
        filters *= 2

        for j in range(stage):

            if i != 0 and j == 0:
                x = ResidualBlock(filters, reduction=reduction, strides=2)(x)
            else:
                x = ResidualBlock(filters, reduction=reduction, strides=1)(x)

    if include_top:
        x = kl.GlobalAveragePooling2D()(x)
        x = kl.Dense(classes)(x)
        x = kl.Activation(activation)(x)

    model = km.Model(input, x)
    return model
