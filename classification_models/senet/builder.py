import keras.backend as K
import keras.layers as kl
import keras.models as km

from .blocks import SEResNeXtBottleneck
from .blocks import SEResNetBottleneck
from .blocks import SEBottleneck
from .blocks import bn_params


def build_senet(
        repetitions,
        input_tensor=None,
        input_shape=(None, None, 3),
        groups=1,
        block_type='resnet',
        reduction=16,
        init_filters=64,
        input_3x3=False,
        classes=1000,
        include_top=True,
        activation='softmax',
        dropout=None,
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
    classes (int): Number of outputs in `last_linear` layer.
        - For all models: 1000
    """

    # choose block type
    if block_type == 'resnet':
        residual_block = SEResNetBottleneck
    elif block_type == 'resnext':
        residual_block = SEResNeXtBottleneck
    elif block_type == 'senet':
        residual_block = SEBottleneck
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
        x = kl.Conv2D(init_filters,  (3, 3), strides=2,
                      use_bias=False, kernel_initializer='he_uniform')(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(init_filters, (3, 3), use_bias=False,
                      kernel_initializer='he_uniform')(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

        x = kl.ZeroPadding2D(1)(x)
        x = kl.Conv2D(init_filters * 2, (3, 3), use_bias=False,
                      kernel_initializer='he_uniform')(x)
        x = kl.BatchNormalization(**bn_params)(x)
        x = kl.Activation('relu')(x)

    else:
        x = kl.ZeroPadding2D(3)(x)
        x = kl.Conv2D(init_filters, (7, 7), strides=2, use_bias=False,
                      kernel_initializer='he_uniform')(x)
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

            # decrease spatial dimensions for each stage (except first, because we have maxpool before)
            if i == 0 and j == 0:
                x = residual_block(filters, reduction=reduction, strides=1, groups=groups, is_first=True)(x)
            elif i != 0 and j == 0:
                x = residual_block(filters, reduction=reduction, strides=2, groups=groups)(x)
            else:
                x = residual_block(filters, reduction=reduction, strides=1, groups=groups)(x)

    if include_top:
        x = kl.GlobalAveragePooling2D()(x)
        if dropout is not None:
            x = kl.Dropout(dropout)(x)
        x = kl.Dense(classes)(x)
        x = kl.Activation(activation)(x)

    model = km.Model(input, x)
    return model
