import os
import collections

from classification_models import get_submodules_from_kwargs
from ._common_blocks import GroupConv2D
from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions']
)


# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------


def conv_block(filters, stage, block, strides=(2, 2), **kwargs):
    """The conv block is the block that has conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        strides: tuple of integers, strides for conv (3x3) layer in block
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        # extracting params and names for layers
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), strides=strides, **group_conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)

        x = layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        shortcut = layers.Conv2D(filters * 2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = layers.BatchNormalization(name=sc_name + '_bn', **bn_params)(shortcut)
        x = layers.Add()([x, shortcut])

        x = layers.Activation('relu', name=relu_name)(x)
        return x

    return layer


def identity_block(filters, stage, block, **kwargs):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        filters: integer, used for first and second conv layers, third conv layer double this value
        stage: integer, current stage label, used for generating layer names
        block: integer, current block label, used for generating layer names
    # Returns
        Output layer for the block.
    """

    def layer(input_tensor):
        conv_params = get_conv_params()
        group_conv_params = dict(list(conv_params.items()) + list(kwargs.items()))
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(input_tensor)
        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), **group_conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)

        x = layers.Conv2D(filters * 2, (1, 1), name=conv_name + '3', **conv_params)(x)
        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)

        x = layers.Add()([x, input_tensor])

        x = layers.Activation('relu', name=relu_name)(x)
        return x

    return layer


def ResNeXt(
        model_params,
        include_top=True,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        weights='imagenet',
        **kwargs):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='data')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnext bottom
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnext body
    init_filters = 128
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1), **kwargs)(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2), **kwargs)(x)

            else:
                x = identity_block(filters, stage, block, **kwargs)(x)

    # resnext top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = models.Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name,
                               weights, classes, include_top, **kwargs)

    return model


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'resnext50': ModelParams('resnext50', (3, 4, 6, 3)),
    'resnext101': ModelParams('resnext101', (3, 4, 23, 3)),
}


def ResNeXt50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def ResNeXt101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True, **kwargs):
    return ResNeXt(
        MODELS_PARAMS['resnext101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        **kwargs
    )


def preprocess_input(x, **kwargs):
    return x


setattr(ResNeXt50, '__doc__', ResNeXt.__doc__)
setattr(ResNeXt101, '__doc__', ResNeXt.__doc__)
