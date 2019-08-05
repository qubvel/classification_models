from classification_models import get_submodules_from_kwargs


def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


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

    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    slice_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = layers.Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = layers.Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)

        x = layers.Concatenate(axis=slice_axis)(blocks)
        return x

    return layer


def expand_dims(x, channels_axis):
    if channels_axis == 3:
        return x[:, None, None, :]
    elif channels_axis == 1:
        return x[:, :, None, None]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(channels_axis))


def ChannelSE(reduction=16, **kwargs):
    """
    Squeeze and Excitation block, reimplementation inspired by
        https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py

    Args:
        reduction: channels squeeze factor

    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # get number of channels/filters
        channels = backend.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # squeeze and excitation block in PyTorch style with
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = layers.Conv2D(channels // reduction, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(channels, (1, 1), kernel_initializer='he_uniform')(x)
        x = layers.Activation('sigmoid')(x)

        # apply attention
        x = layers.Multiply()([input_tensor, x])

        return x

    return layer
