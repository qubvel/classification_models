import keras.backend as K
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.models import Model
from keras.engine import get_source_inputs

from .params import get_conv_params
from .params import get_bn_params

from .blocks import residual_conv_block
from .blocks import residual_bottleneck_block

from ..common.blocks import SpatialSE
from ..common.blocks import ChannelSE
from ..common.blocks import ChannelSpatialSE


def build_resnet(
     repetitions=(2, 2, 2, 2),
     include_top=True,
     input_tensor=None,
     input_shape=None,
     classes=1000,
     block_type='conv',
     attention=None):
    
    """
    TODO
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    if block_type == 'conv':
        residual_block = residual_conv_block
    elif block_type == 'bottleneck':
        residual_block = residual_bottleneck_block
    else:
        raise ValueError('Block type "{}" not in ["conv", "bottleneck"]'.format(block_type))

    # choose attention block type
    if attention == 'sse':
        attention_block = SpatialSE()
    elif attention == 'cse':
        attention_block = ChannelSE(reduction=16)
    elif attention == 'csse':
        attention_block = ChannelSpatialSE(reduction=2)
    elif attention is None:
        attention_block = None
    else:
        raise ValueError('Supported attention blocks are: sse, cse, csse. Got "{}".'.format(attention))

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # resnet bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
    
    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            
            filters = init_filters * (2**stage)
            
            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='post', attention=attention_block)(x)
                
            elif block == 0:
                x = residual_block(filters, stage, block, strides=(2, 2),
                                   cut='post', attention=attention_block)(x)
                
            else:
                x = residual_block(filters, stage, block, strides=(1, 1),
                                   cut='pre', attention=attention_block)(x)
                
    x = BatchNormalization(name='bn1', **bn_params)(x)
    x = Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = Model(inputs, x)

    return model
