# default parameters for convolution and batchnorm layers of ResNet models
# parameters are obtained from MXNet converted model


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def get_model_params(name):

    params = {

        'resnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': None,
        },

        'resnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': None,
        },

        'resnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'resnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'resnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': None,
        },

        'seresnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': 'cse',
        },

        'seresnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': 'cse',
        },

        'seresnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'seresnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'seresnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': 'cse',
        },

        'csseresnet18': {
            'repetitions': (2, 2, 2, 2),
            'block_type': 'conv',
            'attention': 'csse',
        },

        'csseresnet34': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'conv',
            'attention': 'csse',
        },

        'csseresnet50': {
            'repetitions': (3, 4, 6, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

        'csseresnet101': {
            'repetitions': (3, 4, 23, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

        'csseresnet152': {
            'repetitions': (3, 8, 36, 3),
            'block_type': 'bottleneck',
            'attention': 'csse',
        },

    }

    return params[name]