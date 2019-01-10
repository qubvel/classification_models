from .builder import build_senet
from ..utils import load_model_weights
from ..weights import weights_collection


__all__ = ['SEResNet50', 'SEResNeXt50', 'SEResNet101', 'SEResNeXt101', 'SEResNet152']


models_params = {

    'seresnet50': {
        'repetitions': (3, 4, 6, 3),
        'block_type': 'resnet',
        'input_3x3': False,
        'groups': 1,
        'reduction': 16,
        'init_filters': 64,
    },

    'seresnet101': {
        'repetitions': (3, 4, 23, 3),
        'block_type': 'resnet',
        'input_3x3': False,
        'groups': 1,
        'reduction': 16,
        'init_filters': 64,
    },

    'seresnet152': {
        'repetitions': (3, 8, 36, 3),
        'block_type': 'resnet',
        'input_3x3': False,
        'groups': 1,
        'reduction': 16,
        'init_filters': 64,
    },

    'seresnext50': {
        'repetitions': (3, 4, 6, 3),
        'block_type': 'resnext',
        'input_3x3': False,
        'groups': 32,
        'reduction': 16,
        'init_filters': 64,
    },

    'seresnext101': {
        'repetitions': (3, 4, 23, 3),
        'block_type': 'resnext',
        'input_3x3': False,
        'groups': 32,
        'reduction': 16,
        'init_filters': 64,
    },
}


def _get_senet(name):

    def classifier(input_shape=None, input_tensor=None, weights='imagenet', classes=1000, include_top=True):

        params = models_params[name]

        model = build_senet(
                      input_shape=input_shape,
                      input_tensor=input_tensor,
                      classes=classes,
                      include_top=include_top,
                      **params)

        model.name = name

        if weights is not None:
            load_model_weights(weights_collection, model, weights, classes, include_top)

        return model
    return classifier


SEResNet50 = _get_senet('seresnet50')
SEResNet101 = _get_senet('seresnet101')
SEResNet152 = _get_senet('seresnet152')
SEResNeXt50 = _get_senet('seresnext50')
SEResNeXt101 = _get_senet('seresnext101')