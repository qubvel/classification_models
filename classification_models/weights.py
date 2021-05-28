from . import get_submodules_from_kwargs

__all__ = ['load_model_weights']


def _find_weights(model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(model, model_name, dataset, classes, include_top, **kwargs):
    _, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    weights = _find_weights(model_name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = keras_utils.get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


WEIGHTS_COLLECTION = [

    # ResNet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '64da73012bb70e16c901316c201d9803',
    },

    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5',
        'name': 'resnet18_imagenet_1000_no_top.h5',
        'md5': '318e3ac0cd98d51e917526c9f62f0b50',
    },

    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },

    # ResNet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000.h5',
        'name': 'resnet50_imagenet_1000.h5',
        'md5': 'd0feba4fc650e68ac8c19166ee1ba87f',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet_1000_no_top.h5',
        'name': 'resnet50_imagenet_1000_no_top.h5',
        'md5': 'db3b217156506944570ac220086f09b6',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586.h5',
        'md5': 'bb8963db145bc9906452b3d9c9917275',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'md5': 'd8bf4e7ea082d9d43e37644da217324a',
    },

    # ResNet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000.h5',
        'name': 'resnet101_imagenet_1000.h5',
        'md5': '9489ed2d5d0037538134c880167622ad',
    },

    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet101_imagenet_1000_no_top.h5',
        'name': 'resnet101_imagenet_1000_no_top.h5',
        'md5': '1016e7663980d5597a4e224d915c342d',
    },

    # ResNet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000.h5',
        'name': 'resnet152_imagenet_1000.h5',
        'md5': '1efffbcc0708fb0d46a9d096ae14f905',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet_1000_no_top.h5',
        'name': 'resnet152_imagenet_1000_no_top.h5',
        'md5': '5867b94098df4640918941115db93734',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221.h5',
        'name': 'resnet152_imagenet11k_11221.h5',
        'md5': '24791790f6ef32f274430ce4a2ffee5d',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet152_imagenet11k_11221_no_top.h5',
        'name': 'resnet152_imagenet11k_11221_no_top.h5',
        'md5': '25ab66dec217cb774a27d0f3659cafb3',
    },

    # ResNeXt50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000.h5',
        'name': 'resnext50_imagenet_1000.h5',
        'md5': '7c5c40381efb044a8dea5287ab2c83db',
    },

    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext50_imagenet_1000_no_top.h5',
        'name': 'resnext50_imagenet_1000_no_top.h5',
        'md5': '7ade5c8aac9194af79b1724229bdaa50',
    },

    # ResNeXt101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000.h5',
        'name': 'resnext101_imagenet_1000.h5',
        'md5': '432536e85ee811568a0851c328182735',
    },

    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnext101_imagenet_1000_no_top.h5',
        'name': 'resnext101_imagenet_1000_no_top.h5',
        'md5': '91fe0126320e49f6ee607a0719828c7e',
    },

    # SE models
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000.h5',
        'name': 'seresnet50_imagenet_1000.h5',
        'md5': 'ff0ce1ed5accaad05d113ecef2d29149',
    },

    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet50_imagenet_1000_no_top.h5',
        'name': 'seresnet50_imagenet_1000_no_top.h5',
        'md5': '043777781b0d5ca756474d60bf115ef1',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000.h5',
        'name': 'seresnet101_imagenet_1000.h5',
        'md5': '5c31adee48c82a66a32dee3d442f5be8',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet101_imagenet_1000_no_top.h5',
        'name': 'seresnet101_imagenet_1000_no_top.h5',
        'md5': '1c373b0c196918713da86951d1239007',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000.h5',
        'name': 'seresnet152_imagenet_1000.h5',
        'md5': '96fc14e3a939d4627b0174a0e80c7371',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet152_imagenet_1000_no_top.h5',
        'name': 'seresnet152_imagenet_1000_no_top.h5',
        'md5': 'f58d4c1a511c7445ab9a2c2b83ee4e7b',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000.h5',
        'name': 'seresnext50_imagenet_1000.h5',
        'md5': '5310dcd58ed573aecdab99f8df1121d5',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext50_imagenet_1000_no_top.h5',
        'name': 'seresnext50_imagenet_1000_no_top.h5',
        'md5': 'b0f23d2e1cd406d67335fb92d85cc279',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000.h5',
        'name': 'seresnext101_imagenet_1000.h5',
        'md5': 'be5b26b697a0f7f11efaa1bb6272fc84',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnext101_imagenet_1000_no_top.h5',
        'name': 'seresnext101_imagenet_1000_no_top.h5',
        'md5': 'e48708cbe40071cc3356016c37f6c9c7',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000.h5',
        'name': 'senet154_imagenet_1000.h5',
        'md5': 'c8eac0e1940ea4d8a2e0b2eb0cdf4e75',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/senet154_imagenet_1000_no_top.h5',
        'name': 'senet154_imagenet_1000_no_top.h5',
        'md5': 'd854ff2cd7e6a87b05a8124cd283e0f2',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000.h5',
        'name': 'seresnet18_imagenet_1000.h5',
        'md5': '9a925fd96d050dbf7cc4c54aabfcf749',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet18_imagenet_1000_no_top.h5',
        'name': 'seresnet18_imagenet_1000_no_top.h5',
        'md5': 'a46e5cd4114ac946ecdc58741e8d92ea',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000.h5',
        'name': 'seresnet34_imagenet_1000.h5',
        'md5': '863976b3bd439ff0cc05c91821218a6b',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/seresnet34_imagenet_1000_no_top.h5',
        'name': 'seresnet34_imagenet_1000_no_top.h5',
        'md5': '3348fd049f1f9ad307c070ff2b6ec4cb',
    },

]
