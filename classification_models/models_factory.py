import functools
import keras_applications as ka

from .models import resnet as rn
from .models import resnext as rx
from .models import senet as sn


class ModelsFactory:
    _models = {

        # ResNets
        'resnet18': [rn.ResNet18, rn.preprocess_input],
        'resnet34': [rn.ResNet34, rn.preprocess_input],
        'resnet50': [rn.ResNet50, rn.preprocess_input],
        'resnet101': [rn.ResNet101, rn.preprocess_input],
        'resnet152': [rn.ResNet152, rn.preprocess_input],

        # SE-Nets
        'seresnet18': [rn.SEResNet18, rn.preprocess_input],
        'seresnet34': [rn.SEResNet34, rn.preprocess_input],
        'seresnet50': [sn.SEResNet50, sn.preprocess_input],
        'seresnet101': [sn.SEResNet101, sn.preprocess_input],
        'seresnet152': [sn.SEResNet152, sn.preprocess_input],
        'seresnext50': [sn.SEResNeXt50, sn.preprocess_input],
        'seresnext101': [sn.SEResNeXt101, sn.preprocess_input],
        'senet154': [sn.SENet154, sn.preprocess_input],

        # Resnet V2
        'resnet50v2': [ka.resnet_v2.ResNet50V2, ka.resnet_v2.preprocess_input],
        'resnet101v2': [ka.resnet_v2.ResNet101V2, ka.resnet_v2.preprocess_input],
        'resnet152v2': [ka.resnet_v2.ResNet152V2, ka.resnet_v2.preprocess_input],

        # ResNext
        'resnext50': [rx.ResNeXt50, rx.preprocess_input],
        'resnext101': [rx.ResNeXt101, rx.preprocess_input],

        # VGG
        'vgg16': [ka.vgg16.VGG16, ka.vgg16.preprocess_input],
        'vgg19': [ka.vgg19.VGG19, ka.vgg19.preprocess_input],

        # Densnet
        'densenet121': [ka.densenet.DenseNet121, ka.densenet.preprocess_input],
        'densenet169': [ka.densenet.DenseNet169, ka.densenet.preprocess_input],
        'densenet201': [ka.densenet.DenseNet201, ka.densenet.preprocess_input],

        # Inception
        'inceptionresnetv2': [ka.inception_resnet_v2.InceptionResNetV2,
                              ka.inception_resnet_v2.preprocess_input],
        'inceptionv3': [ka.inception_v3.InceptionV3, ka.inception_v3.preprocess_input],
        'xception': [ka.xception.Xception, ka.xception.preprocess_input],

        # Nasnet
        'nasnetlarge': [ka.nasnet.NASNetLarge, ka.nasnet.preprocess_input],
        'nasnetmobile': [ka.nasnet.NASNetMobile, ka.nasnet.preprocess_input],

        # MobileNet
        'mobilenet': [ka.mobilenet.MobileNet, ka.mobilenet.preprocess_input],
        'mobilenetv2': [ka.mobilenet_v2.MobileNetV2, ka.mobilenet_v2.preprocess_input],
    }

    @property
    def models(self):
        return self._models

    def models_names(self):
        return list(self.models.keys())

    @staticmethod
    def get_kwargs():
        return {}

    def inject_submodules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            modules_kwargs = self.get_kwargs()
            new_kwargs = dict(list(kwargs.items()) + list(modules_kwargs.items()))
            return func(*args, **new_kwargs)

        return wrapper

    def get(self, name):
        if name not in self.models_names():
            raise ValueError('No such model `{}`, available models: {}'.format(
                name, list(self.models_names())))

        model_fn, preprocess_input = self.models[name]
        model_fn = self.inject_submodules(model_fn)
        preprocess_input = self.inject_submodules(preprocess_input)
        return model_fn, preprocess_input
