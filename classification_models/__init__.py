from .__version__ import __version__

from . import resnet as rn
from . import senet as sn
from . import keras_applications as ka


__all__ = ['__version__', 'Classifiers']


class Classifiers:

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
        'resnext50': [ka.resnext.ResNeXt50, ka.resnext.preprocess_input],
        'resnext101': [ka.resnext.ResNeXt101, ka.resnext.preprocess_input],

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

    @classmethod
    def names(cls):
        return sorted(cls._models.keys())

    @classmethod
    def get(cls, name):
        """
        Access to classifiers and preprocessing functions

        Args:
            name (str): architecture name

        Returns:
            callable: function to build keras model
            callable: function to preprocess image data

        """
        return cls._models.get(name)

    @classmethod
    def get_classifier(cls, name):
        return cls._models.get(name)[0]

    @classmethod
    def get_preprocessing(cls, name):
        return cls._models.get(name)[1]
