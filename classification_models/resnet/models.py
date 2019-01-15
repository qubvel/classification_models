from .builder import build_resnet
from ..utils import load_model_weights
from ..weights import weights_collection
from .params import get_model_params

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
           'SEResNet18', 'SEResNet34', 'preprocess_input']

# preprocessing function
preprocess_input = lambda x: x


def _get_resnet(name):
    def classifier(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
        model_params = get_model_params(name)
        model = build_resnet(input_tensor=input_tensor,
                             input_shape=input_shape,
                             classes=classes,
                             include_top=include_top,
                             **model_params)

        model.name = name

        if weights:
            load_model_weights(weights_collection, model, weights, classes, include_top)

        return model
    return classifier


# classic resnet models
ResNet18 = _get_resnet('resnet18')
ResNet34 = _get_resnet('resnet34')
ResNet50 = _get_resnet('resnet50')
ResNet101 = _get_resnet('resnet101')
ResNet152 = _get_resnet('resnet152')

# resnets with squeeze and excitation attention block
SEResNet18 = _get_resnet('seresnet18')
SEResNet34 = _get_resnet('seresnet34')
# SEResNet50 = _get_resnet('seresnet50')
# SEResNet101 = _get_resnet('seresnet101')
# SEResNet152 = _get_resnet('seresnet152')
#
# # resnets with concurrent squeeze and excitation attention block
# CSSEResNet18 = _get_resnet('csseresnet18')
# CSSEResNet34 = _get_resnet('csseresnet34')
# CSSEResNet50 = _get_resnet('csseresnet50')
# CSSEResNet101 = _get_resnet('csseresnet101')
# CSSEResNet152 = _get_resnet('csseresnet152')
