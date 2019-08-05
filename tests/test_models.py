import os
import six
import random
import pytest

import numpy as np
from skimage.io import imread
from keras_applications.imagenet_utils import decode_predictions

if os.environ.get('TF_KERAS'):
    import tensorflow.keras as keras
    from classification_models.tfkeras import Classifiers
else:
    import keras
    from classification_models.keras import Classifiers

KWARGS = Classifiers.get_kwargs()
MODELS_NAMES = Classifiers.models_names()

RESNET_LIST = [
    ('resnet18', 512),
    ('resnet34', 512),
    ('resnet50', 2048),
    ('resnet101', 2048),
    ('resnet152', 2048),
]

SERESNET_LIST_1 = [
    ('seresnet18', 512),
    ('seresnet34', 512),
]

SERESNET_LIST_2 = [
    ('seresnet50', 2048),
    ('seresnet101', 2048),
    ('seresnet152', 2048),
]

SERESNEXT_LIST = [
    ('seresnext50', 2048),
    ('seresnext101', 2048),
]

SENET_LIST = [
    ('senet154', 2048)
]


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """

    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        keras.backend.clear_session()
        return output

    return wrapper


def _select_names(names):
    is_full = os.environ.get('FULL_TEST', False)
    if not is_full:
        return [random.choice(names)]
    else:
        return names


def _get_img():
    """Read image for processing"""
    x = imread('./tests/data/dog.jpg').astype('float32')  # cast for keras 2.1.x compatibility
    return np.expand_dims(x, axis=0)


def _get_output_shape(model, preprocess_input=None):
    if preprocess_input is None:
        return model.output_shape
    else:
        x = _get_img()
        x = preprocess_input(x)
        return model.output_shape, model.predict(x)


@keras_test
def _test_save_load(name, input_shape=(224, 224, 3)):
    # create first model
    classifier, preprocess_input = Classifiers.get(name)
    model1 = classifier(input_shape=input_shape, weights=None)
    model1.save('model.h5')

    # load same model from file
    model2 = keras.models.load_model('model.h5', compile=False)
    os.remove('model.h5')

    x = _get_img()
    y1 = model1.predict(x)
    y2 = model2.predict(x)

    assert np.allclose(y1, y2)


@keras_test
def _test_application(name, input_shape=(224, 224, 3), last_dim=1000, label='bull_mastiff'):
    classifier, preprocess_input = Classifiers.get(name)
    model = classifier(input_shape=input_shape, weights='imagenet')

    output_shape, preds = _get_output_shape(model, preprocess_input)
    assert output_shape == (None, last_dim)

    names = [p[1] for p in decode_predictions(preds, **KWARGS)[0]]
    assert label in names[:3]


@keras_test
def _test_application_notop(name, input_shape=(None, None, 3), last_dim=1024):
    classifier, _ = Classifiers.get(name)
    model = classifier(input_shape=input_shape, weights=None, include_top=False)
    assert model.output_shape == (None, None, None, last_dim)


@keras_test
def _test_application_variable_input_channels(name, last_dim=1024):
    _test_application_notop(name, input_shape=(None, None, 1), last_dim=last_dim)
    _test_application_notop(name, input_shape=(None, None, 4), last_dim=last_dim)


@pytest.mark.parametrize('name', MODELS_NAMES)
def test_imports(name):
    data = Classifiers.get(name)
    assert data is not None


@pytest.mark.parametrize(['name', 'last_dim'], _select_names(RESNET_LIST))
def test_resnets(name, last_dim):
    _test_application(name)
    _test_application_notop(name, last_dim=last_dim)
    _test_application_variable_input_channels(name, last_dim=last_dim)


@pytest.mark.parametrize(['name', 'last_dim'], _select_names(SERESNET_LIST_1))
def test_seresnets_1(name, last_dim):
    _test_application(name)
    _test_application_notop(name, last_dim=last_dim)
    _test_application_variable_input_channels(name, last_dim=last_dim)


@pytest.mark.parametrize(['name', 'last_dim'], _select_names(SERESNET_LIST_2))
def test_seresnets_2(name, last_dim):
    _test_application(name)
    _test_application_notop(name, last_dim=last_dim)
    _test_application_variable_input_channels(name, last_dim=last_dim)


@pytest.mark.parametrize(['name', 'last_dim'], _select_names(SERESNEXT_LIST))
def test_seresnexts(name, last_dim):
    _test_application(name)
    _test_application_notop(name, last_dim=last_dim)
    _test_application_variable_input_channels(name, last_dim=last_dim)


@pytest.mark.parametrize(['name', 'last_dim'], _select_names(SENET_LIST))
def test_senets(name, last_dim):
    if not os.environ.get('TRAVIS', False):  # perform only local tests
        _test_application(name)
        _test_application_notop(name, last_dim=last_dim)
        _test_application_variable_input_channels(name, last_dim=last_dim)


def test_save_load():
    name, last_dim = SERESNEXT_LIST[0]
    _test_save_load(name)


if __name__ == '__main__':
    pytest.main([__file__])
