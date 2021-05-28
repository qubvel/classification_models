from tensorflow import keras
import keras_applications as ka
from .__version__ import __version__

# def get_submodules_from_kwargs(kwargs):
#     backend = kwargs.get('backend', ka._KERAS_BACKEND)
#     layers = kwargs.get('layers', ka._KERAS_LAYERS)
#     models = kwargs.get('models', ka._KERAS_MODELS)
#     utils = kwargs.get('utils', ka._KERAS_UTILS)
#     return backend, layers, models, utils

""" Modified to solve following error:
module 'keras.utils' has no attribute 'get_file'
Reference: https://www.programcreek.com/python/?CodeExample=get+submodules+from+kwargs
"""


def get_submodules_from_kwargs(kwargs):
    backend = keras.backend
    layers = keras.backend
    models = keras.models
    keras_utils = keras.utils

    return backend, layers, models, keras_utils
