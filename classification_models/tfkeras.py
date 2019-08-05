import tensorflow.keras as tfkeras
from .models_factory import ModelsFactory


class TFKerasModelsFactory(ModelsFactory):

    @staticmethod
    def get_kwargs():
        return {
            'backend': tfkeras.backend,
            'layers': tfkeras.layers,
            'models': tfkeras.models,
            'utils': tfkeras.utils,
        }


Classifiers = TFKerasModelsFactory()
