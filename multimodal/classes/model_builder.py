from abc import ABC, abstractmethod
import keras

from .unimodal import Unimodal
from .nonlinearities import *


class ModelBuilder(ABC):
    @abstractmethod
    def build_model(self, configuration, unimodals: list[Unimodal]):
        pass


class DefaultModelBuilder(ModelBuilder):

    def __init__(self, n_classes, fusion_sizes, dropouts=[], l2=[], batch_norm=[], weight_store=None):
        self.n_classes = n_classes
        self.fusion_sizes = fusion_sizes
        self.weight_store = weight_store
        self.dropouts = dropouts
        self.batch_norm = batch_norm
        self.l2 = l2


    def build_model(self, configuration, unimodals):
        modalities = [u.get_modality_name() for u in unimodals]
        unimodal_models = [u.build_model() for u in unimodals]
        inputs = [m.input[0] for m in unimodal_models]

        last_fusion_layer = None

        for fusion_idx, layer_conf in enumerate(configuration):
            layers = []
            nonlinearity = layer_conf[-1]

            if nonlinearity == NONLINEARITY_RELU:
                nonlinearity = "relu"
            elif nonlinearity == NONLINEARITY_SIGMOID:
                nonlinearity = "sigmoid"
            else:
                assert False

            fusion_size = self.fusion_sizes[fusion_idx]

            for unimodal_idx, layer_idx in enumerate(layer_conf[:-1]):
                layer = unimodals[unimodal_idx].get_layer(unimodal_models[unimodal_idx], layer_idx)
                    
                if len(layer.output.shape) > 2:
                    flatten = keras.layers.Flatten()
                    flatten(layer.output)
                    layer = flatten

                layers.append(layer)

            if last_fusion_layer is not None:
                layers.append(last_fusion_layer)

            x = keras.layers.Concatenate()([l.output for l in layers])

            if len(self.dropouts) > fusion_idx:
                x = keras.layers.Dropout(self.dropouts[fusion_idx])(x)

            last_fusion_layer = keras.layers.Dense(
                fusion_size,
                kernel_regularizer=keras.regularizers.l2(self.l2[fusion_idx]) if len(self.l2) > fusion_idx else None,
                name=self.weight_store.get_name_for_layer(fusion_idx) if self.weight_store else None,
            )
            last_fusion_layer(x)

            if len(self.batch_norm) > fusion_idx and self.batch_norm[fusion_idx]:
                x = last_fusion_layer.output
                last_fusion_layer = keras.layers.BatchNormalization()
                last_fusion_layer(x)

            x = last_fusion_layer.output
            last_fusion_layer = keras.layers.Activation(nonlinearity)
            last_fusion_layer(x)

        if len(self.dropouts) > len(configuration):
            x = last_fusion_layer.output
            last_fusion_layer = keras.layers.Dropout(self.dropouts[len(configuration)])
            last_fusion_layer(x)

        classifier = keras.layers.Dense(
            self.n_classes, 
            activation="softmax", 
            name="classifier",
            kernel_regularizer=keras.regularizers.l2(self.l2[len(configuration)]) if len(self.l2) > len(configuration) else None,
        )(last_fusion_layer.output)

        return keras.Model(
            inputs=dict(zip(modalities, inputs)),
            outputs=classifier
        )