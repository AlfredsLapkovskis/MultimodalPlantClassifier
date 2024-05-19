from abc import ABC, abstractmethod
import keras

from common.constants import IMAGE_SHAPE


class Unimodal(ABC):
    @abstractmethod
    def get_modality_name(self) -> str:
        pass
    @abstractmethod
    def build_model(self) -> keras.Model:
        pass
    @abstractmethod
    def get_fusable_layer_indices(self, fusion_idx) -> list[int]:
        pass
    @abstractmethod
    def get_layer(self, model, idx) -> keras.Layer:
        pass


class PlantUnimodal(Unimodal):
    
    _INPUT_LAYER_IDX = 0
    _OUTPUT_LAYER_IDX = 156


    def __init__(self, modality):
        self.modality = modality


    def get_modality_name(self) -> str:
        return self.modality


    def build_model(self) -> keras.Model:
        input = keras.Input(IMAGE_SHAPE, name=self.modality)

        model = keras.applications.MobileNetV3Small(
            input_shape=IMAGE_SHAPE,
            input_tensor=input,
            include_preprocessing=False,
            include_top=False,
            pooling="avg",
        )

        model.trainable = False
        model.compile()

        model.layers[0].name = self.modality
        for layer in model.layers[1:]:
            layer.name = f"{self.modality}_{layer.name}"

        return model


    def get_fusable_layer_indices(self, fusion_idx) -> list[int]:
        """
        15   <BatchNormalization name=expanded_conv_project_bn, built=True>
        24   <BatchNormalization name=expanded_conv_1_project_bn, built=True>
        33   <Add name=expanded_conv_2_add, built=True>
        48   <BatchNormalization name=expanded_conv_3_project_bn, built=True>
        63   <Add name=expanded_conv_4_add, built=True>
        78   <Add name=expanded_conv_5_add, built=True>
        92   <BatchNormalization name=expanded_conv_6_project_bn, built=True>
        107  <Add name=expanded_conv_7_add, built=True>
        122  <BatchNormalization name=expanded_conv_8_project_bn, built=True>
        137  <Add name=expanded_conv_9_add, built=True>
        152  <Add name=expanded_conv_10_add, built=True>
        """

        return [
            # input layer
            # self._INPUT_LAYER_IDX,

            # first activation
            3,

            # inverted residual blocks
            15, 
            # 24, 
            # 33, 
            # 48, 
            # 63, 
            78, 
            # 92, 
            # 107, 
            # 122, 
            # 137, 
            152,

            # output layer
            self._OUTPUT_LAYER_IDX,
        ]

    def get_layer(self, model, idx) -> keras.Layer:
        if idx == self._INPUT_LAYER_IDX:
            return model.layers[0]
        elif idx == self._OUTPUT_LAYER_IDX:
            return model.layers[self._OUTPUT_LAYER_IDX]
        else:
            layer = keras.layers.GlobalAveragePooling2D()
            layer(model.layers[idx].output)
            return layer