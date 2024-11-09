from abc import ABC, abstractmethod
import keras


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
    def get_layer(self, model, idx) -> keras.layers.Layer:
        pass


class PlantUnimodal(Unimodal):
    
    _INPUT_LAYER_IDX = 0
    _OUTPUT_LAYER_IDX = -1


    def __init__(self, modality, model_path):
        self.modality = modality
        self.model_path = model_path


    def get_modality_name(self) -> str:
        return self.modality


    def build_model(self) -> keras.Model:
        custom_objects = {"loss": None}

        model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
        
        # Unique names are necessary for fusion; unfortunately, public property .name is readonly.
        model.layers[0]._name = self.modality
        for layer in model.layers[1:]:
            layer._name = f"{self.modality}_{layer.name}"

        model.trainable = False
        model.compile()

        return model


    def get_fusable_layer_indices(self, fusion_idx) -> list[int]:
        """
        20    expanded_conv/project/BatchNorm    <keras.layers.normalization.batch_normalization.BatchNormalization object at 0x7fe9177659f0>
        29    expanded_conv_1/project/BatchNorm    <keras.layers.normalization.batch_normalization.BatchNormalization object at 0x7fe90c0e1810>
        38    expanded_conv_2/Add    <keras.layers.merging.add.Add object at 0x7fe9177e9b40>
        61    expanded_conv_3/project/BatchNorm    <keras.layers.normalization.batch_normalization.BatchNormalization object at 0x7fe90c0c4190>
        84    expanded_conv_4/Add    <keras.layers.merging.add.Add object at 0x7fe90c1744c0>
        107    expanded_conv_5/Add    <keras.layers.merging.add.Add object at 0x7fea54abb970>
        129    expanded_conv_6/project/BatchNorm    <keras.layers.normalization.batch_normalization.BatchNormalization object at 0x7fe900557250>
        152    expanded_conv_7/Add    <keras.layers.merging.add.Add object at 0x7fe900597c70>
        175    expanded_conv_8/project/BatchNorm    <keras.layers.normalization.batch_normalization.BatchNormalization object at 0x7fe9005f3760>
        198    expanded_conv_9/Add    <keras.layers.merging.add.Add object at 0x7fe900457370>
        221    expanded_conv_10/Add    <keras.layers.merging.add.Add object at 0x7fe9004b6410>
        """

        return [
            # input layer
            # self._INPUT_LAYER_IDX,

            # first activation
            4,

            # inverted residual blocks
            20, 
            # 29, 
            # 38, 
            # 61, 
            # 84, 
            107, 
            # 129, 
            # 152, 
            # 175, 
            # 198, 
            221,

            # dense layer
            229,

            # output layer
            self._OUTPUT_LAYER_IDX,
        ]

    def get_layer(self, model, idx) -> keras.layers.Layer:
        layer = model.layers[idx]
        shape_len = len(layer.output.shape)

        if shape_len > 2:
            avgpool = keras.layers.GlobalAveragePooling2D() if shape_len > 3 else keras.layers.GlobalAveragePooling1D()
            avgpool(layer.output)
            return avgpool
        return layer
