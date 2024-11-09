from abc import ABC, abstractmethod
import numpy as np
from common.json_io import JsonIO


class WeightStore(ABC):
    @abstractmethod
    def load_weights(self, model, configuration):
        pass
    @abstractmethod
    def save_weights(self, model, configuration):
        pass
    @abstractmethod
    def get_name_for_layer(self, layer_idx):
        pass
    @abstractmethod
    def is_fusion_layer(self, layer):
        pass


class JsonWeightStore(WeightStore):

    def __init__(self, json_path, merge_from_dir=None, pretty=False):
        self._json_io = JsonIO(
            json_path=json_path,
            merge_from_dir=merge_from_dir,
            pretty=pretty,
            backup_enabled=True,
        )

        self._weights = self._json_io.read() or dict()


    def load_weights(self, model, configuration):
        for layer_idx, layer_conf in enumerate(configuration):
            layer = model.get_layer(self.get_name_for_layer(layer_idx))
            key = self._get_layer_weights_key(layer, layer_conf, layer_idx)

            if key in self._weights:
                layer.set_weights([np.array(w) for w in self._weights[key]])
    

    def save_weights(self, model, configuration):
        for layer_idx, layer_conf in enumerate(configuration):
            layer = model.get_layer(self.get_name_for_layer(layer_idx))
            key = self._get_layer_weights_key(layer, layer_conf, layer_idx)
        
            self._weights[key] = [w.tolist() for w in layer.get_weights()]

        self._json_io.write(self._weights)


    def get_name_for_layer(self, layer_idx):
        return f"fusion_{layer_idx}"
    

    def is_fusion_layer(self, layer):
        return layer.name.startswith("fusion_")
    
    
    def _get_layer_weights_key(self, layer, layer_conf, layer_idx):
        activation = layer_conf[-1]

        def get_size(tensor):
            size = tensor.shape[1] # skip batch size
            for s in tensor.shape[2:]:
                size *= s
            return size

        return f"{layer_idx}_{get_size(layer.input)}_{get_size(layer.output)}_{activation}"
