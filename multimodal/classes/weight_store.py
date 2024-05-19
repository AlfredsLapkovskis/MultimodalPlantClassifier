from abc import ABC, abstractmethod
import os
import json
import numpy as np


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
        self.pretty = pretty
        self._json_path = json_path
        self._weights = {}

        if merge_from_dir is not None and os.path.exists(merge_from_dir):
            for path in os.listdir(merge_from_dir):
                if path.endswith(".json"):
                    with open(os.path.join(merge_from_dir, path), "r") as f:
                        self._weights.update(json.loads(f.read()))

            if json_path is not None and self._weights:
                self._save_to_json()

        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                self._weights = json.loads(f.read())


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

            self._save_to_json()


    def _save_to_json(self):
        dir = os.path.dirname(self._json_path)
        os.makedirs(dir, exist_ok=True)
        with open(self._json_path, "w") as f:
            f.write(json.dumps(self._weights, indent=2 if self.pretty else None))


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
    

    def __str__(self) -> str:
        return f"{type(self).__name__}({len(self._weights)} weights)"