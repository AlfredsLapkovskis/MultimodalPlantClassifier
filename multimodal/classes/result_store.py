from abc import ABC, abstractmethod
import os
import json
import numpy as np


class ResultStore(ABC):
    @abstractmethod
    def add_result(self, configuration, iteration, layer_idx, score):
        pass
    @abstractmethod
    def contains(self, configuration, iteration, layer_idx):
        pass
    @abstractmethod
    def get_results(self):
        pass
    @abstractmethod
    def get_best(self, count):
        pass


class JsonResultStore(ResultStore):

    def __init__(self, json_path, merge_from_dir=None, pretty=False):
        self.pretty = pretty
        self._json_path = json_path
        self._results = {}

        if merge_from_dir is not None and os.path.exists(merge_from_dir):
            for path in os.listdir(merge_from_dir):
                if path.endswith(".json"):
                    with open(os.path.join(merge_from_dir, path), "r") as f:
                        json_dict = json.loads(f.read())
                        self._add_from_dict(json_dict)

            if json_path is not None and self._results:
                self._save_to_json()
        
        elif os.path.exists(json_path):
            with open(json_path, "r") as f:
                json_dict = json.loads(f.read())
                self._add_from_dict(json_dict)


    def add_result(self, configuration, iteration, layer_idx, score):
        self._add_result(configuration, iteration, layer_idx, score, True)

    
    def get_results(self):
        configurations = []
        scores = []

        for x, y, _, _ in self._results.values():
            configurations.append(x)
            scores.append(y)

        return configurations, scores
    

    def contains(self, configuration, iteration, layer_idx):
        if not self._results:
            return False

        key = str(configuration)

        if key in self._results:
            result = self._results[key]
            return result[2] == iteration and result[3] == layer_idx
        
        return False
    

    def get_best(self, count):
        values = [(v[0], v[1]) for v in self._results.values()]
        values = sorted(values, key=lambda x: x[1], reverse=True)

        return values[:min(count, len(values))]
    

    def _add_from_dict(self, json_dict):
        for value in json_dict.values():
            configuration = np.array(value[0])
            score = value[1]
            iteration = value[2]
            layer_idx = value[3]

            self._add_result(configuration, iteration, layer_idx, score, False)
    

    def _add_result(self, configuration, iteration, layer_idx, score, save_to_json):
        key = str(configuration)
        best_score = max(self._results[key][1], score) if (key in self._results) else score

        self._results[key] = [configuration, best_score, iteration, layer_idx]

        if save_to_json:
            self._save_to_json()
    
    
    def _save_to_json(self):
        json_dict = {
            key: [result[0].tolist(), result[1], result[2], result[3]]
            for key, result in self._results.items()
        }

        dir = os.path.dirname(self._json_path)
        os.makedirs(dir, exist_ok=True)
        with open(self._json_path, "w") as f:
            f.write(json.dumps(json_dict, indent=2 if self.pretty else None))

    
    def __str__(self) -> str:
        return f"{type(self).__name__}({len(self._results)} results)"