from abc import ABC, abstractmethod
import numpy as np
from common.json_io import JsonIO


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
    def get_result(self, configuration, iteration, layer_idx):
        pass
    @abstractmethod
    def get_best(self, count):
        pass


class JsonResultStore(ResultStore):


    _IDX_CONFIGURATION = 0
    _IDX_SCORE = 1
    _IDX_ITERATION = 2
    _IDX_LAYER = 3


    def __init__(self, json_path, merge_from_dir=None, pretty=False):
        self._json_io = JsonIO(
            json_path=json_path,
            merge_from_dir=merge_from_dir,
            pretty=pretty,
            backup_enabled=True,
        )
        self.pretty = pretty
        self._json_path = json_path
        self._results = dict()
        
        self._add_from_dict(self._json_io.read() or dict())


    def add_result(self, configuration, iteration, layer_idx, score):
        self._add_result(configuration, iteration, layer_idx, score, True)

    
    def get_results(self):
        configurations = []
        scores = []

        for x, y, _, _ in self._results.values():
            configurations.append(x)
            scores.append(y)

        return configurations, scores
    

    def get_result(self, configuration, iteration, layer_idx):
        if not self._results:
            return None
        
        key = str(configuration)
        if key in self._results:
            result = self._results[key]
            if result[self._IDX_ITERATION] == iteration and result[self._IDX_LAYER] == layer_idx:
                return result[self._IDX_SCORE]
        
        return None
    

    def contains(self, configuration, iteration, layer_idx):
        return self.get_result(configuration, iteration, layer_idx) != None
    

    def get_best(self, count):
        values = [(v[0], v[1]) for v in self._results.values()]
        values = sorted(values, key=lambda x: x[1], reverse=True)

        return values[:min(count, len(values))]
    

    def _add_from_dict(self, json_dict):
        for value in json_dict.values():
            configuration = np.array(value[self._IDX_CONFIGURATION])
            score = value[self._IDX_SCORE]
            iteration = value[self._IDX_ITERATION]
            layer_idx = value[self._IDX_LAYER]

            self._add_result(configuration, iteration, layer_idx, score, False)
    

    def _add_result(self, configuration, iteration, layer_idx, score, save_to_json):
        key = str(configuration)
        best_score = max(self._results[key][self._IDX_SCORE], score) if (key in self._results) else score

        self._results[key] = [configuration, best_score, iteration, layer_idx]

        if save_to_json:
            self._save_to_json()
    
    
    def _save_to_json(self):
        self._json_io.write({
            key: [result[self._IDX_CONFIGURATION].tolist(), result[self._IDX_SCORE], result[self._IDX_ITERATION], result[self._IDX_LAYER]]
            for key, result in self._results.items()
        })
