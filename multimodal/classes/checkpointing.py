import numpy as np
from abc import ABC, abstractmethod
from common.json_io import JsonIO


class Checkpoint:
    def __init__(
        self,
        iteration: int | None=None, 
        layer_idx: int | None=None, 
        sampled_configurations: list[np.array] | None=None, 
        scores: list[float] | None=None,
    ):
        assert((iteration is None) == (layer_idx is None) == (sampled_configurations is None))

        self.is_checkpointed = iteration is not None and layer_idx is not None and sampled_configurations is not None
        self.is_scored = isinstance(scores, list) and len(scores) > 0

        assert(not self.is_scored or self.is_checkpointed)

        self.iteration = iteration
        self.layer_idx = layer_idx
        self.sampled_configurations = sampled_configurations
        self.scores = scores


class Checkpointer(ABC):
    @abstractmethod
    def get_checkpoint(self) -> Checkpoint:
        return Checkpoint()
    @abstractmethod
    def set_checkpoint(self, checkpoint: Checkpoint):
        pass


class DummyCheckpointer(Checkpointer):
    def get_checkpoint(self):
        pass
    def set_checkpoint(self, checkpoint: Checkpoint):
        pass


class JsonCheckpointer(Checkpointer):

    def __init__(self, json_path, pretty=False):
        self._json_io = JsonIO(
            json_path=json_path,
            pretty=pretty,
            backup_enabled=True,
        )
        self._checkpoint = self._deserialize(self._json_io.read())


    def get_checkpoint(self):
        return self._checkpoint


    def set_checkpoint(self, checkpoint: Checkpoint):
        self._checkpoint = checkpoint
        self._save_to_json()


    def _deserialize(self, json_dict):
        return Checkpoint(
            iteration=json_dict["iteration"],
            layer_idx=json_dict["layer_idx"],
            sampled_configurations=[np.array(c) for c in json_dict["sampled_configurations"]],
            scores=json_dict["scores"],
        ) if json_dict else Checkpoint()


    def _save_to_json(self):
        self._json_io.write({
            "iteration": self._checkpoint.iteration,
            "layer_idx": self._checkpoint.layer_idx,
            "sampled_configurations": [c.tolist() for c in self._checkpoint.sampled_configurations],
            "scores": self._checkpoint.scores,
        })
