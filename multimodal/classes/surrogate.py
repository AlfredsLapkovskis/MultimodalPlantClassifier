from abc import ABC, abstractmethod
import keras
import numpy as np
import os

from common.utils import log as _log
from config import Config
from .checkpointing import Checkpoint


class Surrogate(ABC):
    @abstractmethod
    def update(self, model_configurations, model_scores):
        pass
    @abstractmethod
    def predict(self, model_configuration):
        pass
    @abstractmethod
    def checkpoint(self, checkpoint: Checkpoint):
        pass


class PlantSurrogate(Surrogate):

    def __init__(
        self,
        config: Config,
        vocabulary,
        n_epochs=50, 
        n_neurons=100, 
        batch_size=64, 
        input_shape=5,
        max_fusions=4,
        iterations=5,
        verbose=1,
        optimizer="adam",
        loss="mse",
        checkpoint: Checkpoint | None=None,
    ):
        self.config = config
        self.vocabulary = np.unique(list(vocabulary))
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_fusions = max_fusions
        self.iterations = iterations
        self._lookup = keras.layers.IntegerLookup(vocabulary=self.vocabulary)

        model_path = None

        if checkpoint is not None:
            model_path = self._get_model_path(checkpoint)
            if model_path is not None and os.path.exists(model_path):
                self._model = keras.models.load_model(model_path)
            else:
                model_path = None

        if model_path is None:
            self._model = keras.Sequential([
                keras.layers.Flatten(input_shape=[max_fusions, input_shape]),
                keras.layers.Embedding(len(self.vocabulary) + 1, n_neurons, mask_zero=True, input_length=input_shape),
                keras.layers.LSTM(n_neurons, input_shape=[max_fusions, n_neurons]),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
        
            self._model.compile(
                optimizer=optimizer,
                loss=loss,
            )

        _log(f"""
        ==============================================
             
        [{type(self).__name__}] initialized with params:
        
        {vocabulary=}
        {n_epochs=}
        {n_neurons=}
        {batch_size=}
        {input_shape=}
        {max_fusions=}
        {verbose=}
        {optimizer=}
        {loss=}
        {model_path=}

        ==============================================
        """)


    def update(self, model_configurations, model_scores):
        _log(f"""
        ==============================================
             
        [{type(self).__name__}] updating with:

        {len(model_configurations)=}
        {len(model_scores)=}
             
        ==============================================        
        """)

        history = self._model.fit(
            self._preprocess(model_configurations),
            np.array(model_scores),
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
        )

        return history.history["loss"][-1]
    

    def predict(self, model_configurations):
        _log(f"""
        ==============================================

        [{type(self).__name__}] predicting with:

        {model_configurations=}

        ==============================================
        """)
        
        predictions = self._model.predict(self._preprocess(model_configurations))
        return np.reshape(predictions, -1)
    

    def checkpoint(self, checkpoint):
        if checkpoint.is_checkpointed:
            path = self._get_model_path(checkpoint)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._model.save(path)


    def _preprocess(self, model_configurations):
        model_configurations = [self._lookup(conf) for conf in model_configurations]
        model_configurations = keras.preprocessing.sequence.pad_sequences(
            model_configurations, 
            padding="post",
            maxlen=self.max_fusions,
        )
        return np.array(model_configurations)
    
    def _get_model_path(self, checkpoint: Checkpoint):
        iteration = None
        layer_idx = None

        if checkpoint.is_scored:
            iteration = checkpoint.iteration
            layer_idx = checkpoint.layer_idx
        elif checkpoint.is_checkpointed:
            if checkpoint.layer_idx == 0:
                if checkpoint.iteration > 0:
                    iteration = checkpoint.iteration - 1
                    layer_idx = self.max_fusions - 1
            else:
                iteration = checkpoint.iteration
                layer_idx = checkpoint.layer_idx - 1
        
        if iteration is not None and layer_idx is not None:
            return os.path.join(self.config.get_models_dir(), f"plant_surrogate_{iteration}_{layer_idx}.keras")

        return None
