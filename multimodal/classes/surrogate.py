from abc import ABC, abstractmethod
import keras
import numpy as np

from common.utils import log as _log


class Surrogate(ABC):
    @abstractmethod
    def update(self, model_configurations, model_scores):
        pass
    @abstractmethod
    def predict(self, model_configuration):
        pass


class PlantSurrogate(Surrogate):

    def __init__(
        self, 
        n_epochs=50, 
        n_neurons=100, 
        batch_size=64, 
        input_shape=5,
        max_fusions=3,
        n_unique_tokens=7, # 5 layers, 2 nonlinearities
        verbose=1,
        optimizer="adam",
        loss="mse",
    ):
        _log(f"""
        ==============================================
             
        [{type(self).__name__}] initializing with params:
        
        {n_epochs=}
        {n_neurons=}
        {batch_size=}
        {input_shape=}
        {max_fusions=}
        {n_unique_tokens=}
        {verbose=}
        {optimizer=}
        {loss=}

        ==============================================
        """)

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_fusions = max_fusions
        self._model = keras.Sequential([
            keras.layers.Flatten(input_shape=[max_fusions, input_shape]),
            keras.layers.Embedding(n_unique_tokens + 1, n_neurons, mask_zero=True, input_shape=input_shape),
            keras.layers.LSTM(n_neurons, input_shape=[max_fusions, n_neurons]),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        
        self._model.compile(
            optimizer=optimizer,
            loss=loss,
        )


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
    

    def _preprocess(self, model_configurations):
        model_configurations = [np.array(conf) + 1 for conf in model_configurations]
        model_configurations = keras.preprocessing.sequence.pad_sequences(
            model_configurations, 
            padding="post",
            maxlen=self.max_fusions,
        )
        return np.array(model_configurations)