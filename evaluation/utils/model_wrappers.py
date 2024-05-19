import numpy as np
import keras
import tensorflow as tf
from abc import ABC, abstractmethod

from common.constants import IMAGE_SHAPE


class Model(ABC):
    @abstractmethod
    def predict(self, dataset):
        pass


class UnimodalModel(Model):

    def __init__(self, path):
        self._model = keras.saving.load_model(path)
        self._model.summary()
        self._model.compile()

    def predict(self, dataset):
        return self._model.predict(dataset)
    

class LateFusionModel(Model):

    def __init__(self, modalities, paths):
        assert len(modalities) > 0
        assert len(modalities) == len(paths)

        self._models = {m: keras.saving.load_model(p) for m, p in zip(modalities, paths)}

    def predict(self, dataset: tf.data.Dataset):
        predictions = {}
        for modality, model in self._models.items():
            ds = dataset.map(lambda x, y: (x[modality], y))
            predictions[modality] = model.predict(ds)
        dataset = dataset.unbatch()

        averaged_predictions = []

        for idx, instance in enumerate(dataset):
            pred = []
            for modality, image in instance[0].items():
                if np.any(image != 0):
                    pred.append(predictions[modality][idx])
            averaged_predictions.append(np.mean(pred, axis=0) if len(pred) > 0 else 0)

        return averaged_predictions
    

class MultimodalModel(Model):
    
    def __init__(self, path):
        self._model = tf.saved_model.load(path).signatures["serving_default"]

    def predict(self, dataset):
        y_pred_proba = []
        for i, batch in enumerate(dataset):
            y_pred_proba.extend(self._model(**batch[0])["classifier"])
            print(f"\rFinished batch {i + 1}", end="")
        print()
        return np.array(y_pred_proba)