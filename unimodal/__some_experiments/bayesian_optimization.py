import keras
import keras_tuner
import pandas as pd
import os
from datetime import datetime
import json

from config import Config
from dataset.loading import load_unimodal_dataset
from common.utils import get_class_weight
from common.constants import N_CLASSES, IMAGE_SHAPE


#************************* BaseHyperModel *************************#

class BaseHyperModel(keras_tuner.HyperModel):

    def __init__(self, config, modality):
        self.config = config
        self.modality = modality
        self.train_df = pd.read_csv(config.get_unimodal_csv_file_path("train", modality))
        self.valid_df = pd.read_csv(config.get_unimodal_csv_file_path("validation", modality))
        self.class_weight = get_class_weight(self.train_df, N_CLASSES)

        super().__init__()

    def get_name(self):
        return None

    def get_base_model(self, model):
        return None
    
    def compile(self, model, hp, force_low_lr=False):
        pass

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [64, 128])
        
        augmentations = [aug for aug in [1, 2, 3, 4, 6, 7, 8] if hp.Boolean(f"augm_{aug}")]

        train_ds = load_unimodal_dataset(
            self.config,
            self.modality,
            "train",
            augmentations=augmentations,
            df=self.train_df,
            shuffle=True,
            batch_size=batch_size,
        )

        valid_ds = load_unimodal_dataset(
            self.config,
            self.modality,
            "validation",
            df=self.valid_df,
            batch_size=batch_size,
        )

        return model.fit(
            train_ds,
            *args,
            validation_data=valid_ds,
            class_weight=self.class_weight,
            **kwargs,
        )



#************************* MobileNetV3Small *************************#

class MobileNetHyperModel(BaseHyperModel):

    def get_name(self):
        return f"MobileNetV3Small-{self.modality}"

    def get_base_model(self, model):
        return model.get_layer("base")
    
    # TODO: CHANGE!!!!!!!!!
    def compile(self, model, hp, force_low_lr=False):
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-5 if force_low_lr else hp.Float("lr", 1e-5, 1e-3),
                weight_decay=hp.Float("weight_decay", 0, 0.1),
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def build(self, hp: keras_tuner.HyperParameters):
        base_model = keras.applications.MobileNetV3Small(
            input_shape=IMAGE_SHAPE,
            include_top=False,
            pooling="avg",
            include_preprocessing=False,
        )

        base_model.trainable = False
        base_model.name = "base"

        inputs = keras.Input(shape=IMAGE_SHAPE)
        x = base_model(inputs, training=False)

        n_layers = hp.Int("n_layers", 0, 2)

        for i in range(n_layers):
            x = keras.layers.Dropout(hp.Float(f"dropout_{i}", 0, 0.3), name=f"a_dropout_{i}")(x)
            x = keras.layers.Dense(hp.Choice(f"dense_{i}", [512, 1024, 2048]), activation="relu", name=f"a_dense_{i}")(x)

        x = keras.layers.Dropout(hp.Float("c_dropout", 0, 0.3))(x)
        outputs = keras.layers.Dense(N_CLASSES, activation="softmax", kernel_regularizer=keras.regularizers.l2(hp.Float("c_l2", 0, 0.01)))(x)

        model = keras.Model(inputs, outputs)

        self.compile(model, hp)

        return model


def train_model(config, hypermodel: BaseHyperModel, test_mode=False):
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=hypermodel,
        objective="val_loss",
        max_trials=2 if test_mode else 100,
        directory=os.path.join(config.get_trials_dir(), f"{hypermodel.get_name()}_hp-{datetime.now()}"),
    )

    tuner.search(
        epochs=1 if test_mode else 10,
        verbose=1,
        callbacks=_get_callbacks(config, hypermodel),
    )

    best_hp = tuner.get_best_hyperparameters()
    best_model = tuner.get_best_models()[0]

    try:
        json_str = json.dumps([hp.values for hp in best_hp], indent=2)
        
        dir_name = os.path.join(config.get_log_dir(), "bayesian")
        os.makedirs(dir_name, exist_ok=True)

        with open(os.path.join(dir_name, f"{hypermodel.get_name()}.json"), "w") as f:
            f.write(json_str)
    except:
        pass

    # hypermodel.get_base_model(best_model).trainable = not test_mode
    # hypermodel.compile(best_model, best_hp, force_low_lr=True)

    # hypermodel.fit(
    #     best_hp,
    #     best_model,
    #     epochs=1 if test_mode else 1000,
    #     verbose=1,
    #     callbacks=_get_callbacks(config, hypermodel, fine_tuned=True),
    # )
    
    model_path = os.path.join(config.get_models_dir(), f"{hypermodel.get_name()}-{datetime.now()}.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    best_model.save(model_path)


def _get_callbacks(config, hypermodel: BaseHyperModel, fine_tuned=False):
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.get_log_dir(), f"{hypermodel.get_name()}{'_final' if fine_tuned else ''}-{datetime.now()}"),
            histogram_freq=1,
        ),
    ]

    if fine_tuned:
        callbacks.append(keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(config.get_checkpoint_dir(), hypermodel.get_name())
        ))

    return callbacks


if __name__ == "__main__":
    config = Config("config.json")
    train_model(config, MobileNetHyperModel(config, "Flower"))