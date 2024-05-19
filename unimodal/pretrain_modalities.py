import keras
import tensorflow as tf
import pandas as pd
from datetime import datetime
import os

from config import Config
from dataset.loading import load_unimodal_dataset
from common.utils import get_class_weight
from common.constants import N_CLASSES, IMAGE_SHAPE, MODALITY_LIST


def run():
    config = Config("config.json")

    for modality in MODALITY_LIST:
        train_df = pd.read_csv(config.get_unimodal_csv_file_path("train", modality))
        valid_df = pd.read_csv(config.get_unimodal_csv_file_path("validation", modality))

        train_ds = load_unimodal_dataset(
            config,
            modality,
            "train",
            df=train_df,
            augmentations=[],
            shuffle=True,
            batch_size=128,
        )
        valid_ds = load_unimodal_dataset(
            config,
            modality,
            "validation",
            df=valid_df,
            batch_size=128,
        )

        base_model = keras.applications.MobileNetV3Small(
            input_shape=IMAGE_SHAPE,
            include_top=False,
            include_preprocessing=False,
            pooling="avg",
        )
        base_model.trainable = False

        inputs = keras.Input(shape=IMAGE_SHAPE)
        x = base_model(inputs, training=False)
        outputs = keras.layers.Dense(
            N_CLASSES,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l1_l2(0.00001, 0.00001),
        )(x)

        model = keras.Model(inputs, outputs)

        def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), N_CLASSES)
            return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=300,
                    decay_rate=0.95,
                ),
            ),
            loss=sparse_categorical_crossentropy_with_smoothing,
            metrics=[
                "accuracy",
            ]
        )

        dt_str = f"{datetime.now()}"

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=1000,
            verbose=1,
            class_weight=get_class_weight(train_df, N_CLASSES),
            callbacks=[
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(config.get_log_dir(), "modalities", f"{modality}_{dt_str}"),
                    histogram_freq=1,
                ),
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.BackupAndRestore(
                    backup_dir=os.path.join(config.get_checkpoint_dir(), "modalities", f"{modality}.weights.h5"),
                    delete_checkpoint=False,
                ),
            ],
        )

        weights_path = os.path.join(config.get_models_dir(), "modalities", f"{modality}_{dt_str}.weights.h5")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        model.save_weights(weights_path)


if __name__ == "__main__":
    run()
