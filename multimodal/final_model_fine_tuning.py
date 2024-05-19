import pandas as pd
import numpy as np
import os
import tensorflow as tf
import keras

from config import Config
from dataset.loading import load_multimodal_dataset
from common.utils import get_class_weight
from common.constants import N_CLASSES, MODALITY_LIST
from .classes.unimodal import PlantUnimodal
from .classes.model_builder import DefaultModelBuilder


def main(configuration):
    builder = DefaultModelBuilder(
        n_classes=N_CLASSES,
        fusion_sizes=[512] * 3,
        dropouts=[0.1, 0.2, 0.2, 0.1],
        l2=[0.00001] * 4,
        batch_norm=[True] * 3,
    )

    config = Config("config.json")

    unimodals = [PlantUnimodal(modality) for modality in MODALITY_LIST]

    train_df = pd.read_csv(config.get_multimodal_csv_file_path("train"))
    valid_df = pd.read_csv(config.get_multimodal_csv_file_path("validation"))

    train_ds = load_multimodal_dataset(config, "train", MODALITY_LIST, [], shuffle=True, batch_size=128, dropout=0.125, df=train_df)
    valid_ds = load_multimodal_dataset(config, "validation", MODALITY_LIST, [], shuffle=False, batch_size=128, df=valid_df)

    model = builder.build_model(configuration, unimodals)

    def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), N_CLASSES)
        return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.02)

    """
    Total params: 6,183,548 (23.59 MB)
    Trainable params: 2,481,596 (9.47 MB)
    Non-trainable params: 3,701,952 (14.12 MB)
    """
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=200,
                decay_rate=0.95,
            ),
            weight_decay=0.01,
        ),
        loss=sparse_categorical_crossentropy_with_smoothing,
        metrics=["accuracy"],
    )

    weights_path = os.path.join(config.get_models_dir(), "final", f"mm_pre.weights.h5")

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        model.fit(
            train_ds,
            validation_data=valid_ds,
            class_weight=get_class_weight(train_df, N_CLASSES),
            verbose=1,
            epochs=100,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                ),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(config.get_log_dir(), "mm_final_fine_tuning", f"pre_fine_tuning"),
                    histogram_freq=1,
                ),
                keras.callbacks.BackupAndRestore(
                    backup_dir=os.path.join(config.get_checkpoint_dir(), "mm_final_pre"),
                    delete_checkpoint=False,
                ),
            ],
        )

        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        model.save_weights(weights_path)

    for layer in model.layers:
        layer.trainable = True

    weights_path = os.path.join(config.get_models_dir(), "final", f"mm_final.weights.h5")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=200,
                decay_rate=0.95,
            ),
            weight_decay=0.01,
        ),
        loss=sparse_categorical_crossentropy_with_smoothing,
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        class_weight=get_class_weight(train_df, N_CLASSES),
        verbose=1,
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.get_log_dir(), "mm_final_fine_tuning", f"fine_tuning"),
                histogram_freq=1,
            ),
            keras.callbacks.BackupAndRestore(
                backup_dir=os.path.join(config.get_checkpoint_dir(), "mm_final"),
                delete_checkpoint=False,
            ),
        ],
    )

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)
    

if __name__ == "__main__":
    configuration = np.array([
        [152,  15,   3,  15,   2], 
        [78,  15,   3, 78,   2], 
        [156, 152, 156, 156,   1],
    ])

    print(f"{configuration=}")

    main(configuration)