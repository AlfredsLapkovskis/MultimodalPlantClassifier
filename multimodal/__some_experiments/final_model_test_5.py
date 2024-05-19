from mfas import DefaultModelBuilder, PlantUnimodal
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import keras
import argparse
from datetime import datetime

from config import Config
from dataset.loading import load_multimodal_dataset
from common.utils import get_class_weight
from common.constants import N_CLASSES, MODALITY_LIST


_ALVIS_TO_LOCAL = {
    # nonlinearities
    1: 1,
    2: 2,

    # layers
    4: 3,
    20: 15,
    107: 78,
    221: 152,
    228: 156,
}

def _map_alvis_to_local(conf):
    shape = conf.shape
    return np.reshape([_ALVIS_TO_LOCAL[v] for v in np.reshape(conf, -1)], shape)


def main(configuration, case):
    description = "default"
    if case == 1:
        description = "class_dropout"
    elif case == 2:
        description = "label_smoothing"
    elif case == 3:
        description = "brightness"
    elif case == 4:
        description = "flip"
    elif case == 5:
        description = "multimodal_dropout"
    elif case == 6:
        description = "no_lrs"
    elif case == 7:
        description = "wd"
    elif case == 8:
        description = "rotation"
    elif case == 9:
        description = "batch_64"

    builder = DefaultModelBuilder(
        n_classes=N_CLASSES,
        fusion_sizes=[256, 256, 256],
        dropouts=[0.1, 0.2, 0.2, (0.1 if case == 1 else 0.0)],
    )

    config = Config("config.json")

    unimodals = [PlantUnimodal(modality) for modality in MODALITY_LIST]

    train_df = pd.read_csv(config.get_multimodal_csv_file_path("train"))
    valid_df = pd.read_csv(config.get_multimodal_csv_file_path("validation"))

    augmentations = []
    if case == 3:
        augmentations.append(1)
    elif case == 4:
        augmentations.extend([3, 4])
    elif case == 8:
        augmentations.append(8)

    batch_size = 128
    if case == 9:
        batch_size = 64

    train_ds = load_multimodal_dataset(config, "train", MODALITY_LIST, augmentations, shuffle=True, batch_size=batch_size, dropout=0.25 if case == 5 else 0.125, df=train_df)
    valid_ds = load_multimodal_dataset(config, "validation", MODALITY_LIST, [], shuffle=False, batch_size=batch_size, df=valid_df)

    model = builder.build_model(configuration, unimodals)

    def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), N_CLASSES)
        return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.05)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001 if case == 6 else keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=200,
                decay_rate=0.95,
            ),
            weight_decay=0.01 if case == 7 else None,
        ),
        loss=sparse_categorical_crossentropy_with_smoothing if case == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=valid_ds,
        class_weight=get_class_weight(train_df, N_CLASSES),
        verbose=1,
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.get_log_dir(), "final_model_test_5", f"{description}_{datetime.now()}"),
                histogram_freq=1,
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c",
        type=int,
        required=True,
    )

    configuration = np.array([[221,  20,   4,  20,   2], [107,  20,   4, 107,   2], [228, 221, 228, 228,   1]])

    configuration = _map_alvis_to_local(configuration)

    print(f"{configuration=}")

    main(configuration, parser.parse_args().c)