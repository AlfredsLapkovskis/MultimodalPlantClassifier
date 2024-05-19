from mfas import DefaultModelBuilder, PlantUnimodal
import pandas as pd
import numpy as np
import os
import keras
import argparse
from datetime import datetime
from itertools import product

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


def main(configuration, dropouts):
    builder = DefaultModelBuilder(
        n_classes=N_CLASSES,
        fusion_sizes=[256, 256, 256],
        dropouts=dropouts,
    )

    config = Config("config.json")

    unimodals = [PlantUnimodal(modality) for modality in MODALITY_LIST]

    train_df = pd.read_csv(config.get_multimodal_csv_file_path("train"))
    valid_df = pd.read_csv(config.get_multimodal_csv_file_path("validation"))

    train_ds = load_multimodal_dataset(config, "train", MODALITY_LIST, [], shuffle=True, batch_size=128, dropout=0.125, df=train_df)
    valid_ds = load_multimodal_dataset(config, "validation", MODALITY_LIST, [], shuffle=False, batch_size=128, df=valid_df)

    model = builder.build_model(configuration, unimodals)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=200,
                decay_rate=0.95,
            ),
        ),
        loss="sparse_categorical_crossentropy",
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
                log_dir=os.path.join(config.get_log_dir(), "final_model_test_3", f"{dropouts[0]}x{dropouts[1]}x{dropouts[2]}_{datetime.now()}"),
                histogram_freq=1,
            ),
        ],
    )


if __name__ == "__main__":
    possible_dropouts = [0., 0.1, 0.2]
    dropouts = product(possible_dropouts, possible_dropouts, possible_dropouts)
    dropouts = [d for d in dropouts if d[0] <= d[1] and d[1] <= d[2]]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c",
        help="neuron configuration index",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    dropouts = dropouts[args.c]

    configuration = np.array([[221,  20,   4,  20,   2], [107,  20,   4, 107,   2], [228, 221, 228, 228,   1]])

    configuration = _map_alvis_to_local(configuration)

    print(f"{configuration=}")

    main(configuration, dropouts)