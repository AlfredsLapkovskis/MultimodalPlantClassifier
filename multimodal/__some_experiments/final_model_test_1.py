from mfas import DefaultModelBuilder, PlantUnimodal
import pandas as pd
import numpy as np
import os
import argparse
import keras
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


def main(configuration):
    builder = DefaultModelBuilder(
        n_classes=N_CLASSES,
        fusion_sizes=[64, 64, 64],
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
                log_dir=os.path.join(config.get_log_dir(), "final_model_test", f"{index}_{datetime.now()}"),
                histogram_freq=1,
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c",
        help="configuration index",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    index = args.c

    configurations = [
        [[221,  20,   4,  20,   2], [107,  20,   4, 107,   2], [228, 221, 228, 228,   1]],
        [[221, 107, 107, 221,   2], [221, 221, 228,   4,   1], [228, 228, 221, 228,   1]],
        [[ 20,  20,   4, 107,   1], [  4, 221, 221,  20,   2], [228, 228, 221, 228,   1]],
        [[  4, 107, 228,  20,   1], [221, 221, 228,   4,   1], [228, 228, 221, 228,   1]],
        [[221, 107, 221, 107,   1], [ 20,   4, 228,  20,   2], [228, 221, 228, 228,   1]],
        [[221, 107,  20, 221,   1], [221, 221, 228,   4,   1], [228, 228, 221, 228,   1]],
        [[228,  20, 107, 107,   2], [107, 228, 228, 228,   2], [228, 221,   4,  20,   1]],
        [[221, 107,   4,  20,   2], [228,  20, 107,  20,   1], [228, 221, 228, 228,   1]],
        [[221, 228, 221, 228,   1], [107, 228, 228, 228,   2], [228, 221,   4,  20,   1]],
        [[221,  20,   4,  20,   2], [228,  20, 107, 221,   1], [228, 221, 228, 228,   1]],
    ]

    configuration = np.array(configurations[index])

    configuration = _map_alvis_to_local(configuration)

    print(f"{configuration=}")

    main(configuration)