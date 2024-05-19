import keras
import pandas as pd
import os

from config import Config
from dataset.loading import load_unimodal_dataset
from common.constants import N_CLASSES, IMAGE_SHAPE


def run():
    config = Config("config.json")
    modality = "Flower"

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
        pooling="avg",
        include_preprocessing=False,
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=IMAGE_SHAPE)
    x = inputs
    # x = keras.layers.RandomFlip('horizontal')(x)
    # x = keras.layers.RandomRotation(0.2)(x)
    x = base_model(x, training=False)
    # x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name="a_softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=1e-4,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    pre_fine_tuning_path = os.path.join(config.get_models_dir(), "pre_fine_tuning_1.weights.h5")

    if os.path.exists(pre_fine_tuning_path):
        model.load_weights(pre_fine_tuning_path)
    else:
        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=100,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    monitor="val_loss",
                    min_delta=0.02,
                ),
                keras.callbacks.BackupAndRestore(
                    backup_dir=os.path.join(config.get_checkpoint_dir(), "pre_fine_tuning_1"),
                ),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(config.get_log_dir(), "pre_fine_tuning_1"),
                ),
            ],
        )
        os.makedirs(os.path.dirname(pre_fine_tuning_path), exist_ok=True)
        model.save_weights(pre_fine_tuning_path)

    for layer in base_model.layers[138:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=1e-5,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=1000,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            keras.callbacks.BackupAndRestore(
                backup_dir=os.path.join(config.get_checkpoint_dir(), "fine_tuning_1"),
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.get_log_dir(), "fine_tuning_1"),
            ),
            keras.callbacks.ReduceLROnPlateau(
                patience=5,
                min_lr=1e-6,
            ),
        ],
    )

    fine_tuning_path = os.path.join(config.get_models_dir(), "fine_tuning_1.weights.h5")
    os.makedirs(os.path.dirname(fine_tuning_path), exist_ok=True)

    model.save_weights(fine_tuning_path)

if __name__ == "__main__":
    run()