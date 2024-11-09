import keras
import tensorflow as tf
import os
from tensorboard.plugins.hparams import api as hp
import argparse

from config import Config
from dataset.loading import *
from common.utils import *
from common.constants import N_CLASSES, IMAGE_SHAPE
from common.save_mode import SaveMode
from .experiment import Experiment


ensure_no_file_locking()


def run(config: Config, experiment: Experiment, save_mode: SaveMode):
    train_ds, valid_ds = _load_datasets(config, experiment)
    train_labels = _load_train_labels(config, experiment)

    model = _build(experiment)

    _compile_fit(model, experiment, train_ds, valid_ds, train_labels)
    _save_model_if_needed(model, experiment, save_mode)

    if experiment.fine_tune:
        _unfreeze_for_fine_tuning(model, experiment)
        _compile_fit(model, experiment, train_ds, valid_ds, train_labels, fine_tuning=True)
        _save_model_if_needed(model, experiment, save_mode, True)


def _build(experiment: Experiment):
    inputs = keras.Input(shape=IMAGE_SHAPE)
    
    base_model = keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        input_tensor=inputs,
        include_preprocessing=False,
        pooling="avg",
    )
    base_model.trainable = False

    base_model(inputs, training=False)

    x = base_model.output # ensure that base_model is unfolded
    
    if experiment.intermediate_layer:
        x = _build_intermediate_layer(
            x, 
            experiment.intermediate_neurons, 
            experiment.intermediate_activation, 
            experiment.intermediate_dropout, 
            experiment.intermediate_initializer, 
            experiment.intermediate_l1, 
            experiment.intermediate_l2
        )
    if experiment.intermediate_layer_2:
        x = _build_intermediate_layer(
            x, 
            experiment.intermediate_neurons_2, 
            experiment.intermediate_activation_2, 
            experiment.intermediate_dropout_2, 
            experiment.intermediate_initializer_2, 
            experiment.intermediate_2_l1, 
            experiment.intermediate_2_l2
        )
    if experiment.classifier_dropout > 0:
        x = keras.layers.Dropout(experiment.classifier_dropout)(x)

    outputs = keras.layers.Dense(
        N_CLASSES,
        activation="softmax",
        name="classifier",
        kernel_regularizer=keras.regularizers.l1_l2(experiment.classifier_l1, experiment.classifier_l2),
    )(x)

    return keras.Model(inputs, outputs)


def _compile_fit(model: keras.Model, experiment: Experiment, train_ds, valid_ds, train_labels, fine_tuning=False):
    if experiment.focal_loss:
        loss = get_categorical_focal_crossentropy(train_labels, N_CLASSES, experiment.focal_loss_gamma)
        class_weight = None
    else:
        loss = get_categorical_crossentropy_with_smoothing(experiment.label_smoothing)
        class_weight = get_sklearn_class_weight(train_labels, N_CLASSES) if experiment.sklearn_class_weight else get_class_weight(train_labels, N_CLASSES)

    if fine_tuning:
        lr = experiment.fine_tune_lr
        decay = experiment.fine_tune_decay
        decay_steps = experiment.fine_tune_decay_steps
        decay_rate = experiment.fine_tune_decay_rate
        epochs = experiment.fine_tune_epochs
        es_patience = experiment.fine_tune_patience
        es_start_epoch = experiment.fine_tune_early_stopping_start_epoch
    else:
        lr = experiment.lr
        decay = experiment.decay
        decay_steps = experiment.decay_steps
        decay_rate = experiment.decay_rate
        epochs = experiment.epochs
        es_patience = experiment.patience
        es_start_epoch = experiment.early_stopping_start_epoch

    trial_id = _get_trial_id(experiment, fine_tuning)
    log_dir = _get_log_dir(experiment, fine_tuning)
    hparams = experiment.build_hparams(fine_tuning)

    model.compile(
        optimizer=keras.optimizers.Adam(
            weight_decay=experiment.weight_decay,
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
            ) if decay else lr,
        ),
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.F1Score(
                average="macro",
            ),
        ]
    )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
        hp.KerasCallback(log_dir, hparams, trial_id=trial_id),
    ]

    if valid_ds is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            patience=es_patience,
            start_from_epoch=es_start_epoch,
            restore_best_weights=True,
        ))

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        verbose=experiment.verbose,
        class_weight=class_weight,
        callbacks=callbacks,
    )


def _load_datasets(config: Config, experiment: Experiment):
    def encode_labels(labels):
        labels = tf.one_hot(labels, N_CLASSES)
        return tf.cast(labels, dtype=tf.float32)

    train_ds = load_unimodal_dataset(
        config,
        experiment.modality,
        ["train", "validation"] if experiment.merge_validation_set else "train",
        augmentations=_get_augmentations(experiment),
        shuffle=True,
        batch_size=experiment.batch_size,
    ).map(lambda x, y: (x, encode_labels(y)), num_parallel_calls=tf.data.AUTOTUNE)

    valid_ds = None
    if not experiment.merge_validation_set:
        valid_ds = load_unimodal_dataset(
            config,
            experiment.modality,
            "validation",
            batch_size=experiment.batch_size,
        ).map(lambda x, y: (x, encode_labels(y)), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, valid_ds


def _load_train_labels(config: Config, experiment: Experiment):
    return load_unimodal_labels(
        config, 
        experiment.modality, 
        ["train", "validation"] if experiment.merge_validation_set else "train",
    )


def _build_intermediate_layer(x, n_neurons, activation, dropout, initializer, l1, l2):
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(
        n_neurons, 
        use_bias=False, 
        kernel_initializer=initializer, 
        kernel_regularizer=keras.regularizers.l1_l2(l1, l2)
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    return keras.layers.Activation(activation=activation)(x)


def _save_model_if_needed(model: keras.Model, experiment: Experiment, save_mode: SaveMode, fine_tuning=False):
    model_dir = _get_model_dir(experiment)
    model_path = os.path.join(model_dir, f"{experiment.modality}{'_ft' if fine_tuning else ''}.keras")

    save_trained = not fine_tuning and save_mode.value >= SaveMode.TRAINED.value
    save_fine_tuned = fine_tuning and save_mode.value >= SaveMode.TRAINED_AND_FINE_TUNED.value

    if save_trained or save_fine_tuned:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)


def _get_log_dir(experiment: Experiment, fine_tuning=False):
    return os.path.join(
        config.get_log_dir(),
        "experiments", "unimodal", 
        experiment.modality, 
        f"exp{experiment.index}{'_ft' if fine_tuning else ''}"
    )


def _get_trial_id(experiment: Experiment, fine_tuning=False):
    return f"{experiment.index}{'_ft' if fine_tuning else ''}"


def _get_model_dir(experiment: Experiment):
    return os.path.join(
        config.get_models_dir(), 
        "experiments", 
        "unimodal", 
        experiment.modality, 
        f"exp{experiment.index}"
    )


def _unfreeze_for_fine_tuning(model: keras.Model, experiment: Experiment):
    last_layer = experiment.fine_tune_last_layer
    if last_layer != "":
        try:
            first_index = next(i for i, layer in enumerate(model.layers) if layer.name == last_layer)
            for layer in model.layers[first_index:]:
                print(f"Fine tuning {layer=}")
                layer.trainable = True
        except StopIteration:
            model.trainable = True
    else:
        model.trainable = True


def _get_augmentations(experiment: Experiment):
    augmentations = 0

    if experiment.brightness:
        augmentations |= AUGMENTATION_BRIGHTNESS
    if experiment.contrast:
        augmentations |= AUGMENTATION_CONTRAST
    if experiment.flip_left_right:
        augmentations |= AUGMENTATION_FLIP_LEFT_RIGHT
    if experiment.flip_up_down:
        augmentations |= AUGMENTATION_FLIP_UP_DOWN
    if experiment.saturation:
        augmentations |= AUGMENTATION_SATURATION
    if experiment.crop:
        augmentations |= AUGMENTATION_CROP
    if experiment.rotation:
        augmentations |= AUGMENTATION_ROTATION

    return augmentations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        help="Index of the experiment in the /experiment dir",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-s",
        help="Save mode",
        choices=[m.value for m in SaveMode],
        type=int,
        required=False,
        default=SaveMode.NONE.value,
    )

    args = parser.parse_args()

    config = Config("config.json")
    experiment = Experiment(args.e)
    save_mode = SaveMode(args.s)
    
    run(config, experiment, save_mode)
