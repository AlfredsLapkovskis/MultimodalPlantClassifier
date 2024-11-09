import keras
import tensorflow as tf
import os
from tensorboard.plugins.hparams import api as hp
import argparse

from config import Config
from dataset.loading import *
from common.utils import *
from common.constants import N_CLASSES
from common.save_mode import SaveMode
from .experiment import Experiment
from .classes.model_builder import DefaultModelBuilder
from .classes.unimodal import PlantUnimodal


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
    model_builder = DefaultModelBuilder(
        n_classes=N_CLASSES,
        fusion_sizes=experiment.fusion_neurons,
        dropouts=experiment.dropouts,
        l1=experiment.l1,
        l2=experiment.l2,
        batch_norm=[True] * len(experiment.configuration),
    )

    unimodals = [PlantUnimodal(modality, path) for modality, path in experiment.model_paths.items()]

    return model_builder.build_model(experiment.configuration, unimodals)


def _load_datasets(config: Config, experiment: Experiment):
    def encode_labels(labels):
        labels = tf.one_hot(labels, N_CLASSES)
        return tf.cast(labels, dtype=tf.float32)

    train_ds = load_multimodal_dataset(
        config,
        ["train", "validation"] if experiment.merge_validation_set else "train",
        modalities=MODALITY_LIST,
        modality_augmentations={
            "Flower": AUGMENTATION_CONTRAST | AUGMENTATION_FLIP_LEFT_RIGHT,
            "Leaf": AUGMENTATION_FLIP_LEFT_RIGHT | AUGMENTATION_FLIP_UP_DOWN,
            "Fruit": AUGMENTATION_CONTRAST | AUGMENTATION_FLIP_LEFT_RIGHT,
            "Stem": AUGMENTATION_FLIP_LEFT_RIGHT,
        } if experiment.augmentations else dict(),
        shuffle=True,
        batch_size=experiment.batch_size,
        dropout=experiment.multimodal_dropout,
        cache_batches=experiment.cache_batches,
    ).map(lambda x, y: (x, encode_labels(y)), num_parallel_calls=tf.data.AUTOTUNE)

    valid_ds = None
    if not experiment.merge_validation_set:
        valid_ds = load_multimodal_dataset(
            config,
            "validation",
            modalities=MODALITY_LIST,
            batch_size=experiment.batch_size,
            cache_batches=experiment.cache_batches,
        ).map(lambda x, y: (x, encode_labels(y)), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, valid_ds


def _load_train_labels(config: Config, experiment: Experiment):
    return load_multimodal_labels(
        config,
        ["train", "validation"] if experiment.merge_validation_set else "train",
    )


def _compile_fit(model: keras.Model, experiment: Experiment, train_ds, valid_ds, train_labels, fine_tuning=False):
    loss = get_categorical_crossentropy_with_smoothing(experiment.label_smoothing)
    class_weight = get_sklearn_class_weight(train_labels, N_CLASSES)

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
        ],
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


def _save_model_if_needed(model: keras.Model, experiment: Experiment, save_mode: SaveMode, fine_tuning=False):
    model_dir = _get_model_dir()
    model_path = os.path.join(model_dir, f"exp{experiment.index}{'_ft' if fine_tuning else ''}.keras")

    save_trained = not fine_tuning and save_mode.value >= SaveMode.TRAINED.value
    save_fine_tuned = fine_tuning and save_mode.value >= SaveMode.TRAINED_AND_FINE_TUNED.value

    if save_trained or save_fine_tuned:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)


def _get_log_dir(experiment: Experiment, fine_tuning=False):
    return os.path.join(
        config.get_log_dir(), 
        "experiments", "multimodal",
        f"exp{experiment.index}{'_ft' if fine_tuning else ''}"
    )


def _get_trial_id(experiment: Experiment, fine_tuning=False):
    return f"{experiment.index}{'_ft' if fine_tuning else ''}"

    
def _get_model_dir():
    return os.path.join(
        config.get_models_dir(), 
        "experiments", 
        "multimodal",
    )
    
    
def _unfreeze_for_fine_tuning(base_model: keras.Model, experiment: Experiment):
    last_layer = experiment.fine_tune_last_layer
    if last_layer != "":
        not_found_counter = 0

        for modality in MODALITY_LIST:
            modality_last_layer = f"{modality}_{last_layer}"
            modality_layers = [layer for layer in base_model.layers if layer.name.startswith(modality)]
            try:
                first_index = next(i for i, layer in enumerate(modality_layers) if layer.name == modality_last_layer)
                for layer in modality_layers[first_index:]:
                    print(f"Fine tuning {layer.name=}")
                    layer.trainable = True
            except StopIteration:
                print(f"Warning: Layer {modality_last_layer} not found")
                not_found_counter += 1

        if not_found_counter == len(MODALITY_LIST):
            print(f"ERROR: Layer {last_layer} not found in for any modality, fine-tuning all layers!")
            base_model.trainable = True
    else:
        print("Fine-tuning all layers")
        base_model.trainable = True
    

if  __name__ == "__main__":
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
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        run(config, experiment, save_mode)
