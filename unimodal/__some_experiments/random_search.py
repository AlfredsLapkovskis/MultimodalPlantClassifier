import keras
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from time import time
import os
import json

from config import Config
from dataset.loading import load_unimodal_dataset
from common.constants import N_CLASSES, IMAGE_SHAPE


@dataclass
class Layer:
    neurons: int
    dropout: float

@dataclass
class HyperParams:
    layers: list[Layer]
    learning_rate: float
    lr_scheduler: int
    batch_size: int
    weight_decay: float
    classification_dropout: float
    classification_l1: float
    classification_l2: float
    optimizer: str


def train_model(config: Config, modality: str, base_model_name: str, hyperparams: HyperParams, augmentations={}, monitor="val_loss", epochs=100, verbose=0):
    train_df = pd.read_csv(config.get_unimodal_csv_file_path("train", modality))
    valid_df = pd.read_csv(config.get_unimodal_csv_file_path("validation", modality))

    class_weight = _get_class_weight(train_df)

    base_model: keras.Model = _get_base_model(base_model_name)

    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SHAPE)
    x = base_model(inputs, training=False)

    for i, layer in enumerate(hyperparams.layers):
        x = keras.layers.Dropout(layer.dropout, name=f"a_dropout_{i + 1}")(x)
        x = keras.layers.Dense(layer.neurons, activation="relu", name=f"a_dense_{i + 1}")(x)

    x = keras.layers.Dropout(hyperparams.classification_dropout)(x)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name="a_softmax", kernel_regularizer=keras.regularizers.l1_l2(hyperparams.classification_l1, hyperparams.classification_l2))(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=_get_optimizer_for_model(base_model_name, hyperparams),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds = load_unimodal_dataset(
        config,
        modality,
        "train",
        df=train_df,
        augmentations=augmentations,
        shuffle=True,
        batch_size=hyperparams.batch_size,
    )
    valid_ds = load_unimodal_dataset(
        config,
        modality,
        "validation",
        df=valid_df,
        batch_size=hyperparams.batch_size,
    )

    log_name = f"{datetime.now()}__l{len(hyperparams.layers)}({','.join(map(lambda l: f'{l.neurons}:{l.dropout}', hyperparams.layers))})_{hyperparams.optimizer}_lr{hyperparams.learning_rate}_bs{hyperparams.batch_size}_wd{hyperparams.weight_decay}_cd{hyperparams.classification_dropout}_cl1{hyperparams.classification_l1}_cl2{hyperparams.classification_l2}_a{augmentations}"

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        class_weight=class_weight,
        epochs=epochs,
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor=monitor,
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(config.get_log_dir(), base_model_name, modality, log_name),
                histogram_freq=1,
            ),
        ],
    )

    try:
        json_str = json.dumps(history.history, indent=2)
        with open(os.path.join(config.get_log_dir(), base_model_name, modality, f"{log_name}.json"), "w") as f:
            f.write(json_str)
    except:
        pass



def _get_base_model(name):
    match name:
        case "mobilenet":
            return keras.applications.MobileNetV3Small(
                input_shape=IMAGE_SHAPE,
                include_top=False,
                pooling="avg",
                include_preprocessing=False,
            )
        case "inception":
            return keras.applications.InceptionV3(
                input_shape=IMAGE_SHAPE,
                include_top=False,
                pooling="avg",
            )
        case "densenet":
            return keras.applications.DenseNet121(
                input_shape=IMAGE_SHAPE,
                include_top=False,
                pooling="avg",
            )
        case "nasnet":
            return keras.applications.NASNetMobile(
                input_shape=IMAGE_SHAPE,
                include_top=False,
                pooling="avg",
            )
        case _:
            raise RuntimeError("Undefined model name")
        

def _get_optimizer_for_model(name, hyperparams: HyperParams):
    match name:
        case "mobilenet":
            match hyperparams.lr_scheduler:
                case 1:
                    scheduler = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-3,
                        decay_steps=150,
                        decay_rate=0.75,
                    )
                case 2:
                    scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
                        boundaries=[450, 4500],
                        values=[1e-3, 1e-4, 1e-5],
                    )
                case 3: 
                    scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(
                        boundaries=[1, 5], # Good results 1,3 | 1e-1,1e-2,1e-3
                        values=[1e-1, 1e-2, 1e-3],
                    )
                case _:
                    scheduler = None
            match hyperparams.optimizer:
                case "sgd":
                    return keras.optimizers.SGD(
                        learning_rate=scheduler if scheduler else hyperparams.learning_rate,
                        momentum=0.9,
                        weight_decay=hyperparams.weight_decay,
                    )
                case "rmsprop":
                    return keras.optimizers.RMSprop(
                        learning_rate=scheduler if scheduler else hyperparams.learning_rate, # 0.1 every 3 epochs decay 0.01
                        weight_decay=hyperparams.weight_decay, # 1e-5 
                    )
                case _:
                    return None    
        case "inception":
            return keras.optimizers.RMSprop(
                learning_rate=hyperparams.learning_rate, # 0.045
                weight_decay=hyperparams.weight_decay, # 4e-5 every 2 epochs exp 0.94
            )
        case "densenet":
            return keras.optimizers.SGD(
                learning_rate=hyperparams.learning_rate, # 0.1
                weight_decay=hyperparams.weight_decay, # 1e-4 lower by 10 times at epochs 30 and 60
            )
        case "nasnet":
            return keras.optimizers.RMSprop(
                learning_rate=hyperparams.learning_rate,
                weight_decay=hyperparams.weight_decay, # every 2 epochs exp 0.94
            )
        case _:
            raise RuntimeError("Undefined model name")


def _get_class_weight(df):
    return {c: 1 - (df["Label"] == c).sum() / len(df) for c in range(N_CLASSES)}

def _get_random_n_layers():
    return np.random.randint(0, 4)

def _get_random_neurons(model, n_layers):
    match model:
        case "mobilenet":
            return np.random.choice([128, 256, 512, 800], n_layers, replace=True)
        case "inception":
            return np.random.choice([512, 1024, 2048, 2500], n_layers, replace=True)
        case "densenet":
            return np.random.choice([256, 512, 1024, 1500], n_layers, replace=True)
        case "nasnet":
            return np.random.choice([256, 512, 1024, 1500], n_layers, replace=True)
        case _:
            return RuntimeError("Undefined model name")

def _get_random_dropout(n_layers):
    return np.random.choice([0, 0.1, 0.25, 0.3], n_layers)

def _get_random_batch_size():
    return np.random.choice([32, 64, 128])

def _get_random_weight_decay():
    return np.random.uniform(0, 0.1)

def _get_random_learning_rate():
    return np.random.choice([1e-3, 1e-4])

def _get_random_augmentations():
    augmentations = [1, 2, 3, 4, 6, 7, 8]
    return set(np.random.choice(augmentations, np.random.random_integers(0, len(augmentations)), replace=True))

def _get_random_classification_l1():
    return np.random.choice([1e-3, 1e-4, 1e-5])

def _get_random_classification_l2():
    return np.random.choice([1e-3, 1e-4, 1e-5])


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()

    argparse.add_argument(
        "--modality",
        type=str,
        required=True,
    )
    argparse.add_argument(
        "--model",
        type=str,
        required=True,
    )
    argparse.add_argument(
        "--n_layers",
        type=int,
    )
    argparse.add_argument(
        "--n",
        help="neurons per layer",
        type=int,
        nargs="+",
    )
    argparse.add_argument(
        "--d",
        help="dropout per layer",
        type=float,
        nargs="+",
    )
    argparse.add_argument(
        "--cd",
        help="classification layer dropout",
        type=float,
    )
    argparse.add_argument(
        "--lr",
        help="learning rate",
        type=float,
    )
    argparse.add_argument(
        "--lr_s",
        help="learning rate scheduler",
        type=int,
    )
    argparse.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
    )
    argparse.add_argument(
        "--batch",
        type=int,
    )
    argparse.add_argument(
        "--cl1",
        help="classification l1 regularization",
        type=float,
    )
    argparse.add_argument(
        "--cl2",
        help="classification l2 regularization",
        type=float,
    )
    argparse.add_argument(
        "--weight_decay",
        type=float,
    )
    argparse.add_argument(
        "--augmentations",
        type=int,
        nargs="*",
        default=[],
    )
    argparse.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
    )
    argparse.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    argparse.add_argument(
        "--verbose",
        type=int,
        default=0,
    )
    argparse.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = argparse.parse_args()

    seed = int(time()) - np.random.default_rng(args.seed).integers(0, 1000000)
    np.random.seed(seed)

    config = Config("config.json")

    modality = args.modality
    model = args.model
    n_layers = args.n_layers if args.n_layers != None else _get_random_n_layers()
    neurons = args.n if args.n else _get_random_neurons(model, n_layers)
    dropouts = args.d if args.d else _get_random_dropout(n_layers)

    assert n_layers >= 0 and n_layers == len(neurons) and n_layers == len(dropouts)

    layers = []
    for i in range(n_layers):
        layers.append(Layer(neurons=neurons[i], dropout=dropouts[i]))

    classification_dropout = args.cd if args.cd != None else _get_random_dropout(1)[0]
    classification_l1 = args.cl1 if args.cl1 != None else _get_random_classification_l1()
    classification_l2 = args.cl2 if args.cl2 != None else _get_random_classification_l2()
    batch_size = args.batch if args.batch else _get_random_batch_size()
    weight_decay = args.weight_decay if args.weight_decay != None else _get_random_weight_decay()
    learning_rate = args.lr if args.lr else _get_random_learning_rate()
    lr_scheduler = args.lr_s
    optimizer = args.optimizer
    augmentations = set(args.augmentations) if args.augmentations else _get_random_augmentations()
    monitor = args.monitor
    epochs = args.epochs
    verbose = args.verbose

    if verbose >= 1:
        print(f"modality: {modality}")
        print(f"model: {model}")
        print(f"layers: {layers}")
        print(f"classification dropout: {classification_dropout}")
        print(f"classification l1: {classification_l1}")
        print(f"classification l2: {classification_l2}")
        print(f"batch_size: {batch_size}")
        print(f"weight_decay: {weight_decay}")
        print(f"learning_rate: {learning_rate}")
        print(f"learning rate scheduler: {lr_scheduler}")
        print(f"optimizer: {optimizer}")
        print(f"augmentations: {augmentations}")
        print(f"monitor: {monitor}")
        print(f"epochs: {epochs}")
        print(f"verbose: {verbose}")
        print(f"seed: {seed}")

    hyperparams = HyperParams(
        layers=layers,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        batch_size=batch_size,
        weight_decay=weight_decay,
        classification_dropout=classification_dropout,
        classification_l1=classification_l1,
        classification_l2=classification_l2,
        optimizer=optimizer,
    )

    train_model(
        config=config,
        modality=modality,
        base_model_name=model,
        hyperparams=hyperparams,
        augmentations=augmentations,
        monitor=monitor,
        epochs=epochs,
        verbose=verbose,
    )