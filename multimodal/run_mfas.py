import numpy as np
import os
from sklearn.metrics import f1_score
import argparse

from config import Config
from dataset.loading import *
from .classes.mfas import *
from .classes.checkpointing import *
from .classes.unimodal import *
from .classes.surrogate import *
from .classes.temperature_scheduler import *
from .classes.weight_store import *
from .classes.result_store import *
from .classes.model_builder import *
from .classes.nonlinearities import *
from common.utils import log as _log, get_sklearn_class_weight, ensure_no_file_locking
from common.constants import N_CLASSES


ensure_no_file_locking()


def run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
    )
    arg_parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        required=True,
    )
    arg_parser.add_argument(
        "--n_batches",
        type=int,
    )
    arg_parser.add_argument(
        "--batch",
        type=int,
    )
    arg_parser.add_argument(
        "--merge_batched_results",
        type=bool,
        default=False,
    )

    args = arg_parser.parse_args()
    modalities = args.modalities
    paths = args.paths
    n_batches = args.n_batches
    batch_idx = args.batch
    merge_batched_results = args.merge_batched_results

    assert (n_batches is None) == (batch_idx is None)

    config = Config("config.json")

    max_fusion_layers = 4
    n_iterations = 5
    n_epochs = 2

    unimodals = [PlantUnimodal(modality, path) for modality, path in zip(modalities, paths)]

    def get_vocabulary():
        vocabulary = set(NONLINEARITIES)
        for fusion_idx in range(max_fusion_layers):
            for u in unimodals:
                vocabulary.update(u.get_fusable_layer_indices(fusion_idx))
        return vocabulary
    
    checkpointer = JsonCheckpointer(
        json_path=os.path.join(config.get_checkpoint_dir(), f"mfas_checkpoint.json"),
        pretty=True,
    ) if batch_idx is None else DummyCheckpointer()

    merge_batched_results = merge_batched_results and not checkpointer.get_checkpoint().is_checkpointed

    surrogate = PlantSurrogate(
        config=config,
        vocabulary=get_vocabulary(),
        checkpoint=checkpointer.get_checkpoint(),
        max_fusions=max_fusion_layers,
        iterations=n_iterations,
    )

    temperature_scheduler = InvExpTemperatureScheduler(
        max_temperature=10.0,
        min_temperature=0.2,
        decay_rate=4.0,
    )

    batched_suffix = "batched_"
    dir_suffix = "" if batch_idx is None else batched_suffix
    file_name_suffix = "" if batch_idx is None else f"_batch_{batch_idx}_of_{n_batches}"

    weight_store = JsonWeightStore(
        json_path=os.path.join(config.get_checkpoint_dir(), f"{dir_suffix}weights", f"weights{file_name_suffix}.json"),
        pretty=False,
    )

    result_store = JsonResultStore(
        json_path=os.path.join(config.get_checkpoint_dir(), f"{dir_suffix}results", f"results{file_name_suffix}.json"),
        merge_from_dir=os.path.join(config.get_checkpoint_dir(), f"{batched_suffix}results") if merge_batched_results else None,
        pretty=False,
    )

    model_builder = DefaultModelBuilder(
        n_classes=N_CLASSES, 
        fusion_sizes=[64, 64, 64, 64],
        weight_store=weight_store,
    )

    train_ds = load_multimodal_dataset(
        config, 
        "train", 
        modalities,
        [], 
        shuffle=True, 
        batch_size=256,
        cache_batches=True,
    )

    valid_ds = load_multimodal_dataset(
        config, 
        "validation", 
        modalities,
        [], 
        shuffle=False, 
        batch_size=256,
        cache_batches=True,
    )

    train_labels = load_multimodal_labels(config, "train")
    valid_labels = load_multimodal_labels(config, "validation")

    def evaluation_metric(y_pred_proba, y_true):
        y_pred = np.argmax(y_pred_proba, axis=1)
        return f1_score(y_pred, y_true, average="macro")

    n_sampled = 50

    mfas = MFAS(
        unimodals=unimodals,
        surrogate=surrogate,
        max_fusion_layers=max_fusion_layers,
        n_iterations=n_iterations,
        n_epochs=n_epochs,
        n_classes=N_CLASSES,
        n_sampled_architectures=n_sampled,
        temperature_scheduler=temperature_scheduler,
        weight_store=weight_store,
        result_store=result_store,
        model_builder=model_builder,
        checkpointer=checkpointer,
    )

    if batch_idx is not None:
        best_configurations = mfas.search_batch(
            iteration=0,
            layer_idx=0,
            n_batches=n_batches,
            batch_idx=batch_idx,
            train_data=train_ds,
            validation_data=valid_ds,
            y_true=valid_labels,
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            class_weight=get_sklearn_class_weight(train_labels, N_CLASSES),
            verbose=1,
            evaluation_metric=evaluation_metric,
        )
    else:
        best_configurations = mfas.search(
            train_data=train_ds,
            validation_data=valid_ds,
            y_true=valid_labels,
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            class_weight=get_sklearn_class_weight(train_labels, N_CLASSES),
            verbose=1,
            evaluation_metric=evaluation_metric,
            sampled_configurations=np.array([r[0] for r in result_store.get_best(n_sampled)]) if merge_batched_results else np.empty(0),
        )

    _log(f"""
    Best configurations:
         
    {best_configurations}

    """)


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        run()
