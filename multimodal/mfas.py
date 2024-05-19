from itertools import product
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
import argparse
from datetime import datetime

from config import Config
from dataset.loading import load_multimodal_dataset
from .classes.unimodal import *
from .classes.surrogate import *
from .classes.temperature_scheduler import *
from .classes.weight_store import *
from .classes.result_store import *
from .classes.model_builder import *
from .classes.nonlinearities import *
from common.utils import log as _log, get_class_weight
from common.constants import N_CLASSES, MODALITY_LIST


class MFAS:

    def __init__(
        self,
        unimodals: list[Unimodal],
        surrogate: Surrogate,
        max_fusion_layers: int,
        n_iterations: int,
        n_epochs: int,
        n_classes: int,
        n_sampled_architectures: int,
        temperature_scheduler: TemperatureScheduler,
        weight_store: WeightStore,
        result_store: ResultStore,
        model_builder: ModelBuilder,
    ):
        _log(f"""
        ==============================================
             
        {type(self).__name__} initializing with params:

        {unimodals=}
        {surrogate=}
        {max_fusion_layers=}
        {n_iterations=}
        {n_epochs=}
        {n_classes=}
        {n_sampled_architectures=}
        {temperature_scheduler=}
        {weight_store=}
        {result_store=}
        {model_builder=}

        ==============================================
        """)

        self.unimodals = unimodals
        self.surrogate = surrogate
        self.max_fusion_layers = max_fusion_layers
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.n_sampled_architectures = n_sampled_architectures
        self.temperature_scheduler = temperature_scheduler
        self.weight_store = weight_store
        self.result_store = result_store
        self.model_builder = model_builder


    def search(
        self,
        train_data,
        validation_data,
        y_true,
        optimizer,
        loss,
        metrics,
        class_weight,
        verbose,
        evaluation_metric,
        sampled_configurations=np.empty(0),
    ):
        _log(f"""
        ==============================================
             
        {type(self).__name__} searching with params:

        {train_data=}
        {validation_data=}
        {y_true=}
        {optimizer=}
        {loss=}
        {metrics=}
        {class_weight=}
        {verbose=}
        {evaluation_metric=}

        ==============================================
        """)

        for iteration in range(0, self.n_iterations):
            for layer_idx in range(0, self.max_fusion_layers):
                sampled_configurations = self.search_batch(
                    iteration=iteration,
                    layer_idx=layer_idx,
                    n_batches=None,
                    batch_idx=None,
                    train_data=train_data,
                    validation_data=validation_data,
                    y_true=y_true,
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    class_weight=class_weight,
                    verbose=verbose,
                    evaluation_metric=evaluation_metric,
                    sampled_configurations=sampled_configurations,
                )
        
        return self.result_store.get_best(self.n_sampled_architectures)
    

    def search_batch(
        self,
        iteration,
        layer_idx,
        n_batches,
        batch_idx,
        train_data,
        validation_data,
        y_true,
        optimizer,
        loss,
        metrics,
        class_weight,
        verbose,
        evaluation_metric,
        sampled_configurations=np.empty(0),
    ):
        def build_train_and_evaluate(configurations):
            scores = []

            for i, conf in enumerate(configurations):
                _log(f"""

                Training configuration {i + 1}/{len(configurations)}

                {conf=}

                """)

                model = self.model_builder.build_model(conf, self.unimodals)
                self._train_fusion_model(model, conf, train_data, optimizer, loss, metrics, class_weight, verbose)
                score = self._evaluate_model(model, validation_data, y_true, evaluation_metric)

                self.result_store.add_result(conf, iteration, layer_idx, score)

                scores.append(score)

                _log(f"""

                Finished training configuration {i + 1}/{len(configurations)}

                {conf=}
                {score=}

                """)
            
            return scores
        
        _log(f"""
            ----------------------------------------------
                     
            Iteration {iteration} | Layer {layer_idx}

            temperature={self.temperature_scheduler.get_temperature()}
            {len(sampled_configurations)=}
            {self.weight_store=}
            {self.result_store=}
                     
            ----------------------------------------------
        """)

        is_first_iteration = iteration + layer_idx == 0

        layer_configurations = self._generate_layer_configurations(layer_idx)
        model_configurations = self._generate_model_configurations(sampled_configurations, layer_configurations, layer_idx)

        retraining_from_results = False

        if is_first_iteration and sampled_configurations.size > 0:
            retraining_from_results = True
            model_configurations = sampled_configurations
        
        if n_batches is not None and batch_idx is not None:
            model_configurations = np.array_split(model_configurations, n_batches)[batch_idx]

        if not retraining_from_results:
            model_configurations = [c for c in model_configurations if not self.result_store.contains(c, iteration, layer_idx)]

        if is_first_iteration:
            scores = build_train_and_evaluate(model_configurations)

            all_configurations, all_scores = self.result_store.get_results()
            self.surrogate.update(all_configurations, all_scores)

            surrogate_scores = self.surrogate.predict(model_configurations)
            surrogate_error = np.abs(np.array(scores) - np.array(surrogate_scores))

            _log(f"""

            Trained architectures:
                         
            {model_configurations=}
            {scores=}
            {surrogate_error=}
                    
            """)
        else:
            scores = self.surrogate.predict(model_configurations)

            _log(f"""

            Predicted scores:
                         
            {model_configurations=}
            {scores=}

            """)

        sampled_configurations = self._sample_configurations(model_configurations, scores, self.temperature_scheduler.get_temperature())

        if not is_first_iteration:
            scores = build_train_and_evaluate(sampled_configurations)

            all_configurations, all_scores = self.result_store.get_results()
            surrogate_loss = self.surrogate.update(all_configurations, all_scores)

            _log(f"""

            {scores=}
            {surrogate_loss=}

            """)

        self.temperature_scheduler.update(iteration * layer_idx + layer_idx)

        return sampled_configurations
        

    def _generate_layer_configurations(self, layer_idx):
        return np.array(list(product(
            *map(lambda u: u.get_fusable_layer_indices(layer_idx), self.unimodals),
            NONLINEARITIES,
        )))
    

    def _generate_model_configurations(self, sampled_configurations: np.array, layer_configurations: np.array, layer_idx):
        if len(sampled_configurations) == 0:
            assert layer_idx == 0
            return np.expand_dims(layer_configurations, 1)
        
        model_configurations = []

        for sampled_conf in sampled_configurations:
            for layer_conf in layer_configurations:
                model_conf = np.array(sampled_conf)

                if layer_idx < len(sampled_conf):
                    model_conf[layer_idx] = layer_conf
                else:
                    layer_conf = np.expand_dims(layer_conf, 0)
                    model_conf = np.concatenate([sampled_conf, layer_conf])
                
                model_configurations.append(model_conf)
        
        return np.array(model_configurations)


    def _train_fusion_model(
        self, 
        model,
        model_conf,
        train_data,
        optimizer,
        loss,
        metrics,
        class_weight,
        verbose,
    ):
        self.weight_store.load_weights(model, model_conf)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        if verbose >= 1:
            model.summary()

        model.fit(
            train_data,
            epochs=self.n_epochs,
            class_weight=class_weight,
            verbose=verbose,
        )

        self.weight_store.save_weights(model, model_conf)


    def _evaluate_model(self, model, validation_data, y_true, evaluation_metric):
        y_pred_proba = model.predict(validation_data)
        return evaluation_metric(y_pred_proba, y_true)
            
    
    def _sample_configurations(self, model_configurations, scores, temperature):
        scores = np.array(scores) / np.sum(scores)
        scores = pow(scores, 1. / temperature)
        scores /= scores.sum()

        indices = np.random.choice(len(model_configurations), self.n_sampled_architectures, replace=False, p=scores)
        return [model_configurations[i] for i in indices]



def main():
    arg_parser = argparse.ArgumentParser()
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
    n_batches = args.n_batches
    batch_idx = args.batch
    merge_batched_results = args.merge_batched_results

    assert (n_batches is None) == (batch_idx is None)

    config = Config("config.json")

    unimodals = [PlantUnimodal(modality) for modality in MODALITY_LIST]

    surrogate = PlantSurrogate()

    temperature_scheduler = InvExpTemperatureScheduler(
        max_temperature=10.0,
        min_temperature=0.2,
        decay_rate=4.0,
    )

    batched_suffix = "batched_"
    dir_suffix = "" if batch_idx is None else batched_suffix
    file_name_suffix = f"{datetime.now()}" if batch_idx is None else f"batch_{batch_idx}_of_{n_batches}"

    weight_store = JsonWeightStore(
        json_path=os.path.join(config.get_checkpoint_dir(), f"{dir_suffix}weights", f"weights_{file_name_suffix}.json"),
        pretty=False,
    )

    result_store = JsonResultStore(
        json_path=os.path.join(config.get_checkpoint_dir(), f"{dir_suffix}results", f"results_{file_name_suffix}.json"),
        merge_from_dir=os.path.join(config.get_checkpoint_dir(), f"{batched_suffix}results") if merge_batched_results else None,
        pretty=False,
    )

    model_builder = DefaultModelBuilder(
        n_classes=N_CLASSES, 
        fusion_sizes=[64, 64, 64],
        weight_store=weight_store,
    )

    train_df = pd.read_csv(config.get_multimodal_csv_file_path("train"))
    valid_df = pd.read_csv(config.get_multimodal_csv_file_path("validation"))

    train_ds = load_multimodal_dataset(config, "train", MODALITY_LIST, [], shuffle=True, batch_size=128, dropout=0.125, df=train_df)
    valid_ds = load_multimodal_dataset(config, "validation", MODALITY_LIST, [], shuffle=False, batch_size=128, df=valid_df)

    def evaluation_metric(y_pred_proba, y_true):
        y_pred = np.argmax(y_pred_proba, axis=1)
        return f1_score(y_pred, y_true, average="macro")

    n_sampled = 50

    mfas = MFAS(
        unimodals=unimodals,
        surrogate=surrogate,
        max_fusion_layers=3,
        n_iterations=10,
        n_epochs=2,
        n_classes=N_CLASSES,
        n_sampled_architectures=n_sampled,
        temperature_scheduler=temperature_scheduler,
        weight_store=weight_store,
        result_store=result_store,
        model_builder=model_builder,
    )

    if batch_idx is not None:
        best_configurations = mfas.search_batch(
            iteration=0,
            layer_idx=0,
            n_batches=n_batches,
            batch_idx=batch_idx,
            train_data=train_ds,
            validation_data=valid_ds,
            y_true=valid_df["Label"],
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            class_weight=get_class_weight(train_df, N_CLASSES),
            verbose=1,
            evaluation_metric=evaluation_metric,
        )
    else:
        best_configurations = mfas.search(
            train_data=train_ds,
            validation_data=valid_ds,
            y_true=valid_df["Label"],
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            class_weight=get_class_weight(train_df, N_CLASSES),
            verbose=1,
            evaluation_metric=evaluation_metric,
            sampled_configurations=np.array([r[0] for r in result_store.get_best(n_sampled)]) if merge_batched_results else np.empty(0),
        )

    _log(f"""
    Best configurations:
         
    {best_configurations}

    """)


if __name__ == "__main__":
    main()
