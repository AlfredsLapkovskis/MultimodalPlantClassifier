from itertools import product
import numpy as np

from dataset.loading import *
from .checkpointing import *
from .unimodal import *
from .surrogate import *
from .temperature_scheduler import *
from .weight_store import *
from .result_store import *
from .model_builder import *
from .nonlinearities import *
from common.utils import log as _log


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
        checkpointer: Checkpointer,
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
        {checkpointer=}

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
        self.checkpointer = checkpointer


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

        def get_best():
            return self.result_store.get_best(self.n_sampled_architectures)

        initial_iteration = 0
        initial_layer_idx = 0

        checkpoint = self.checkpointer.get_checkpoint()
        if checkpoint.is_checkpointed:
            sampled_configurations = checkpoint.sampled_configurations

            if checkpoint.is_scored:
                if checkpoint.layer_idx + 1 >= self.max_fusion_layers:
                    initial_iteration = checkpoint.iteration + 1
                else:
                    initial_iteration = checkpoint.iteration
                    initial_layer_idx = checkpoint.layer_idx + 1
            else:
                initial_iteration = checkpoint.iteration
                initial_layer_idx = checkpoint.layer_idx

        if initial_iteration >= self.n_iterations:
            return get_best()

        for iteration in range(initial_iteration, self.n_iterations):
            current_initial_layer_idx = initial_layer_idx if iteration == initial_iteration else 0

            for layer_idx in range(current_initial_layer_idx, self.max_fusion_layers):
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
        
        return get_best()
    

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
                score = self.result_store.get_result(conf, iteration, layer_idx)
                if score != None:
                    _log(f"""
                         
                    Configuration {i + 1}/{len(configurations)} has already been trained

                    {conf=}
                    {score=}

                    """)

                    scores.append(score)
                    continue

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


        def create_checkpoint(sampled_configurations, scores=None):
            return Checkpoint(
                iteration=iteration,
                layer_idx=layer_idx,
                sampled_configurations=sampled_configurations,
                scores=scores,
            )
        

        self.temperature_scheduler.update(iteration * layer_idx + layer_idx)
        
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

        using_initial_results = is_first_iteration and sampled_configurations.size > 0
        pretraining_batch = n_batches is not None and batch_idx is not None

        if using_initial_results:
            model_configurations = sampled_configurations
        
        if pretraining_batch:
            model_configurations = np.array_split(model_configurations, n_batches)[batch_idx]

        current_checkpoint = self.checkpointer.get_checkpoint()
        skip_sampling = not pretraining_batch and \
            current_checkpoint.is_checkpointed and \
            not current_checkpoint.is_scored and \
            current_checkpoint.iteration == iteration and \
            current_checkpoint.layer_idx == layer_idx and \
            len(sampled_configurations) > 0

        if not skip_sampling:
            if is_first_iteration:
                scores = build_train_and_evaluate(model_configurations)
                self._update_surrogate()

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
            self.checkpointer.set_checkpoint(create_checkpoint(sampled_configurations))
        else:
            _log(f"""

            Proceeding with checkpointed sampled configurations:
                 
            {sampled_configurations=}

            """)

        if not is_first_iteration:
            scores = build_train_and_evaluate(sampled_configurations)
            surrogate_loss = self._update_surrogate()

            _log(f"""

            {scores=}
            {surrogate_loss=}

            """)

        if not pretraining_batch:
            checkpoint = create_checkpoint(sampled_configurations, scores)

            self.surrogate.checkpoint(checkpoint)
            self.checkpointer.set_checkpoint(checkpoint)

        return sampled_configurations


    def _update_surrogate(self):
        all_configurations, all_scores = self.result_store.get_results()
        surrogate_loss = self.surrogate.update(all_configurations, all_scores)

        return surrogate_loss
        

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
        if verbose >= 2:
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
