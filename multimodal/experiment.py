import json
import os
from tensorboard.plugins.hparams import api as hp


_DEFAULT_LR = 1e-3
_DEFAULT_DECAY = False
_DEFAULT_DECAY_STEPS = 200
_DEFAULT_DECAY_RATE = 0.95
_DEFAULT_LABEL_SMOOTHING = 0.0
_DEFAULT_L1 = 0.0
_DEFAULT_L2 = 0.0
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_EPOCHS = 100
_DEFAULT_VERBOSE = 1
_DEFAULT_PATIENCE = 10
_DEFAULT_EARLY_STOPPING_START_EPOCH = 0
_DEFAULT_DROPOUT = 0.0
_DEFAULT_FUSION_NEURONS = 64
_DEFAULT_FINE_TUNE = False
_DEFAULT_FINE_TUNE_LAST_LAYER = ""
_DEFAULT_FINE_TUNE_LR = 1e-4
_DEFAULT_FINE_TUNE_EPOCHS = 100
_DEFAULT_FINE_TUNE_DECAY = False
_DEFAULT_FINE_TUNE_DECAY_STEPS = 200
_DEFAULT_FINE_TUNE_DECAY_RATE = 0.95
_DEFAULT_FINE_TUNE_PATIENCE = 10
_DEFAULT_FINE_TUNE_EARLY_STOPPING_START_EPOCH = 0
_DEFAULT_MULTIMODAL_DROPOUT = 0.0
_DEFAULT_WEIGHT_DECAY = 0.0
_DEFAULT_MERGE_VALIDATION_SET = False
_DEFAULT_AUGMENTATIONS = False
_DEFAULT_CACHE_BATCHES = False


class Experiment:

    def __init__(self, idx):
        self.index = idx
        self.config_file_path = os.path.join(os.getcwd(), "multimodal", "experiments", f"exp{idx}.json")

        with open(self.config_file_path) as f:
            experiment = json.loads(f.read())

        self.model_paths = experiment["model_paths"]
        self.configuration = experiment["configuration"]

        fusion_count = len(self.configuration)
        layer_count = fusion_count + 1

        self.lr = experiment["lr"] if "lr" in experiment else _DEFAULT_LR
        self.decay = experiment["decay"] if "decay" in experiment else _DEFAULT_DECAY
        self.decay_steps = experiment["decay_steps"] if "decay_steps" in experiment else (_DEFAULT_DECAY_STEPS if self.decay else 0)
        self.decay_rate = experiment["decay_rate"] if "decay_rate" in experiment else (_DEFAULT_DECAY_RATE if self.decay else 0.0)
        self.label_smoothing = experiment["label_smoothing"] if "label_smoothing" in experiment else _DEFAULT_LABEL_SMOOTHING
        self.batch_size = experiment["batch_size"] if "batch_size" in experiment else _DEFAULT_BATCH_SIZE
        self.epochs = experiment["epochs"] if "epochs" in experiment else _DEFAULT_EPOCHS
        self.verbose = experiment["verbose"] if "verbose" in experiment else _DEFAULT_VERBOSE
        self.patience = experiment["patience"] if "patience" in experiment else _DEFAULT_PATIENCE
        self.early_stopping_start_epoch = experiment["early_stopping_start_epoch"] if "early_stopping_start_epoch" in experiment else _DEFAULT_EARLY_STOPPING_START_EPOCH
        self.fusion_neurons = experiment["fusion_neurons"] if "fusion_neurons" in experiment else [_DEFAULT_FUSION_NEURONS] * fusion_count
        self.dropouts = experiment["dropouts"] if "dropouts" in experiment else [_DEFAULT_DROPOUT] * layer_count
        self.l1 = experiment["l1"] if "l1" in experiment else [_DEFAULT_L1] * layer_count
        self.l2 = experiment["l2"] if "l2" in experiment else [_DEFAULT_L2] * layer_count
        self.multimodal_dropout = experiment["multimodal_dropout"] if "multimodal_dropout" in experiment else _DEFAULT_MULTIMODAL_DROPOUT
        self.weight_decay = experiment["weight_decay"] if "weight_decay" in experiment else _DEFAULT_WEIGHT_DECAY
        self.fine_tune = experiment["fine_tune"] if "fine_tune" in experiment else _DEFAULT_FINE_TUNE
        self.fine_tune_last_layer = experiment["fine_tune_last_layer"] if "fine_tune_last_layer" in experiment else _DEFAULT_FINE_TUNE_LAST_LAYER
        self.fine_tune_lr = experiment["fine_tune_lr"] if "fine_tune_lr" in experiment else (_DEFAULT_FINE_TUNE_LR if self.fine_tune else 0.0)
        self.fine_tune_epochs = experiment["fine_tune_epochs"] if "fine_tune_epochs" in experiment else (_DEFAULT_FINE_TUNE_EPOCHS if self.fine_tune else 0)
        self.fine_tune_decay = experiment["fine_tune_decay"] if "fine_tune_decay" in experiment else _DEFAULT_FINE_TUNE_DECAY
        self.fine_tune_decay_steps = experiment["fine_tune_decay_steps"] if "fine_tune_decay_steps" in experiment else (_DEFAULT_FINE_TUNE_DECAY_STEPS if self.fine_tune_decay else 0)
        self.fine_tune_decay_rate = experiment["fine_tune_decay_rate"] if "fine_tune_decay_rate" in experiment else (_DEFAULT_FINE_TUNE_DECAY_RATE if self.fine_tune_decay else 0.0)
        self.fine_tune_patience = experiment["fine_tune_patience"] if "fine_tune_patience" in experiment else _DEFAULT_FINE_TUNE_PATIENCE
        self.fine_tune_early_stopping_start_epoch = experiment["fine_tune_early_stopping_start_epoch"] if "fine_tune_early_stopping_start_epoch" in experiment else _DEFAULT_FINE_TUNE_EARLY_STOPPING_START_EPOCH
        self.merge_validation_set = experiment["merge_validation_set"] if "merge_validation_set" in experiment else _DEFAULT_MERGE_VALIDATION_SET
        self.augmentations = experiment["augmentations"] if "augmentations" in experiment else _DEFAULT_AUGMENTATIONS
        self.cache_batches = experiment["cache_batches"] if "cache_batches" in experiment else _DEFAULT_CACHE_BATCHES

    
    def build_hparams(self, with_fine_tuning=False):
        return {
            hp.HParam("lr"): self.lr,
            hp.HParam("decay"): self.decay,
            hp.HParam("decay_steps"): self.decay_steps,
            hp.HParam("decay_rate"): self.decay_rate,
            hp.HParam("label_smoothing"): self.label_smoothing,
            hp.HParam("batch_size"): self.batch_size,
            hp.HParam("patience"): self.patience,
            hp.HParam("early_stopping_start_epoch"): self.early_stopping_start_epoch,
            **{
                hp.HParam(f"fusion_{idx}"): neurons
                for idx, neurons in enumerate(self.fusion_neurons)
            },
            **{
                hp.HParam(f"dropout_{idx}"): neurons
                for idx, neurons in enumerate(self.dropouts)
            },
            **{
                hp.HParam(f"l1_{idx}"): neurons
                for idx, neurons in enumerate(self.l1)
            },
            **{
                hp.HParam(f"l2_{idx}"): neurons
                for idx, neurons in enumerate(self.l2)
            },
            hp.HParam("multimodal_dropout"): self.multimodal_dropout,
            hp.HParam("weight_decay"): self.weight_decay,
            hp.HParam("fine_tune"): self.fine_tune and with_fine_tuning,
            hp.HParam("fine_tune_last_layer"): self.fine_tune_last_layer if with_fine_tuning else "",
            hp.HParam("fine_tune_lr"): self.fine_tune_lr,
            hp.HParam("fine_tune_epochs"): self.fine_tune_epochs,
            hp.HParam("fine_tune_decay"): self.fine_tune_decay,
            hp.HParam("fine_tune_decay_steps"): self.fine_tune_decay_steps,
            hp.HParam("fine_tune_decay_rate"): self.fine_tune_decay_rate,
            hp.HParam("fine_tune_patience"): self.fine_tune_patience,
            hp.HParam("fine_tune_early_stopping_start_epoch"): self.fine_tune_early_stopping_start_epoch,
            hp.HParam("augmentations"): self.augmentations,
            hp.HParam("cache_batches"): self.cache_batches,
            hp.HParam("merge_validation_set"): self.merge_validation_set,
        }
