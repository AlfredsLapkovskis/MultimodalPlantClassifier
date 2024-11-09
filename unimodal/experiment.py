import json
import os
from tensorboard.plugins.hparams import api as hp


_DEFAULT_LR = 1e-3
_DEFAULT_DECAY = False
_DEFAULT_DECAY_STEPS = 200
_DEFAULT_DECAY_RATE = 0.95
_DEFAULT_LABEL_SMOOTHING = 0.0
_DEFAULT_CLASSIFIER_L1 = 0.0
_DEFAULT_CLASSIFIER_L2 = 0.0
_DEFAULT_BATCH_SIZE = 128
_DEFAULT_EPOCHS = 100
_DEFAULT_VERBOSE = 1
_DEFAULT_PATIENCE = 10
_DEFAULT_EARLY_STOPPING_START_EPOCH = 0
_DEFAULT_FOCAL_LOSS = False
_DEFAULT_FOCAL_LOSS_GAMMA = 2.0
_DEFAULT_BRIGHTNESS = False
_DEFAULT_CONTRAST = False
_DEFAULT_FLIP_LEFT_RIGHT = False
_DEFAULT_FLIP_UP_DOWN = False
_DEFAULT_SATURATION = False
_DEFAULT_CROP = False
_DEFAULT_ROTATION = False
_DEFAULT_CLASSIFIER_DROPOUT = 0.0
_DEFAULT_INTERMEDIATE_LAYER = False
_DEFAULT_INTERMEDIATE_NEURONS = 2048
_DEFAULT_INTERMEDIATE_DROPOUT = 0.0
_DEFAULT_INTERMEDIATE_INITIALIZER = "glorot_uniform"
_DEFAULT_INTERMEDIATE_ACTIVATION = "relu"
_DEFAULT_INTERMEDIATE_L1 = 0.0
_DEFAULT_INTERMEDIATE_L2 = 0.0
_DEFAULT_FINE_TUNE = False
_DEFAULT_FINE_TUNE_LAST_LAYER = ""
_DEFAULT_FINE_TUNE_LR = 1e-4
_DEFAULT_FINE_TUNE_EPOCHS = 100
_DEFAULT_FINE_TUNE_DECAY = False
_DEFAULT_FINE_TUNE_DECAY_STEPS = 200
_DEFAULT_FINE_TUNE_DECAY_RATE = 0.95
_DEFAULT_FINE_TUNE_PATIENCE = 10
_DEFAULT_FINE_TUNE_EARLY_STOPPING_START_EPOCH = 0
_DEFAULT_WEIGHT_DECAY = 0.0
_DEFAULT_SKLEARN_CLASS_WEIGHT = False
_DEFAULT_MERGE_VALIDATION_SET = False


class Experiment:
    
    def __init__(self, idx):
        self.index = idx
        self.config_file_path = os.path.join(os.getcwd(), "unimodal", "experiments", f"exp{idx}.json")

        with open(self.config_file_path) as f:
            experiment = json.loads(f.read())

        self.modality = experiment["modality"]
        self.lr = experiment["lr"] if "lr" in experiment else _DEFAULT_LR
        self.decay = experiment["decay"] if "decay" in experiment else _DEFAULT_DECAY
        self.decay_steps = experiment["decay_steps"] if "decay_steps" in experiment else (_DEFAULT_DECAY_STEPS if self.decay else 0)
        self.decay_rate = experiment["decay_rate"] if "decay_rate" in experiment else (_DEFAULT_DECAY_RATE if self.decay else 0.0)
        self.label_smoothing = experiment["label_smoothing"] if "label_smoothing" in experiment else _DEFAULT_LABEL_SMOOTHING
        self.classifier_l1 = experiment["classifier_l1"] if "classifier_l1" in experiment else _DEFAULT_CLASSIFIER_L1
        self.classifier_l2 = experiment["classifier_l2"] if "classifier_l2" in experiment else _DEFAULT_CLASSIFIER_L2
        self.batch_size = experiment["batch_size"] if "batch_size" in experiment else _DEFAULT_BATCH_SIZE
        self.epochs = experiment["epochs"] if "epochs" in experiment else _DEFAULT_EPOCHS
        self.verbose = experiment["verbose"] if "verbose" in experiment else _DEFAULT_VERBOSE
        self.patience = experiment["patience"] if "patience" in experiment else _DEFAULT_PATIENCE
        self.early_stopping_start_epoch = experiment["early_stopping_start_epoch"] if "early_stopping_start_epoch" in experiment else _DEFAULT_EARLY_STOPPING_START_EPOCH
        self.focal_loss = experiment["focal_loss"] if "focal_loss" in experiment else _DEFAULT_FOCAL_LOSS
        self.focal_loss_gamma = experiment["focal_loss_gamma"] if "focal_loss_gamma" in experiment else (_DEFAULT_FOCAL_LOSS_GAMMA if self.focal_loss else 0.0)
        self.brightness = experiment["brightness"] if "brightness" in experiment else _DEFAULT_BRIGHTNESS
        self.contrast = experiment["contrast"] if "contrast" in experiment else _DEFAULT_CONTRAST
        self.flip_left_right = experiment["flip_left_right"] if "flip_left_right" in experiment else _DEFAULT_FLIP_LEFT_RIGHT
        self.flip_up_down = experiment["flip_up_down"] if "flip_up_down" in experiment else _DEFAULT_FLIP_UP_DOWN
        self.saturation = experiment["saturation"] if "saturation" in experiment else _DEFAULT_SATURATION
        self.crop = experiment["crop"] if "crop" in experiment else _DEFAULT_CROP
        self.rotation = experiment["rotation"] if "rotation" in experiment else _DEFAULT_ROTATION
        self.classifier_dropout = experiment["classifier_dropout"] if "classifier_dropout" in experiment else _DEFAULT_CLASSIFIER_DROPOUT
        self.intermediate_layer = experiment["intermediate_layer"] if "intermediate_layer" in experiment else _DEFAULT_INTERMEDIATE_LAYER
        self.intermediate_neurons = experiment["intermediate_neurons"] if "intermediate_neurons" in experiment else (_DEFAULT_INTERMEDIATE_NEURONS if self.intermediate_layer else 0)
        self.intermediate_dropout = experiment["intermediate_dropout"] if "intermediate_dropout" in experiment else (_DEFAULT_INTERMEDIATE_DROPOUT if self.intermediate_layer else 0.0)
        self.intermediate_initializer = experiment["intermediate_initializer"] if "intermediate_initializer" in experiment else (_DEFAULT_INTERMEDIATE_INITIALIZER if self.intermediate_layer else "None")
        self.intermediate_activation = experiment["intermediate_activation"] if "intermediate_activation" in experiment else (_DEFAULT_INTERMEDIATE_ACTIVATION if self.intermediate_layer else "None")
        self.intermediate_l1 = experiment["intermediate_l1"] if "intermediate_l1" in experiment else _DEFAULT_INTERMEDIATE_L1
        self.intermediate_l2 = experiment["intermediate_l2"] if "intermediate_l2" in experiment else _DEFAULT_INTERMEDIATE_L2
        self.intermediate_layer_2 = experiment["intermediate_layer_2"] if "intermediate_layer_2" in experiment else _DEFAULT_INTERMEDIATE_LAYER
        self.intermediate_neurons_2 = experiment["intermediate_neurons_2"] if "intermediate_neurons_2" in experiment else (_DEFAULT_INTERMEDIATE_NEURONS if self.intermediate_layer_2 else 0)
        self.intermediate_dropout_2 = experiment["intermediate_dropout_2"] if "intermediate_dropout_2" in experiment else (_DEFAULT_INTERMEDIATE_DROPOUT if self.intermediate_layer_2 else 0.0)
        self.intermediate_initializer_2 = experiment["intermediate_initializer_2"] if "intermediate_initializer_2" in experiment else (_DEFAULT_INTERMEDIATE_INITIALIZER if self.intermediate_layer_2 else "None")
        self.intermediate_activation_2 = experiment["intermediate_activation_2"] if "intermediate_activation_2" in experiment else (_DEFAULT_INTERMEDIATE_ACTIVATION if self.intermediate_layer_2 else "None")
        self.intermediate_2_l1 = experiment["intermediate_2_l1"] if "intermediate_2_l1" in experiment else _DEFAULT_INTERMEDIATE_L1
        self.intermediate_2_l2 = experiment["intermediate_2_l2"] if "intermediate_2_l2" in experiment else _DEFAULT_INTERMEDIATE_L2
        self.fine_tune = experiment["fine_tune"] if "fine_tune" in experiment else _DEFAULT_FINE_TUNE
        self.fine_tune_last_layer = experiment["fine_tune_last_layer"] if "fine_tune_last_layer" in experiment else _DEFAULT_FINE_TUNE_LAST_LAYER
        self.fine_tune_lr = experiment["fine_tune_lr"] if "fine_tune_lr" in experiment else (_DEFAULT_FINE_TUNE_LR if self.fine_tune else 0.0)
        self.fine_tune_epochs = experiment["fine_tune_epochs"] if "fine_tune_epochs" in experiment else (_DEFAULT_FINE_TUNE_EPOCHS if self.fine_tune else 0)
        self.fine_tune_decay = experiment["fine_tune_decay"] if "fine_tune_decay" in experiment else _DEFAULT_FINE_TUNE_DECAY
        self.fine_tune_decay_steps = experiment["fine_tune_decay_steps"] if "fine_tune_decay_steps" in experiment else (_DEFAULT_FINE_TUNE_DECAY_STEPS if self.fine_tune_decay else 0)
        self.fine_tune_decay_rate = experiment["fine_tune_decay_rate"] if "fine_tune_decay_rate" in experiment else (_DEFAULT_FINE_TUNE_DECAY_RATE if self.fine_tune_decay else 0.0)
        self.fine_tune_patience = experiment["fine_tune_patience"] if "fine_tune_patience" in experiment else _DEFAULT_FINE_TUNE_PATIENCE
        self.fine_tune_early_stopping_start_epoch = experiment["fine_tune_early_stopping_start_epoch"] if "fine_tune_early_stopping_start_epoch" in experiment else _DEFAULT_FINE_TUNE_EARLY_STOPPING_START_EPOCH
        self.weight_decay = experiment["weight_decay"] if "weight_decay" in experiment else _DEFAULT_WEIGHT_DECAY
        self.sklearn_class_weight = experiment["sklearn_class_weight"] if "sklearn_class_weight" in experiment else _DEFAULT_SKLEARN_CLASS_WEIGHT
        self.merge_validation_set = experiment["merge_validation_set"] if "merge_validation_set" in experiment else _DEFAULT_MERGE_VALIDATION_SET

    def build_hparams(self, with_fine_tuning=False):
        return {
            hp.HParam("lr"): self.lr,
            hp.HParam("decay"): self.decay,
            hp.HParam("decay_steps"): self.decay_steps,
            hp.HParam("decay_rate"): self.decay_rate,
            hp.HParam("label_smoothing"): self.label_smoothing,
            hp.HParam("classifier_l1"): self.classifier_l1,
            hp.HParam("classifier_l2"): self.classifier_l2,
            hp.HParam("batch_size"): self.batch_size,
            hp.HParam("patience"): self.patience,
            hp.HParam("early_stopping_start_epoch"): self.early_stopping_start_epoch,
            hp.HParam("focal_loss"): self.focal_loss,
            hp.HParam("focal_loss_gamma"): self.focal_loss_gamma,
            hp.HParam("brightness"): self.brightness,
            hp.HParam("contrast"): self.contrast,
            hp.HParam("flip_left_right"): self.flip_left_right,
            hp.HParam("flip_up_down"): self.flip_up_down,
            hp.HParam("saturation"): self.saturation,
            hp.HParam("crop"): self.crop,
            hp.HParam("rotation"): self.rotation,
            hp.HParam("classifier_dropout"): self.classifier_dropout,
            hp.HParam("intermediate_layer"): self.intermediate_layer,
            hp.HParam("intermediate_neurons"): self.intermediate_neurons,
            hp.HParam("intermediate_dropout"): self.intermediate_dropout,
            hp.HParam("intermediate_initializer"): self.intermediate_initializer,
            hp.HParam("intermediate_activation"): self.intermediate_activation,
            hp.HParam("intermediate_l1"): self.intermediate_l1,
            hp.HParam("intermediate_l2"): self.intermediate_l2,
            hp.HParam("intermediate_layer_2"): self.intermediate_layer_2,
            hp.HParam("intermediate_neurons_2"): self.intermediate_neurons_2,
            hp.HParam("intermediate_dropout_2"): self.intermediate_dropout_2,
            hp.HParam("intermediate_initializer_2"): self.intermediate_initializer_2,
            hp.HParam("intermediate_activation_2"): self.intermediate_activation_2,
            hp.HParam("intermediate_2_l1"): self.intermediate_2_l1,
            hp.HParam("intermediate_2_l2"): self.intermediate_2_l2,
            hp.HParam("fine_tune"): self.fine_tune and with_fine_tuning,
            hp.HParam("fine_tune_last_layer"): self.fine_tune_last_layer if with_fine_tuning else "",
            hp.HParam("fine_tune_lr"): self.fine_tune_lr,
            hp.HParam("fine_tune_epochs"): self.fine_tune_epochs,
            hp.HParam("fine_tune_decay"): self.fine_tune_decay,
            hp.HParam("fine_tune_decay_steps"): self.fine_tune_decay_steps,
            hp.HParam("fine_tune_decay_rate"): self.fine_tune_decay_rate,
            hp.HParam("fine_tune_patience"): self.fine_tune_patience,
            hp.HParam("fine_tune_early_stopping_start_epoch"): self.fine_tune_early_stopping_start_epoch,
            hp.HParam("weight_decay"): self.weight_decay,
            hp.HParam("sklearn_class_weight"): self.sklearn_class_weight,
            hp.HParam("merge_validation_set"): self.merge_validation_set,
        }
