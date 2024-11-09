import os
from textwrap import dedent
import numpy as np
import keras
from sklearn.utils.class_weight import compute_class_weight


def ensure_no_file_locking():
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def get_class_weight(labels, n_classes):
    return {c: 1 - (labels == c).sum() / len(labels) for c in range(n_classes)}


def get_sklearn_class_weight(labels, n_classes):
    unique = np.unique(labels)
    class_weight = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
    class_weight = dict(zip(unique, class_weight))
    return {c: class_weight[c] if c in class_weight else 0 for c in range(n_classes)}


def get_categorical_focal_crossentropy(labels, n_classes, gamma=2.0):
    class_weight = get_sklearn_class_weight(labels, n_classes)
    class_weight = [class_weight[c] for c in range(n_classes)]
    
    def loss(y_true, y_pred):
        return keras.losses.categorical_focal_crossentropy(y_true, y_pred, alpha=class_weight, gamma=gamma)
    
    return loss


def get_categorical_crossentropy_with_smoothing(smoothing):
    def loss(y_true, y_pred):
        return keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=smoothing)
        
    return loss


def log(msg):
    print(dedent(msg))
