import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score

from common.constants import N_CLASSES


def print_metrics(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)

    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")

    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    accuracy = accuracy_score(y_true, y_pred)

    top_5_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=5, labels=range(N_CLASSES))
    top_10_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=10, labels=range(N_CLASSES))
    
    print(f"{precision_micro=}")
    print(f"{precision_macro=}")

    print(f"{recall_micro=}")
    print(f"{recall_macro=}")

    print(f"{f1_micro=}")
    print(f"{f1_macro=}")

    print(f"{accuracy=}")
    print(f"{top_5_accuracy=}")
    print(f"{top_10_accuracy=}")