import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from common.constants import N_CLASSES


def print_metrics(y_true, y_pred_proba):
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    top_5_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=5, labels=range(N_CLASSES))
    top_10_accuracy = top_k_accuracy_score(y_true, y_pred_proba, k=10, labels=range(N_CLASSES))
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    
    print("accuracy = {:.4f}".format(accuracy))
    print("top_5_accuracy = {:.4f}".format(top_5_accuracy))
    print("top_10_accuracy = {:.4f}".format(top_10_accuracy))
    print("precision_macro = {:.4f}".format(precision_macro))
    print("recall_macro = {:.4f}".format(recall_macro))
    print("f1_macro = {:.4f}".format(f1_macro))


def print_mcnemars_test(y_true, y_pred1, y_pred2):
    true1 = y_true == y_pred1
    true2 = y_true == y_pred2

    n_00 = (~true1 & ~true2).sum()
    n_01 = (~true1 & true2).sum()
    n_10 = (~true2 & true1).sum()
    n_11 = (true1 & true2).sum()

    table = np.array([
        [n_00, n_01],
        [n_10, n_11],
    ])

    print(table)

    print(mcnemar(table, exact=False, correction=True))
