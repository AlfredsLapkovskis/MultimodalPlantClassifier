import pandas as pd
from textwrap import dedent


def get_class_weight(df: pd.DataFrame, n_classes):
    return {c: 1 - (df["Label"] == c).sum() / len(df) for c in range(n_classes)}


def log(msg):
    print(dedent(msg))
