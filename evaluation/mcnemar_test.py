import numpy as np
import argparse
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from config import Config
from dataset.loading import load_multimodal_dataset
from .utils.model_wrappers import MultimodalModel, LateFusionModel


def main(multimodal_model, late_fusion_model, modalities):
    config = Config("config.json")

    test_df = pd.read_csv(config.get_multimodal_csv_file_path("test"))
    test_ds = load_multimodal_dataset(config, "test", modalities, [], batch_size=128, df=test_df)

    y_pred_proba_late_fusion = late_fusion_model.predict(test_ds)
    y_pred_late_fusion = np.argmax(y_pred_proba_late_fusion, axis=1)

    y_pred_proba_final = multimodal_model.predict(test_ds)
    y_pred_final = np.argmax(y_pred_proba_final, axis=1)

    y_true = test_df["Label"]

    true_final = y_true == y_pred_final
    true_late_fusion = y_true == y_pred_late_fusion

    n_00 = (~true_final & ~true_late_fusion).sum()
    n_01 = (~true_final & true_late_fusion).sum()
    n_10 = (~true_late_fusion & true_final).sum()
    n_11 = (true_final & true_late_fusion).sum()

    table = np.array([
        [n_00, n_01],
        [n_10, n_11],
    ])

    print(table)

    print(mcnemar(table, exact=False, correction=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--multimodal",
        help="path to multimodal model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--unimodals",
        help="paths to unimodal models",
        type=str,
        nargs="+",
        required=True,
    )

    args = parser.parse_args()
    modalities = args.modalities
    multimodal_path = args.multimodal
    unimodal_paths = args.unimodals

    multimodal_model = MultimodalModel(multimodal_path)
    late_fusion_model = LateFusionModel(modalities, unimodal_paths)

    main(multimodal_model, late_fusion_model, modalities)