import argparse
import pandas as pd
from itertools import combinations
from math import isnan

from config import Config
from dataset.loading import load_multimodal_dataset
from .utils.model_wrappers import MultimodalModel, LateFusionModel
from .utils.print_metrics import *


def main(multimodal_model, late_fusion_model, modalities):
    config = Config("config.json")

    for k in range(1, len(modalities)):
        for organs in combinations(modalities, k):
            organs = set(organs)
            skipped_modalities = set([o for o in modalities if o not in organs])

            test_df = pd.read_csv(config.get_multimodal_csv_file_path("test"))
            test_ds = load_multimodal_dataset(config, "test", modalities, skipped_modalities, [], batch_size=128, df=test_df)

            y_true = list(test_df["Label"])

            print(f"COMBINATION: {organs}===================")

            y_pred_proba_late_fusion = list(late_fusion_model.predict(test_ds))
            y_pred_proba_final = list(multimodal_model.predict(test_ds))
            
            removed_indices = set()

            for row in test_df.iterrows():
                if np.any([isnan(row[1][o]) for o in organs]):
                    removed_indices.add(row[0])

            y_true = [y for i, y in enumerate(y_true) if i not in removed_indices]
            y_pred_proba_late_fusion = [y for i, y in enumerate(y_pred_proba_late_fusion) if i not in removed_indices]
            y_pred_proba_final = [y for i, y in enumerate(y_pred_proba_final) if i not in removed_indices]

            print("Late fusion scores:")
            print_metrics(y_true, y_pred_proba_late_fusion)

            print("Final model scores:")
            print_metrics(y_true, y_pred_proba_final)


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

    config = Config("config.json")

    multimodal_model = MultimodalModel(multimodal_path)
    late_fusion_model = LateFusionModel(modalities, unimodal_paths)

    main(multimodal_model, late_fusion_model, modalities)