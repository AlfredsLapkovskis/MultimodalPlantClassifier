import argparse
from itertools import combinations

from config import Config
from dataset.loading import load_multimodal_dataset, load_multimodal_labels
from .utils.model_wrappers import MultimodalModel, LateFusionModel
from .utils.printing import *


def run(multimodal_model, late_fusion_model, modalities):
    config = Config("config.json")

    for k in range(1, len(modalities) + 1):
        for organs in combinations(modalities, k):
            organs = set(organs)
            skipped_modalities = set([o for o in modalities if o not in organs])

            test_ds = load_multimodal_dataset(config, "test", modalities, skipped_modalities, [], batch_size=256)

            y_true = load_multimodal_labels(config, "test")

            print(f"COMBINATION: {organs}===================")

            y_pred_proba_late_fusion = list(late_fusion_model.predict(test_ds))
            y_pred_proba_final = list(multimodal_model.predict(test_ds))
            
            removed_indices = set()

            for i, row in enumerate(test_ds.unbatch()):
                if np.any([np.all(row[0][o] == 0) for o in organs]):
                    removed_indices.add(i)

            y_true = [y for i, y in enumerate(y_true) if i not in removed_indices]
            y_pred_proba_late_fusion = [y for i, y in enumerate(y_pred_proba_late_fusion) if i not in removed_indices]
            y_pred_proba_final = [y for i, y in enumerate(y_pred_proba_final) if i not in removed_indices]

            print(f"Number of instances: {len(y_true)}")

            print("Late fusion scores:")
            print_metrics(y_true, y_pred_proba_late_fusion)

            print("Final model scores:")
            print_metrics(y_true, y_pred_proba_final)

            y_pred_final = np.argmax(y_pred_proba_final, axis=1)
            y_pred_late_fusion = np.argmax(y_pred_proba_late_fusion, axis=1)

            print("McNemar's test:")
            print_mcnemars_test(y_true, y_pred_final, y_pred_late_fusion)


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

    run(multimodal_model, late_fusion_model, modalities)
