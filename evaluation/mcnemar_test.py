import numpy as np
import argparse

from config import Config
from dataset.loading import load_multimodal_dataset, load_multimodal_labels
from .utils.model_wrappers import MultimodalModel, LateFusionModel
from .utils.printing import print_mcnemars_test


def run(multimodal_model, late_fusion_model, modalities):
    config = Config("config.json")

    test_ds = load_multimodal_dataset(config, "test", modalities, [], batch_size=128)

    y_pred_proba_late_fusion = late_fusion_model.predict(test_ds)
    y_pred_late_fusion = np.argmax(y_pred_proba_late_fusion, axis=1)

    y_pred_proba_final = multimodal_model.predict(test_ds)
    y_pred_final = np.argmax(y_pred_proba_final, axis=1)

    y_true = load_multimodal_labels(config, "test")

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

    multimodal_model = MultimodalModel(multimodal_path)
    late_fusion_model = LateFusionModel(modalities, unimodal_paths)

    run(multimodal_model, late_fusion_model, modalities)