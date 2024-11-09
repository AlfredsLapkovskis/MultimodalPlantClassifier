import argparse

from config import Config
from dataset.loading import *
from .utils.model_wrappers import *
from .utils.printing import *


def run(model, test_ds, test_labels):
    y_pred_proba = model.predict(test_ds)
    y_true = test_labels

    print_metrics(y_true, y_pred_proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["unimodal", "multimodal", "late_fusion"],
        required=True,
    )
    parser.add_argument(
        "--modality",
        help="modality for unimodal model",
        type=str,
        default="",
    )
    parser.add_argument(
        "--path",
        help="path to unimodal or multimodal model",
        type=str,
        default="",
    )
    parser.add_argument(
        "--modalities",
        help="modalities for multimodal and late_fusion model",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--paths",
        help="paths for late_fusion model",
        type=str,
        nargs="+",
        default=[],
    )

    args = parser.parse_args()
    mode = args.mode
    modality = args.modality
    path = args.path
    modalities = args.modalities
    paths = args.paths

    config = Config("config.json")

    match args.mode:
        case "unimodal":
            assert modality and path

            model = UnimodalModel(path)
            test_ds = load_unimodal_dataset(config, modality, "test", batch_size=256)
            test_labels = load_unimodal_labels(config, modality, "test")
        case "multimodal":
            assert modalities and path

            model = MultimodalModel(path)
            test_ds = load_multimodal_dataset(config, "test", modalities, [], batch_size=256)
            test_labels = load_multimodal_labels(config, "test")
        case "late_fusion":
            assert modalities and paths

            model = LateFusionModel(modalities, paths)
            test_ds = load_multimodal_dataset(config, "test", modalities, [], batch_size=256)
            test_labels = load_multimodal_labels(config, "test")
        case _:
            raise ValueError("Invalid mode")

    run(model, test_ds, test_labels)