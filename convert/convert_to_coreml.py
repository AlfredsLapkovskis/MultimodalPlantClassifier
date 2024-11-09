import argparse
import os
import coremltools as ct
import keras


def run(src, dst):
    path = os.path.expanduser(src)

    model = keras.saving.load_model(path, custom_objects={"loss": None})

    coreml = ct.convert(model, source="tensorflow")
    coreml.save(os.path.join(dst, "coreml.mlpackage"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        help="path to source model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dst",
        help="destination folder for CoreML model",
        type=str,
        default="",
    )

    args = parser.parse_args()

    run(args.src, args.dst)