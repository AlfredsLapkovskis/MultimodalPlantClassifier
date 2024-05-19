import argparse
import os
import coremltools as ct


def main(src, dst):
    path = os.path.expanduser(src)

    coreml =  ct.convert(path)
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

    main(args.src, args.dst)