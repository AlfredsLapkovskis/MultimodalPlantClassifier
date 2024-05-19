import keras
import argparse

from common.constants import N_CLASSES, IMAGE_SHAPE


def run(src, dst):
    base_model = keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=False,
        pooling="avg",
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=IMAGE_SHAPE)
    x = base_model(inputs, training=False)
    outputs = keras.layers.Dense(
        N_CLASSES,
        activation="softmax",
    )(x)

    model = keras.Model(inputs, outputs)

    model.load_weights(src)
    model.save(dst)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        help="path to .weights.h5 file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dst",
        help="where to save model file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    run(args.src, args.dst)