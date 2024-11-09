import os
import tensorflow as tf
import numpy as np
import scipy.ndimage as ndimage

from config import Config
from common.constants import *


_MIN_CROP_SIDE = 200


AUGMENTATION_BRIGHTNESS = 1 << 0
AUGMENTATION_CONTRAST = 1 << 1
AUGMENTATION_FLIP_LEFT_RIGHT = 1 << 2
AUGMENTATION_FLIP_UP_DOWN = 1 << 3
AUGMENTATION_SATURATION = 1 << 4
AUGMENTATION_CROP = 1 << 5
AUGMENTATION_ROTATION = 1 << 6


def load_unimodal_labels(
    config: Config,
    modality,
    splits: str | list[str],
):
    return np.concatenate([
        _load_labels(config.get_unimodal_labels_file_path(split, modality))
        for split in _list_splits(splits)
    ])


def load_multimodal_labels(
    config: Config,
    splits: str | list[str],
):
    return np.concatenate([
        _load_labels(config.get_multimodal_labels_file_path(split))
        for split in _list_splits(splits)
    ])


def load_unimodal_dataset(
    config: Config,
    modality,
    splits: str | list[str],
    augmentations: int=0, # bit mask
    shuffle=False,
    batch_size=None,
):
    def load_img(image_tensor):
        return _load_image(image_tensor, augmentations)

    def parse(proto):
        parsed_features = tf.io.parse_single_example(proto, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        })

        image = tf.py_function(load_img, [parsed_features["image"]], tf.float32)
        image.set_shape(IMAGE_SHAPE)
        label = parsed_features["label"]

        return image, label
    
    def dataset_for_split(split):
        file_path = config.get_unimodal_file_path(split, modality)
        return tf.data.TFRecordDataset(file_path)
    
    splits = _list_splits(splits)
    assert len(splits) > 0

    ds = dataset_for_split(splits[0])
    for split in splits[1:]:
        ds = ds.concatenate(dataset_for_split(split))

    if shuffle:
        ds = ds.shuffle(10000000)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size is not None:
        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_multimodal_dataset(
    config: Config,
    splits: str | list[str],
    modalities: list[str],
    skipped_modalities: set[str]=[],
    modality_augmentations: dict[str:int]={}, # bit masks
    shuffle=False,
    batch_size=None,
    dropout=0,
    cache_batches=False,
):
    def load_img(image_tensor, augmentations):
        return _load_image(image_tensor, augmentations)
    
    def parse(proto):
        parsed_features = tf.io.parse_single_example(proto, {
            "label": tf.io.FixedLenFeature([], tf.int64),
            **{
                m: tf.io.FixedLenFeature([], tf.string, default_value="") 
                for m in modalities if m not in skipped_modalities
            },
        })

        images = {}
        for m in modalities:
            if m in skipped_modalities:
                images[m] = _blank_image()
            else:
                augmentations = modality_augmentations[m] if m in modality_augmentations else 0
                images[m] = tf.py_function(load_img, [parsed_features[m], augmentations], tf.float32)

        label = parsed_features["label"]

        return images, label
    
    def dropout_batch(images, labels):
        return {
            modality: tf.py_function(_dropout_batch, [batch, dropout], tf.float32)
            for modality, batch in images.items()
        }, labels
    
    def dataset_for_split(split):
        file_path = config.get_multimodal_file_path(split)
        ds = tf.data.TFRecordDataset(file_path)
        if shuffle and not cache_batches:
            ds = ds.shuffle(10000000)
        ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
        if batch_size is not None:
            ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        if cache_batches:
            os.makedirs(config.cache_dir, exist_ok=True)
            ds = ds.cache(os.path.join(config.cache_dir, f"multimodal_{split}"))
        return ds

    splits = _list_splits(splits)
    assert len(splits) > 0

    ds = dataset_for_split(splits[0])
    for split in splits[1:]:
        ds = ds.concatenate(dataset_for_split(split))
    if cache_batches and shuffle:
        ds = ds.shuffle(12)
    if dropout > 0:
        ds = ds.map(dropout_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _load_labels(path):
    return np.fromfile(
        path,
        dtype=np.int64,
        sep="\n",
    )


def _load_image(image_tensor, augmentations):
    if image_tensor == "":
        return _blank_image()
    
    image = tf.image.decode_jpeg(image_tensor, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 127.5 - 1
    
    if augmentations:
        image = _augment(image, augmentations)
    
    return image


def _dropout_batch(images, dropout):
    mask = tf.cast(tf.random.uniform((len(images), 1, 1, 1)) >= dropout, dtype=tf.float32)

    return images * mask

    
def _augment(img, augmentations):
    if augmentations & AUGMENTATION_BRIGHTNESS:
        img = tf.image.adjust_brightness(img, 0.2 * _uniform())
    if augmentations & AUGMENTATION_CONTRAST:
        img = tf.image.adjust_contrast(img, 0.75 + 0.5 * _uniform())
    if augmentations & AUGMENTATION_FLIP_LEFT_RIGHT:
        img = tf.image.flip_left_right(img) if _uniform() >= 0.5 else img
    if augmentations & AUGMENTATION_FLIP_UP_DOWN:
        img = tf.image.flip_up_down(img) if _uniform() >= 0.5 else img
    if augmentations & AUGMENTATION_SATURATION:
        img = tf.image.adjust_saturation(img, 0.75 + 0.5 * _uniform())
    if augmentations & AUGMENTATION_CROP:
        crop_factor = _uniform()

        if crop_factor > 0.05:
            side = int(IMAGE_SHAPE[0] - (IMAGE_SHAPE[0] - _MIN_CROP_SIDE) * crop_factor)
            img = tf.image.central_crop(img, side / IMAGE_SHAPE[0])
            img = tf.image.resize(img, IMAGE_SHAPE_WITHOUT_CHANNELS, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if augmentations & AUGMENTATION_ROTATION:
        angle = int(-30 + 60 * _uniform())
        if abs(angle) >= 5:
            img = ndimage.rotate(img, angle, reshape=False, cval=1)

    return tf.clip_by_value(img, -1, 1)


def _blank_image():
    return tf.zeros(IMAGE_SHAPE, dtype=tf.float32)


def _uniform(rng=None):
    return rng.uniform() if rng != None else np.random.uniform()


def _list_splits(splits):
    if isinstance(splits, str):
        return [splits]
    return splits
