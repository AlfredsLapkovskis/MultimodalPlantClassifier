import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage

from config import Config
from common.constants import *


_MIN_CROP_SIDE = 200


def load_unimodal_dataset(
    config: Config, 
    modality, 
    split, 
    augmentations: set[int] = {}, 
    shuffle=False, 
    batch_size=32, 
    df=None
):
    fat_file_path = config.get_plant_clef_fat_file_path()

    df = df if df is not None else pd.read_csv(config.get_unimodal_csv_file_path(split, modality))

    def load_img(idx):
        return _load_image(idx, fat_file_path, augmentations, 0)
    
    def map_tensors(x, y):
        x = tf.py_function(load_img, [x], tf.float32)
        x.set_shape(IMAGE_SHAPE)
        return x, y

    def create_dataset(df):
        ds = tf.data.Dataset.from_tensor_slices((df["Image"], df["Label"]))
        if shuffle:
            ds = ds.shuffle(len(df))
        ds = ds.map(map_tensors, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    return create_dataset(df)


def load_multimodal_dataset(
    config: Config,
    split,
    modalities: list[str],
    skipped_modalities: set[str]=[],
    augmentations: set[int]={},
    shuffle=False, 
    batch_size=32, 
    dropout=0, 
    df=None
):
    fat_file_path = config.get_plant_clef_fat_file_path()

    df = df if df is not None else pd.read_csv(config.get_multimodal_csv_file_path(split))

    def load_img(idx):
        return _load_image(idx, fat_file_path, augmentations, dropout)
    
    def map_tensors(x, y):
        images = {m: None for m in modalities}
        for _, m in enumerate(modalities):
            if m in skipped_modalities:
                img = tf.zeros(IMAGE_SHAPE, dtype=tf.float32)
            else:
                img = tf.py_function(load_img, [x[m]], tf.float32)
                img.set_shape(IMAGE_SHAPE)
            images[m] = img
        return images, y

    def create_dataset(df):
        ds = tf.data.Dataset.from_tensor_slices((dict(df.iloc[:, :-1]), df.iloc[:, -1]))
        if shuffle:
            ds = ds.shuffle(len(df))
        ds = ds.map(map_tensors, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    return create_dataset(df)


def _load_image(idx, fat_file_path, augmentations, dropout):
    if idx.dtype.is_floating and tf.math.is_nan(idx):
        return _blank_image()

    if dropout > 0 and _uniform() < dropout:
        return _blank_image()

    with open(fat_file_path, "rb") as f:
        f.seek(IMAGE_SIZE * int(idx))
        buffer = f.read(IMAGE_SIZE)
        with tf.device('/device:GPU:0'):
            buffer = tf.io.decode_raw(buffer, tf.uint8)
            buffer = tf.cast(buffer, dtype=tf.float32) / 127.5 - 1
            buffer = tf.reshape(buffer, shape=IMAGE_SHAPE)
            if augmentations:
                buffer = _augment(buffer, augmentations)
    return buffer


def _augment(img, augmentations):
    if 1 in augmentations:
        img = tf.image.adjust_brightness(img, 0.2 * _uniform())
    if 2 in augmentations:
        img = tf.image.adjust_contrast(img, 0.75 + 0.5 * _uniform())
    if 3 in augmentations:
        img = tf.image.flip_left_right(img) if _uniform() >= 0.5 else img
    if 4 in augmentations:
        img = tf.image.flip_up_down(img) if _uniform() >= 0.5 else img
    if 6 in augmentations:
        img = tf.image.adjust_saturation(img, 0.75 + 0.5 * _uniform())
    if 7 in augmentations:
        crop_factor = _uniform()

        if crop_factor > 0.05:
            side = int(IMAGE_SHAPE[0] - (IMAGE_SHAPE[0] - _MIN_CROP_SIDE) * crop_factor)
            img = tf.image.central_crop(img, side / IMAGE_SHAPE[0])
            img = tf.image.resize(img, IMAGE_SHAPE_WITHOUT_CHANNELS, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if 8 in augmentations:
        angle = int(-30 + 60 * _uniform())
        if abs(angle) >= 5:
            img = ndimage.rotate(img, angle, reshape=False, cval=1)

    return tf.clip_by_value(img, -1, 1)


def _blank_image():
    return tf.zeros(IMAGE_SHAPE, dtype=tf.float32)


def _uniform(rng=None):
    return rng.uniform() if rng != None else np.random.uniform()