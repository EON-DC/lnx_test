from __future__ import annotations
from typing import Tuple, Optional
import tensorflow as tf
import tensorflow_datasets as tfds

from src.utils.config import Config


def _num_parallel_calls(cfg: Config):
    return (
        tf.data.AUTOTUNE
        if cfg.data.num_parallel_calls.upper() == "AUTOTUNE"
        else int(cfg.data.num_parallel_calls)
    )


def preprocess_image(
    image: tf.Tensor, label: tf.Tensor, cfg: Config
) -> Tuple[tf.Tensor, tf.Tensor]:
    # resize + rescale
    image = tf.image.resize(image, (cfg.data.img_size, cfg.data.img_size))
    image = tf.cast(image, tf.float32) / 255.0  # type: ignore

    return image, label


def augment_image(
    image: tf.Tensor, label: tf.Tensor, cfg: Config
) -> Tuple[tf.Tensor, tf.Tensor]:
    if not cfg.augment.enabled:
        return image, label

    if cfg.augment.random_flip:
        image = tf.image.random_flip_left_right(image)  # type: ignore

    if cfg.augment.random_brightness and cfg.augment.random_brightness > 0:
        image = tf.image.random_brightness(
            image, max_delta=cfg.augment.random_brightness
        )  # type: ignore

    if cfg.augment.random_contrast and cfg.augment.random_contrast > 0:
        lower = max(0.0, 1.0 - cfg.augment.random_contrast)
        upper = 1.0 + cfg.augment.random_contrast
        image = tf.image.random_contrast(image, lower=lower, upper=upper)  # type: ignore

    return image, label


def _build_tfds_malaria(
    cfg: Config,
) -> Tuple[
    tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], tfds.core.DatasetInfo
]:
    # malaria: (image, label) supervised
    ds, info = tfds.load("malaria", with_info=True, as_supervised=True, split="train")

    # split (train/val/test) from a single split
    total = info.splits["train"].num_examples
    val_n = int(total * cfg.data.val_split)
    test_n = int(total * cfg.data.test_split)
    train_n = total - val_n - test_n

    ds = ds.shuffle(cfg.data.shuffle_buffer, reshuffle_each_iteration=False)

    train_ds = ds.take(train_n)
    rest = ds.skip(train_n)
    val_ds = rest.take(val_n) if val_n > 0 else None
    test_ds = rest.skip(val_n).take(test_n) if test_n > 0 else None

    return train_ds, val_ds, test_ds, info  # type: ignore


def build_datasets(
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], int]:
    """
    Returns:
      train_ds, val_ds, test_ds(optional), num_classes
    """
    if cfg.data.dataset.startswith("tfds:"):
        name = cfg.data.dataset.split(":", 1)[1]
        if name != "malaria":
            raise ValueError(
                f"Only tfds:malaria is implemented in this template. Got: {cfg.data.dataset}"
            )

        train_raw, val_raw, test_raw, info = _build_tfds_malaria(cfg)
        num_classes = info.features["label"].num_classes  # type: ignore

    else:
        raise ValueError(
            "This template currently implements tfds:malaria only. "
            "If you want dir:/path style, tell me your folder labeling format and I'll extend it."
        )

    npc = _num_parallel_calls(cfg)

    def make(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        ds = ds.map(lambda x, y: preprocess_image(x, y, cfg), num_parallel_calls=npc)
        if training:
            ds = ds.map(lambda x, y: augment_image(x, y, cfg), num_parallel_calls=npc)
            ds = ds.shuffle(cfg.data.shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.batch(cfg.data.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make(train_raw, training=True)
    if val_raw is None:
        raise ValueError(
            "val_split=0이면 monitor/early stopping이 애매해집니다. 최소한 val_split을 주는 것을 권장합니다."
        )
    val_ds = make(val_raw, training=False)
    test_ds = make(test_raw, training=False) if test_raw is not None else None

    return train_ds, val_ds, test_ds, num_classes
