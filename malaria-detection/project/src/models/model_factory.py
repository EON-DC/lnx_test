from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L
from src.utils.config import Config


def build_model(cfg: Config, num_classes: int) -> tf.keras.Model:
    if cfg.model.num_classes != num_classes:
        # config와 실제 dataset이 다르면 dataset 기준을 우선
        num_classes = num_classes

    name = cfg.model.name.lower()
    if name == "simple_cnn":
        return _simple_cnn(cfg, num_classes)
    if name == "lenet_like":
        return _lenet_like(cfg, num_classes)
    if name == "resnet50":
        return _resnet50(cfg, num_classes)

    raise ValueError(f"Unknown model name: {cfg.model.name}")


def _simple_cnn(cfg: Config, num_classes: int) -> tf.keras.Model:
    inputs = L.Input(shape=(cfg.data.img_size, cfg.data.img_size, 3), name="image")

    x = L.Conv2D(32, 3, padding="same")(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.MaxPool2D()(x)

    x = L.Conv2D(64, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.MaxPool2D()(x)

    x = L.Conv2D(128, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.GlobalAveragePooling2D()(x)

    x = L.Dropout(cfg.model.dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="simple_cnn")


def _lenet_like(cfg: Config, num_classes: int) -> tf.keras.Model:
    inputs = L.Input(shape=(cfg.data.img_size, cfg.data.img_size, 3), name="image")

    x = L.Conv2D(6, 5, padding="valid", activation="relu")(inputs)
    x = L.AveragePooling2D(pool_size=2)(x)

    x = L.Conv2D(16, 5, padding="valid", activation="relu")(x)
    x = L.AveragePooling2D(pool_size=2)(x)

    x = L.Flatten()(x)
    x = L.Dense(120, activation="relu")(x)
    x = L.Dense(84, activation="relu")(x)
    x = L.Dropout(cfg.model.dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="lenet_like")


def _resnet50(cfg: Config, num_classes: int) -> tf.keras.Model:
    inputs = L.Input(shape=(cfg.data.img_size, cfg.data.img_size, 3), name="image")

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    base.trainable = cfg.model.base_trainable

    x = base.outputs[0]
    x = L.Dropout(cfg.model.dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="resnet50")
