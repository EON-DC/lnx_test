from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import yaml


@dataclass(frozen=True)
class RunCfg:
    name: str
    out_dir: str
    seed: int


@dataclass(frozen=True)
class DataCfg:
    dataset: str
    img_size: int
    batch_size: int
    val_split: float
    test_split: float
    shuffle_buffer: int
    num_parallel_calls: str  # "AUTOTUNE" or int


@dataclass(frozen=True)
class AugmentCfg:
    enabled: bool
    random_flip: bool
    random_brightness: float
    random_contrast: float


@dataclass(frozen=True)
class ModelCfg:
    name: str
    num_classes: int
    dropout: float
    base_trainable: bool


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    optimizer: str
    learning_rate: float
    loss: str
    monitor: str
    monitor_mode: str
    early_stop_patience: int


@dataclass(frozen=True)
class ReportCfg:
    load_tb_after_fit: bool
    is_need_save_draw_auc: bool
    tensorboard_port: int  # default port


@dataclass(frozen=True)
class Config:
    run: RunCfg
    data: DataCfg
    augment: AugmentCfg
    model: ModelCfg
    train: TrainCfg
    report: ReportCfg


def _get(d: Dict[str, Any], path: str) -> Any:
    cur = d
    for p in path.split("."):
        if p not in cur:
            raise KeyError(f"Missing config key: {path}")
        cur = cur[p]
    return cur


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        run=RunCfg(
            name=_get(raw, "run.name"),
            out_dir=_get(raw, "run.out_dir"),
            seed=int(_get(raw, "run.seed")),
        ),
        data=DataCfg(
            dataset=_get(raw, "data.dataset"),
            img_size=int(_get(raw, "data.img_size")),
            batch_size=int(_get(raw, "data.batch_size")),
            val_split=float(_get(raw, "data.val_split")),
            test_split=float(_get(raw, "data.test_split")),
            shuffle_buffer=int(_get(raw, "data.shuffle_buffer")),
            num_parallel_calls=str(_get(raw, "data.num_parallel_calls")),
        ),
        augment=AugmentCfg(
            enabled=bool(_get(raw, "augment.enabled")),
            random_flip=bool(_get(raw, "augment.random_flip")),
            random_brightness=float(_get(raw, "augment.random_brightness")),
            random_contrast=float(_get(raw, "augment.random_contrast")),
        ),
        model=ModelCfg(
            name=_get(raw, "model.name"),
            num_classes=int(_get(raw, "model.num_classes")),
            dropout=float(_get(raw, "model.dropout")),
            base_trainable=bool(_get(raw, "model.base_trainable")),
        ),
        train=TrainCfg(
            epochs=int(_get(raw, "train.epochs")),
            optimizer=_get(raw, "train.optimizer"),
            learning_rate=float(_get(raw, "train.learning_rate")),
            loss=_get(raw, "train.loss"),
            monitor=_get(raw, "train.monitor"),
            monitor_mode=_get(raw, "train.monitor_mode"),
            early_stop_patience=int(_get(raw, "train.early_stop_patience")),
        ),
        report=ReportCfg(
            load_tb_after_fit=bool(_get(raw, "report.load_tb_after_fit")),
            is_need_save_draw_auc=bool(_get(raw, "report.is_need_save_draw_auc")),
            tensorboard_port=int(_get(raw, "report.tensorboard_port")),
        ),
    )
    return cfg
