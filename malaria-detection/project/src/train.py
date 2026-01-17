from __future__ import annotations
import argparse
import os
import platform
import tensorflow as tf
import keras

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.io import make_run_dir, copy_config, save_metadata, try_git_commit_hash
from src.data.pipeline import build_datasets
from src.models.model_factory import build_model


def build_optimizer(name: str, lr: float) -> keras.optimizers.Optimizer:
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def build_metrics(num_classes: int):
    # binary/multi-class 모두 대응 (AUC는 multi_class='ovo'가 필요할 수 있지만 템플릿은 단순화)
    return [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.run.seed)

    run_dir = make_run_dir(cfg.run.out_dir, cfg.run.name)
    copy_config(args.config, run_dir)

    # Data
    train_ds, val_ds, test_ds, num_classes = build_datasets(cfg)

    # Model
    model = build_model(cfg, num_classes=num_classes)

    # Compile
    opt = build_optimizer(cfg.train.optimizer, cfg.train.learning_rate)
    model.compile(
        optimizer=opt,
        loss=cfg.train.loss,
        metrics=build_metrics(num_classes),
    )

    # Callbacks
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "logs")
    best_savedmodel_dir = os.path.join(run_dir, "best_savedmodel")

    callbacks = [
        keras.callbacks.TensorBoard(log_dir=tb_dir),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "last.weights.h5"),
            save_weights_only=True,
            save_best_only=False,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best.weights.h5"),
            save_weights_only=True,
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor=cfg.train.monitor,
            mode=cfg.train.monitor_mode,
            patience=cfg.train.early_stop_patience,
            restore_best_weights=True,
        ),
    ]

    # Metadata (실험 재현성)
    meta = {
        "run_dir": run_dir,
        "git_commit": try_git_commit_hash(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "seed": cfg.run.seed,
        "monitor": cfg.train.monitor,
    }
    save_metadata(run_dir, meta)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
    )

    # Save best model as SavedModel (배포/추론용)
    # EarlyStopping restore_best_weights=True 덕분에 여기서 저장되는 것이 "best"
    model.save(best_savedmodel_dir)

    print(f"[OK] Run saved to: {run_dir}")
    print(f"[OK] Best SavedModel: {best_savedmodel_dir}")

    # (선택) test가 있으면 여기서 즉시 평가도 가능
    if test_ds is not None:
        print("[INFO] Evaluating on test set...")
        model.evaluate(test_ds)


if __name__ == "__main__":
    main()
