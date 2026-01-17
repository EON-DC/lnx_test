from __future__ import annotations
import argparse
import os
import numpy as np
import tensorflow as tf

from src.utils.config import load_config
from src.data.pipeline import build_datasets
from src.utils.report import save_eval_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to yaml config used in training"
    )
    parser.add_argument("--run_dir", required=True, help="runs/exp... directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Dataset (train/val/test split 로직을 그대로 사용)
    train_ds, val_ds, test_ds, num_classes = build_datasets(cfg)
    if test_ds is None:
        # 템플릿은 test_split이 0일 수 있으니 val로 대신 평가
        ds = val_ds
        print("[WARN] test_ds is None. Using val_ds for evaluation.")
    else:
        ds = test_ds

    # Load model
    model_path = os.path.join(args.run_dir, "best_savedmodel")
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    results = model.evaluate(ds, return_dict=True)
    print("[EVAL]", results)

    # Predict + confusion matrix
    y_true = []
    y_pred = []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        pred = np.argmax(probs, axis=1)
        y_pred.append(pred)
        y_true.append(batch_y.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    save_eval_report(args.run_dir, y_true, y_pred, num_classes)
    print(f"[OK] eval_report.json saved in: {args.run_dir}")


if __name__ == "__main__":
    main()
