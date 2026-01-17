from __future__ import annotations
import json
import os
import numpy as np
import tensorflow as tf


def save_eval_report(
    run_dir: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> None:
    cm = (
        tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
        .numpy()
        .tolist()
    )

    report = {
        "num_samples": int(len(y_true)),
        "confusion_matrix": cm,
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) > 0 else None,
    }

    path = os.path.join(run_dir, "eval_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
