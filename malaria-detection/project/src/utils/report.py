from __future__ import annotations
import json
import os
import numpy as np
import tensorflow as tf
import keras
from tensorboard import program
import subprocess
import time


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


_TB_INSTANCE = None  # 중복 실행 방지용(간단)


def start_tensorboard(log_dir: str, host: str = "127.0.0.1", port: int = 6006) -> str:
    """
    TensorBoard 서버를 파이썬에서 직접 실행하고 URL을 반환합니다.
    VSCode Remote/WSL 환경에서는 포트 포워딩으로 Windows에서 localhost:port로 접속 가능합니다.
    """
    global _TB_INSTANCE
    log_dir = os.path.abspath(log_dir)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir, "--host", host, "--port", str(port)])
    url = tb.launch()
    _TB_INSTANCE = tb
    return url


def open_in_windows_browser(url: str) -> None:
    """
    WSL2에서 Windows 쪽 기본 브라우저(대개 Chrome)로 URL 열기.
    """
    # Windows explorer로 URL 열면 기본 브라우저로 열립니다.
    try:
        subprocess.Popen(["/mnt/c/Windows/explorer.exe", url])
    except Exception:
        # fallback: cmd start
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", url])
        except Exception:
            pass


def export_model_graph(log_dir: str, model: keras.Model, sample_batch, step: int = 0):
    """
    log_dir: TensorBoard log dir (callbacks.TensorBoard와 같은 경로 권장)
    model: 그래프를 보고 싶은 모델
    sample_batch: (x0, y0) 또는 x0만
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir)

    if isinstance(sample_batch, (tuple, list)):
        x0 = sample_batch[0]
    else:
        x0 = sample_batch

    # 그래프 트레이싱은 tf.function 경로에서 가장 안정적
    @tf.function
    def forward(x):
        return model(x, training=False)

    tf.summary.trace_on(graph=True, profiler=False)
    _ = forward(x0)  # 여기서 그래프가 캡처됨

    with writer.as_default():
        tf.summary.trace_export(name="model_graph", step=step)

    writer.flush()
    tf.summary.trace_off()
