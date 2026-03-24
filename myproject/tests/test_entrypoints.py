from __future__ import annotations

import json
from pathlib import Path

from myapp.config import EvalConfig, InferConfig, TrainConfig
from myapp.eval import run_eval
from myapp.infer import run_infer
from myapp.train import run_train


def test_run_train_creates_model_artifact(tmp_path: Path) -> None:
    config = TrainConfig(
        app_name="demo",
        mode="train",
        output_root=str(tmp_path / "outputs"),
        seed=7,
        device="cpu",
        epochs=2,
        batch_size=4,
        learning_rate=0.005,
        train_data="data/train.csv",
        model_path=str(tmp_path / "outputs" / "models" / "model.txt"),
    )
    output = run_train(config)
    assert output.exists()
    assert "epochs=2" in output.read_text(encoding="utf-8")


def test_run_infer_creates_prediction_file(tmp_path: Path) -> None:
    config = InferConfig(
        app_name="demo",
        mode="infer",
        output_root=str(tmp_path / "outputs"),
        seed=7,
        device="cpu",
        input_path="data/input.csv",
        model_path=str(tmp_path / "outputs" / "models" / "model.txt"),
        prediction_path=str(tmp_path / "outputs" / "predictions" / "preds.csv"),
    )
    output = run_infer(config)
    text = output.read_text(encoding="utf-8")
    assert output.exists()
    assert "id,prediction" in text


def test_run_eval_creates_report(tmp_path: Path) -> None:
    config = EvalConfig(
        app_name="demo",
        mode="eval",
        output_root=str(tmp_path / "outputs"),
        seed=7,
        device="cpu",
        prediction_path=str(tmp_path / "outputs" / "predictions" / "preds.csv"),
        target_path="data/target.csv",
        report_path=str(tmp_path / "outputs" / "reports" / "eval_report.json"),
    )
    output = run_eval(config)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.exists()
    assert payload["metrics"]["accuracy"] == 0.91
