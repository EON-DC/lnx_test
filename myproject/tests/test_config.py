from __future__ import annotations

from pathlib import Path

import pytest

from myapp.config import ConfigError, EvalConfig, InferConfig, TrainConfig, load_config


def test_load_train_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        """
app_name: demo
mode: train
train:
  epochs: 10
  batch_size: 8
  learning_rate: 0.01
  train_data: data/train.csv
  model_path: outputs/models/model.txt
        """.strip(),
        encoding="utf-8",
    )

    config = load_config(cfg, expected_mode="train")
    assert isinstance(config, TrainConfig)
    assert config.epochs == 10
    assert config.batch_size == 8


def test_env_override_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "MYAPP_SEED=99\nMYAPP_LOG_LEVEL=DEBUG\nMYAPP_DEVICE=cuda",
        encoding="utf-8",
    )
    cfg = tmp_path / "infer.yaml"
    cfg.write_text(
        """
app_name: demo
mode: infer
infer:
  input_path: data/input.csv
  model_path: outputs/models/model.txt
  prediction_path: outputs/predictions/preds.csv
        """.strip(),
        encoding="utf-8",
    )

    config = load_config(cfg, expected_mode="infer")
    assert isinstance(config, InferConfig)
    assert config.seed == 99
    assert config.log_level == "DEBUG"
    assert config.device == "cuda"


def test_mode_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "app_name: demo\nmode: eval\neval: {report_path: outputs/reports/report.json}",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(cfg, expected_mode="train")


def test_load_eval_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        """
app_name: demo
mode: eval
eval:
  prediction_path: outputs/predictions/preds.csv
  target_path: data/target.csv
  report_path: outputs/reports/report.json
        """.strip(),
        encoding="utf-8",
    )

    config = load_config(cfg, expected_mode="eval")
    assert isinstance(config, EvalConfig)
    assert config.report_path.endswith("report.json")
