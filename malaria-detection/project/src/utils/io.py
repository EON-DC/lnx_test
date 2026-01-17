from __future__ import annotations
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict


def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_run_dir(base_out_dir: str, run_name: str) -> str:
    tag = now_tag()
    run_dir = os.path.join(base_out_dir, f"{run_name}_{tag}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "best_savedmodel"), exist_ok=True)
    return run_dir


def copy_config(config_path: str, run_dir: str) -> str:
    dst = os.path.join(run_dir, "config.yaml")
    shutil.copyfile(config_path, dst)
    return dst


def try_git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def save_metadata(run_dir: str, meta: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
