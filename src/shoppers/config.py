"""Shared configuration and helpers for the shoppers pipeline.

Single source of truth for paths and for reading ``params.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# config.py -> shoppers -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
METRICS_DIR: Path = PROJECT_ROOT / "metrics"
LOGS_DIR: Path = Path(os.getenv("SHOPPERS_LOGS_DIR", str(PROJECT_ROOT / "logs")))
REPORTS_DIR: Path = Path(os.getenv("SHOPPERS_REPORTS_DIR", str(PROJECT_ROOT / "reports")))

RAW_CSV: Path = RAW_DIR / "shoppers.csv"
TRAIN_CSV: Path = PROCESSED_DIR / "train.csv"
VALID_CSV: Path = PROCESSED_DIR / "valid.csv"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METRICS_PATH: Path = METRICS_DIR / "metrics.json"
PREDICTION_LOG_PATH: Path = LOGS_DIR / "predictions.jsonl"
PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"
DVC_LOCK_PATH: Path = PROJECT_ROOT / "dvc.lock"

# --- MLflow ----------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_HOST_TRACKING_URI",
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500"),
)
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "shoppers-purchasing-intention"
)
MLFLOW_REGISTERED_MODEL: str = os.getenv("MLFLOW_REGISTERED_MODEL", "shoppers")
MLFLOW_RUN_ID_PATH: Path = MODELS_DIR / "mlflow_run_id"


def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read ``params.yaml``. Every hyperparameter lives there, never in code."""
    path = path or PARAMS_PATH
    with path.open("r", encoding="utf-8") as fh:
        loaded: Any = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"params.yaml must contain a mapping, got {type(loaded)}")
    return loaded


def get_logger(name: str) -> logging.Logger:
    """Structured-ish stdlib logging; never ``print()`` in pipeline code."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"train": {"seed": 1}}` -> `{"train.seed": 1}` -- the shape MLflow wants."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2, sort_keys=True)
        file_handler.write("\n")


def git_sha() -> str:
    """Current commit, with a '-dirty' suffix if the tree has uncommitted changes."""
    try:
        dirty: bool = git_dirty()
        sha: str = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - metadata must never break training
        return "unknown"


def git_dirty() -> bool:
    try:
        dirty: str = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
        ).stdout.strip()
        return bool(dirty)
    except Exception:  # noqa: BLE001 - metadata must never break training
        return True


def dvc_data_hash(dep_suffix: str = "shoppers.csv") -> str:
    """Content hash of the raw dataset, taken from dvc.lock."""
    if not DVC_LOCK_PATH.exists():
        return "unknown"
    lock = yaml.safe_load(DVC_LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        return "unknown"
    stages = lock.get("stages", {})
    if not isinstance(stages, dict):
        return "unknown"
    prepare_stage = stages.get("prepare", {})
    if not isinstance(prepare_stage, dict):
        return "unknown"
    deps = prepare_stage.get("deps", [])
    if not isinstance(deps, list):
        return "unknown"
    for dep in deps:
        if isinstance(dep, dict) and str(dep.get("path", "")).endswith(dep_suffix):
            return str(dep.get("md5", "unknown"))
    return "unknown"


def lineage_tags() -> dict[str, Any]:
    """Tags that connect DVC data lineage to MLflow runs."""
    return {
        "git_sha": git_sha(),
        "dvc_data_hash": dvc_data_hash(),
        "pipeline": "dvc",
        "dirty": git_dirty(),
    }


def configure_mlflow() -> None:
    """Point the MLflow client at the configured tracking URI."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
