"""Shared fixtures for shoppers tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(60):
        rows.append(
            {
                "Administrative": i % 5,
                "Administrative_Duration": float(i),
                "Informational": i % 3,
                "Informational_Duration": float(i) * 0.5,
                "ProductRelated": 10 + i,
                "ProductRelated_Duration": 100.0 + i,
                "BounceRates": 0.01 * (i % 5),
                "ExitRates": 0.02 * (i % 4),
                "PageValues": float(i % 10),
                "SpecialDay": 0.0,
                "Month": "Nov" if i % 2 == 0 else "May",
                "OperatingSystems": 1 + (i % 3),
                "Browser": 1 + (i % 4),
                "Region": 1 + (i % 5),
                "TrafficType": 1 + (i % 6),
                "VisitorType": "Returning_Visitor" if i % 3 else "New_Visitor",
                "Weekend": i % 2 == 0,
                "Revenue": i % 4 == 0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_project(tmp_path: Path, sample_raw_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate all config paths under tmp_path and seed a tiny raw CSV."""
    from src.shoppers import config

    raw_dir = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    models = tmp_path / "models"
    metrics = tmp_path / "metrics"
    logs = tmp_path / "logs"
    reports = tmp_path / "reports"
    for d in (raw_dir, processed, models, metrics, logs, reports):
        d.mkdir(parents=True)

    raw_csv = raw_dir / "shoppers.csv"
    sample_raw_df.to_csv(raw_csv, index=False)

    params = {
        "seed": 42,
        "prepare": {
            "test_size": 0.25,
            "target": "Revenue",
            "drop_columns": [],
            "categorical_columns": [
                "Month",
                "OperatingSystems",
                "Browser",
                "Region",
                "TrafficType",
                "VisitorType",
                "Weekend",
            ],
            "stratify": True,
        },
        "train": {
            "model": "logistic_regression",
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },
        "evaluate": {
            "threshold": 0.5,
            "min_f1": 0.0,
            "slice_column": "VisitorType",
        },
    }
    params_path = tmp_path / "params.yaml"
    import yaml

    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "METRICS_DIR", metrics)
    monkeypatch.setattr(config, "LOGS_DIR", logs)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "RAW_CSV", raw_csv)
    monkeypatch.setattr(config, "TRAIN_CSV", processed / "train.csv")
    monkeypatch.setattr(config, "VALID_CSV", processed / "valid.csv")
    monkeypatch.setattr(config, "MODEL_PATH", models / "model.joblib")
    monkeypatch.setattr(config, "METRICS_PATH", metrics / "metrics.json")
    monkeypatch.setattr(config, "PREDICTION_LOG_PATH", logs / "predictions.jsonl")
    monkeypatch.setattr(config, "PARAMS_PATH", params_path)
    monkeypatch.setattr(config, "DVC_LOCK_PATH", tmp_path / "dvc.lock")
    monkeypatch.setattr(config, "MLFLOW_RUN_ID_PATH", models / "mlflow_run_id")
    monkeypatch.setattr(config, "MLFLOW_REGISTERED_MODEL", "")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_HOST_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_NAME", "shoppers-test")

    return tmp_path
