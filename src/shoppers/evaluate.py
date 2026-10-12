"""DVC stage 3 -- evaluate: score the model on the validation split.

Writes metrics/metrics.json (git-tracked via dvc.yaml cache: false).
Promotes the registered model to Staging only when F1 meets the gate.
"""

from __future__ import annotations

from logging import Logger
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pandas import DataFrame

from src.shoppers import config, scoring

log: Logger = config.get_logger(__name__)


def promote_model_if_passing(f1: float, min_f1: float) -> bool:
    """Promote latest registered model version to Staging when gate passes."""
    if f1 < min_f1:
        log.warning(
            "F1 %.4f is BELOW the gate %.4f -- skipping registry promotion",
            f1,
            min_f1,
        )
        return False

    model_name = config.MLFLOW_REGISTERED_MODEL
    if not model_name:
        return False

    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            log.warning("No registered versions for %s", model_name)
            return False
        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        log.info("Promoted %s v%s to Staging (f1=%.4f)", model_name, latest.version, f1)
        return True
    except Exception as exc:  # noqa: BLE001 - promotion must not break evaluate
        log.warning("Registry promotion failed: %s", exc)
        return False


def main() -> dict[str, float | int | bool]:
    params: dict[str, Any] = config.load_params()
    target = str(params["prepare"]["target"])
    evaluate_params: dict[str, Any] = params["evaluate"]
    threshold = float(evaluate_params["threshold"])
    min_f1 = float(evaluate_params["min_f1"])

    pipe = joblib.load(config.MODEL_PATH)
    valid_df: DataFrame = pd.read_csv(config.VALID_CSV)
    X = valid_df.drop(columns=[target])
    y = pd.Series(valid_df.loc[:, target])

    metrics_float = scoring.compute_metrics(pipe, X, y, threshold)
    slice_col = str(evaluate_params.get("slice_column", "VisitorType"))
    metrics_float.update(scoring.slice_metrics(pipe, X, y, slice_col, threshold))

    metrics: dict[str, float | int | bool] = {
        **metrics_float,
        "n_valid": int(len(valid_df)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": threshold,
        "min_f1": min_f1,
    }

    promoted = metrics_float["f1"] >= min_f1
    metrics["promoted"] = promoted

    config.write_json(config.METRICS_PATH, metrics)
    log.info("Metrics: %s", metrics)

    if config.MLFLOW_RUN_ID_PATH.exists():
        run_id = config.MLFLOW_RUN_ID_PATH.read_text(encoding="utf-8").strip()
        config.configure_mlflow()
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(
                {
                    "threshold": threshold,
                    "min_f1": min_f1,
                }
            )
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() if type(v) in (int, float)
            }
            mlflow.log_metrics(numeric_metrics)
            mlflow.log_artifact(str(config.METRICS_PATH))
            mlflow.set_tag("promoted", str(promoted))

    promote_model_if_passing(metrics_float["f1"], min_f1)
    return metrics


if __name__ == "__main__":  # pragma: no cover - CLI entry via `python -m` / dvc
    main()
