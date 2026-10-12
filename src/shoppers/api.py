"""FastAPI service exposing /train and /predict endpoints.

Run:  uvicorn src.shoppers.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from logging import Logger
from typing import Any

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from src.shoppers import config, evaluate, prepare, train
from src.shoppers.schemas import (
    HealthResponse,
    PredictRequest,
    PredictInstanceResult,
    PredictResponse,
    TrainResponse,
)

log: Logger = config.get_logger(__name__)

app = FastAPI(
    title="Shoppers Purchasing Intention API",
    description="Train and predict endpoints for the Online Shoppers MLOps project.",
    version="0.1.0",
)

_model_cache: Pipeline | None = None
_model_uri: str = "local://models/model.joblib"


def _load_model() -> Pipeline:
    global _model_cache, _model_uri
    if _model_cache is not None:
        return _model_cache
    if not config.MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Call POST /train.")
    loaded: Pipeline = joblib.load(config.MODEL_PATH)
    _model_cache = loaded
    _model_uri = f"file://{config.MODEL_PATH}"
    return loaded


def _reset_model_cache() -> None:
    global _model_cache, _model_uri
    _model_cache = None
    _model_uri = "local://models/model.joblib"


def _append_prediction_log(records: list[dict[str, Any]]) -> None:
    config.ensure_dirs(config.LOGS_DIR)
    with config.PREDICTION_LOG_PATH.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")


def _run_id_or_none() -> str | None:
    if config.MLFLOW_RUN_ID_PATH.exists():
        return config.MLFLOW_RUN_ID_PATH.read_text(encoding="utf-8").strip() or None
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=config.MODEL_PATH.exists())


@app.post("/train", response_model=TrainResponse)
def train_endpoint() -> TrainResponse:
    """Run prepare -> train -> evaluate and return metrics."""
    try:
        prepare.main()
        run_id = train.main()
        metrics = evaluate.main()
        _reset_model_cache()
        _load_model()

        config.configure_mlflow()
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("endpoint", "train")
            if config.METRICS_PATH.exists():
                mlflow.log_artifact(str(config.METRICS_PATH))

        return TrainResponse(
            status="ok",
            run_id=run_id,
            metrics=dict(metrics),
            model_path=str(config.MODEL_PATH),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface training failures to clients
        log.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest) -> PredictResponse:
    """Score one or more feature rows and append request/prediction logs."""
    model = _load_model()
    rows = [inst.model_dump() for inst in body.instances]
    frame = pd.DataFrame(rows)

    # Match prepare casting: categorical columns as strings
    params = config.load_params()
    for col in params["prepare"].get("categorical_columns", []):
        if col in frame.columns:
            frame[col] = frame[col].astype(str)

    try:
        proba = model.predict_proba(frame)[:, 1]
        preds = (proba >= float(params["evaluate"]["threshold"])).astype(int)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    results = [
        PredictInstanceResult(prediction=int(p), probability=round(float(pr), 6))
        for p, pr in zip(preds, proba, strict=True)
    ]

    ts = datetime.now(timezone.utc).isoformat()
    run_id = _run_id_or_none()
    log_records: list[dict[str, Any]] = []
    for row, result in zip(rows, results, strict=True):
        log_records.append(
            {
                "timestamp": ts,
                "request": row,
                "prediction": result.prediction,
                "probability": result.probability,
                "run_id": run_id,
                "model_uri": _model_uri,
            }
        )
    _append_prediction_log(log_records)

    # Log a small predict artifact to MLflow when a tracking run exists
    try:
        config.configure_mlflow()
        config.ensure_dirs(config.LOGS_DIR)
        artifact_path = config.LOGS_DIR / f"predict_batch_{ts.replace(':', '-')}.json"
        artifact_path.write_text(json.dumps(log_records, indent=2), encoding="utf-8")
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(artifact_path))
                mlflow.set_tag("endpoint", "predict")
        else:
            with mlflow.start_run(run_name=f"predict-{ts}"):
                mlflow.log_artifact(str(artifact_path))
                mlflow.set_tag("endpoint", "predict")
    except Exception as exc:  # noqa: BLE001 - prediction must succeed even if logging fails
        log.warning("MLflow predict artifact logging failed: %s", exc)

    return PredictResponse(predictions=results, model_uri=_model_uri, run_id=run_id)


def create_app() -> FastAPI:
    """Factory used by tests and ASGI servers."""
    return app
