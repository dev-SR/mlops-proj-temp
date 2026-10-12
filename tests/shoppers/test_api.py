"""Tests for FastAPI train/predict endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.shoppers import api, config


@pytest.fixture
def client(tmp_project: Path) -> TestClient:
    api._reset_model_cache()
    return TestClient(api.app)


def _sample_instance() -> dict[str, object]:
    return {
        "Administrative": 1,
        "Administrative_Duration": 10.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 20,
        "ProductRelated_Duration": 200.0,
        "BounceRates": 0.01,
        "ExitRates": 0.02,
        "PageValues": 5.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }


def test_health_before_train(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_without_model(client: TestClient) -> None:
    resp = client.post("/predict", json={"instances": [_sample_instance()]})
    assert resp.status_code == 503


def test_train_and_predict(client: TestClient, tmp_project: Path) -> None:
    resp = client.post("/train")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert "run_id" in body
    assert "f1" in body["metrics"]

    health = client.get("/health").json()
    assert health["model_loaded"] is True

    pred = client.post("/predict", json={"instances": [_sample_instance(), _sample_instance()]})
    assert pred.status_code == 200, pred.text
    pred_body = pred.json()
    assert len(pred_body["predictions"]) == 2
    assert "probability" in pred_body["predictions"][0]
    assert config.PREDICTION_LOG_PATH.exists()
    lines = config.PREDICTION_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


def test_train_missing_raw(client: TestClient, tmp_project: Path) -> None:
    config.RAW_CSV.unlink()
    resp = client.post("/train")
    assert resp.status_code == 404


def test_create_app() -> None:
    assert api.create_app() is api.app


def test_train_generic_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.prepare, "main", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.post("/train")
    assert resp.status_code == 500


def test_predict_model_failure(client: TestClient, tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client.post("/train")

    class BadModel:
        def predict_proba(self, _frame: object) -> None:
            raise ValueError("bad features")

    monkeypatch.setattr(api, "_model_cache", BadModel())
    resp = client.post("/predict", json={"instances": [_sample_instance()]})
    assert resp.status_code == 400


def test_run_id_or_none_empty_file(tmp_project: Path) -> None:
    config.MLFLOW_RUN_ID_PATH.write_text("   ", encoding="utf-8")
    assert api._run_id_or_none() is None


def test_predict_without_run_id_logs_new_run(
    client: TestClient, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/train")
    if config.MLFLOW_RUN_ID_PATH.exists():
        config.MLFLOW_RUN_ID_PATH.unlink()
    resp = client.post("/predict", json={"instances": [_sample_instance()]})
    assert resp.status_code == 200


def test_predict_mlflow_log_failure_still_returns(
    client: TestClient, tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/train")

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("mlflow down")

    monkeypatch.setattr(api.mlflow, "start_run", _boom)
    resp = client.post("/predict", json={"instances": [_sample_instance()]})
    assert resp.status_code == 200
