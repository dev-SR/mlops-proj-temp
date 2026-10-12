"""Integration tests for train + evaluate stages."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
import yaml

from src.shoppers import config, evaluate, prepare, train


def test_build_estimator_variants() -> None:
    rf = train.build_estimator(
        "random_forest",
        0,
        {"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 1, "class_weight": "balanced"},
    )
    lr = train.build_estimator("logistic_regression", 0, {"class_weight": "balanced"})
    assert rf is not None and lr is not None
    with pytest.raises(ValueError, match="Unknown model"):
        train.build_estimator("nope", 0, {"class_weight": "balanced"})


def test_train_and_evaluate_pipeline(tmp_project: Path) -> None:
    prepare.main()
    run_id = train.main()
    assert run_id
    assert config.MODEL_PATH.exists()
    assert config.MLFLOW_RUN_ID_PATH.read_text(encoding="utf-8").strip() == run_id
    pipe = joblib.load(config.MODEL_PATH)
    assert hasattr(pipe, "predict_proba")

    metrics = evaluate.main()
    assert config.METRICS_PATH.exists()
    assert "f1" in metrics
    assert metrics["promoted"] is True  # min_f1=0.0 in fixture


def test_train_random_forest(tmp_project: Path) -> None:
    params = config.load_params()
    params["train"]["model"] = "random_forest"
    params["train"]["n_estimators"] = 5
    config.PARAMS_PATH.write_text(yaml.safe_dump(params), encoding="utf-8")
    prepare.main()
    train.main()
    assert config.MODEL_PATH.exists()


def test_evaluate_skips_promotion_when_gate_fails(
    tmp_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare.main()
    train.main()
    params = config.load_params()
    params["evaluate"]["min_f1"] = 0.99
    config.PARAMS_PATH.write_text(yaml.safe_dump(params), encoding="utf-8")
    # Force low f1 by mocking scoring
    monkeypatch.setattr(
        evaluate.scoring,
        "compute_metrics",
        lambda *a, **k: {
            "accuracy": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.1,
            "roc_auc": 0.5,
        },
    )
    metrics = evaluate.main()
    assert metrics["promoted"] is False


def test_promote_model_if_passing_no_name(tmp_project: Path) -> None:
    assert evaluate.promote_model_if_passing(1.0, 0.5) is False


def test_promote_model_success(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MLFLOW_REGISTERED_MODEL", "shoppers")
    fake_version = type("V", (), {"version": "3"})()
    client = type(
        "C",
        (),
        {
            "search_model_versions": staticmethod(lambda *_a, **_k: [fake_version]),
            "transition_model_version_stage": staticmethod(lambda **_k: None),
        },
    )()
    monkeypatch.setattr(evaluate, "MlflowClient", lambda: client)
    assert evaluate.promote_model_if_passing(0.9, 0.5) is True


def test_promote_model_no_versions(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MLFLOW_REGISTERED_MODEL", "shoppers")
    client = type(
        "C",
        (),
        {"search_model_versions": staticmethod(lambda *_a, **_k: [])},
    )()
    monkeypatch.setattr(evaluate, "MlflowClient", lambda: client)
    assert evaluate.promote_model_if_passing(0.9, 0.5) is False


def test_promote_model_exception(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MLFLOW_REGISTERED_MODEL", "shoppers")

    def _boom() -> None:
        raise RuntimeError("registry down")

    monkeypatch.setattr(evaluate, "MlflowClient", _boom)
    assert evaluate.promote_model_if_passing(0.9, 0.5) is False


def test_promote_below_gate(tmp_project: Path) -> None:
    assert evaluate.promote_model_if_passing(0.1, 0.9) is False
