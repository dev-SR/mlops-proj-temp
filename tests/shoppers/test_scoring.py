"""Tests for scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.shoppers import scoring


def _tiny_pipe() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], "seg": ["x", "x", "y", "y", "x", "y"]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X[["a"]], y)
    return pipe, X, y


def test_compute_metrics() -> None:
    pipe, X, y = _tiny_pipe()
    metrics = scoring.compute_metrics(pipe, X[["a"]], y, threshold=0.5)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_slice_metrics_skips_small_and_missing() -> None:
    pipe, X, y = _tiny_pipe()
    # min_rows=10 → no slices
    assert scoring.slice_metrics(pipe, X, y, "seg", min_rows=10) == {}
    # missing column
    assert scoring.slice_metrics(pipe, X[["a"]], y, "seg") == {}


def test_slice_metrics_produces_keys() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=80),
            "seg": ["A"] * 40 + ["B"] * 40,
        }
    )
    y = pd.Series((X["a"] > 0).astype(int))
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X[["a"]], y)
    out = scoring.slice_metrics(pipe, X, y, "seg", min_rows=30)
    assert "f1_slice_seg_A" in out
    assert "f1_slice_seg_B" in out
