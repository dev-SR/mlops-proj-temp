"""Metric computation -- ONE implementation used by train, evaluate, and API."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# sklearn stubs type zero_division narrowly; 0 is valid at runtime
_ZERO_DIVISION: Any = 0


def compute_metrics(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    proba = pipe.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(
            float(precision_score(y, y_pred, zero_division=_ZERO_DIVISION)), 4
        ),
        "recall": round(float(recall_score(y, y_pred, zero_division=_ZERO_DIVISION)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=_ZERO_DIVISION)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def _as_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected a single-column target")
        return cast(pd.Series, y.iloc[:, 0])
    return y


def slice_metrics(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    column: str,
    threshold: float = 0.5,
    min_rows: int = 30,
) -> dict[str, float]:
    """Per-segment F1 for fairness / promotion gates."""
    out: dict[str, float] = {}
    if column not in X.columns:
        return out
    y_series = _as_series(y)
    feature_names = getattr(pipe, "feature_names_in_", None)
    feature_cols = list(feature_names) if feature_names is not None else list(X.columns)
    for value, idx in X.groupby(column, observed=False).groups.items():
        if len(idx) < min_rows:
            continue
        sub_x = X.loc[idx, feature_cols]
        sub_metrics = compute_metrics(pipe, sub_x, y_series.loc[idx], threshold)
        key = f"f1_slice_{column}_{value}".replace(" ", "_")
        out[key] = sub_metrics["f1"]
    return out
