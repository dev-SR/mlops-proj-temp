"""Evidently drift report builders (business logic for Streamlit / Airflow)."""

from __future__ import annotations

import json
from logging import Logger
from pathlib import Path
from typing import Any

import pandas as pd

from src.shoppers import config

log: Logger = config.get_logger(__name__)


def load_prediction_logs(path: Path | None = None) -> pd.DataFrame:
    """Load JSONL prediction logs into a flat feature + prediction DataFrame."""
    path = path or config.PREDICTION_LOG_PATH
    if not path.exists():
        raise FileNotFoundError(f"No prediction log at {path}. Call POST /predict first.")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            flat = dict(record.get("request", {}))
            flat["prediction"] = record.get("prediction")
            flat["probability"] = record.get("probability")
            flat["timestamp"] = record.get("timestamp")
            rows.append(flat)

    if not rows:
        raise ValueError("Prediction log is empty")
    return pd.DataFrame(rows)


def load_reference_frame(path: Path | None = None) -> pd.DataFrame:
    """Reference distribution from the validation split (without target)."""
    path = path or config.VALID_CSV
    if not path.exists():
        raise FileNotFoundError(f"Reference data missing: {path}")
    params = config.load_params()
    target = str(params["prepare"]["target"])
    df = pd.read_csv(path)
    if target in df.columns:
        df = df.drop(columns=[target])
    return df


def build_drift_report(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    output_html: Path | None = None,
    output_json: Path | None = None,
) -> Path:
    """Build an Evidently data-drift report and write HTML (+ optional JSON)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = reference if reference is not None else load_reference_frame()
    current = current if current is not None else load_prediction_logs()

    # Align columns: keep intersection of feature columns
    feature_cols = [c for c in reference.columns if c in current.columns]
    if not feature_cols:
        raise ValueError("No overlapping feature columns between reference and current")

    ref: pd.DataFrame = pd.DataFrame(reference.loc[:, feature_cols]).copy()
    cur: pd.DataFrame = pd.DataFrame(current.loc[:, feature_cols]).copy()

    # Cast categoricals similarly
    params = config.load_params()
    for col in params["prepare"].get("categorical_columns", []):
        if col in ref.columns:
            ref = ref.assign(**{col: ref[col].astype(str)})
        if col in cur.columns:
            cur = cur.assign(**{col: cur[col].astype(str)})

    report = Report([DataDriftPreset()], include_tests=True)
    snapshot = report.run(reference_data=ref, current_data=cur)

    config.ensure_dirs(config.REPORTS_DIR)
    html_path = output_html or (config.REPORTS_DIR / "drift_report.html")
    json_path = output_json or (config.REPORTS_DIR / "drift_report.json")

    snapshot.save_html(str(html_path))
    # Evidently snapshot JSON export
    try:
        snapshot.save_json(str(json_path))
    except Exception:  # noqa: BLE001 - HTML is the primary deliverable
        json_path.write_text(
            json.dumps({"n_reference": len(ref), "n_current": len(cur), "columns": feature_cols}),
            encoding="utf-8",
        )

    log.info("Drift report written to %s", html_path)
    return html_path


def main() -> Path:
    return build_drift_report()


if __name__ == "__main__":  # pragma: no cover - CLI entry via `python -m` / Airflow
    main()
