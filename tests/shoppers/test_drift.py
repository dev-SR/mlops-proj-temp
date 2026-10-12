"""Tests for Evidently drift report builders."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.shoppers import config, drift, prepare, train


def _write_prediction_logs_from_valid(n: int = 10) -> None:
    valid = pd.read_csv(config.VALID_CSV)
    features = valid.drop(columns=["Revenue"]).head(n)
    with config.PREDICTION_LOG_PATH.open("w", encoding="utf-8") as fh:
        for _, row in features.iterrows():
            fh.write(
                json.dumps(
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "request": row.to_dict(),
                        "prediction": 0,
                        "probability": 0.1,
                    },
                    default=str,
                )
                + "\n"
            )


def test_load_prediction_logs_missing(tmp_project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        drift.load_prediction_logs()


def test_load_prediction_logs_empty(tmp_project: Path) -> None:
    config.PREDICTION_LOG_PATH.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        drift.load_prediction_logs()


def test_load_reference_missing(tmp_project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        drift.load_reference_frame()


def test_load_logs_and_reference(tmp_project: Path) -> None:
    prepare.main()
    train.main()
    _write_prediction_logs_from_valid()
    current = drift.load_prediction_logs()
    assert len(current) == 10
    ref = drift.load_reference_frame()
    assert "Revenue" not in ref.columns


def test_build_drift_report_mocked(tmp_project: Path) -> None:
    prepare.main()
    train.main()
    _write_prediction_logs_from_valid()

    fake_snapshot = MagicMock()
    fake_snapshot.save_html = MagicMock()
    fake_snapshot.save_json = MagicMock(side_effect=RuntimeError("no json"))
    fake_report = MagicMock()
    fake_report.run.return_value = fake_snapshot

    with patch("evidently.Report", return_value=fake_report), patch(
        "evidently.presets.DataDriftPreset", return_value=object()
    ):
        path = drift.build_drift_report()

    assert path == config.REPORTS_DIR / "drift_report.html"
    fake_snapshot.save_html.assert_called_once()
    assert config.REPORTS_DIR.joinpath("drift_report.json").exists()


def test_build_drift_report_no_overlap(tmp_project: Path) -> None:
    ref = pd.DataFrame({"only_ref": [1, 2, 3]})
    cur = pd.DataFrame({"only_cur": [1, 2, 3]})
    with patch("evidently.Report"), patch("evidently.presets.DataDriftPreset"):
        with pytest.raises(ValueError, match="overlapping"):
            drift.build_drift_report(reference=ref, current=cur)


def test_main_calls_build(tmp_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = config.REPORTS_DIR / "x.html"
    monkeypatch.setattr(drift, "build_drift_report", lambda: sentinel)
    assert drift.main() == sentinel
