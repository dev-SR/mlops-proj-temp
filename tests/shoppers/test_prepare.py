"""Tests for prepare stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.shoppers import config, prepare


def test_clean_bool_target(sample_raw_df: pd.DataFrame) -> None:
    cleaned = prepare.clean(
        sample_raw_df,
        target="Revenue",
        drop_columns=[],
        categorical_columns=["Month", "VisitorType", "Weekend"],
    )
    assert set(cleaned["Revenue"].unique()).issubset({0, 1})
    assert cleaned["Month"].dtype == object


def test_clean_string_target() -> None:
    df = pd.DataFrame(
        {
            "Revenue": ["True", "False", "yes", "no"],
            "Month": ["Nov", "May", "Nov", "May"],
            "x": [1, 2, 3, 4],
        }
    )
    cleaned = prepare.clean(df, "Revenue", [], ["Month"])
    assert cleaned["Revenue"].tolist() == [1, 0, 1, 0]


def test_clean_unparseable_target() -> None:
    df = pd.DataFrame({"Revenue": ["maybe"], "x": [1]})
    with pytest.raises(ValueError, match="Unparseable"):
        prepare.clean(df, "Revenue", [], [])


def test_clean_missing_target() -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="missing"):
        prepare.clean(df, "Revenue", [], [])


def test_clean_drops_duplicates_and_columns() -> None:
    df = pd.DataFrame(
        {
            "Revenue": [True, True, False],
            "drop_me": [1, 1, 2],
            "Month": ["A", "A", "B"],
        }
    )
    # duplicate first two rows if only Revenue+Month matter after drop
    cleaned = prepare.clean(df, "Revenue", ["drop_me"], ["Month"])
    assert "drop_me" not in cleaned.columns
    assert len(cleaned) <= 3


def test_prepare_main(tmp_project: Path) -> None:
    prepare.main()
    assert config.TRAIN_CSV.exists()
    assert config.VALID_CSV.exists()
    train = pd.read_csv(config.TRAIN_CSV)
    valid = pd.read_csv(config.VALID_CSV)
    assert len(train) + len(valid) == 60
    assert "Revenue" in train.columns


def test_prepare_main_missing_raw(tmp_project: Path) -> None:
    config.RAW_CSV.unlink()
    with pytest.raises(FileNotFoundError):
        prepare.main()
