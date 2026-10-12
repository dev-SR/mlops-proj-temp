"""DVC stage 1 -- prepare: load raw CSV, clean, split into train/valid.

Run:  python -m src.shoppers.prepare      (or: dvc repro prepare)
"""

from __future__ import annotations

from logging import Logger
from typing import Any

import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split

from src.shoppers import config

log: Logger = config.get_logger(__name__)


def _normalise_target(series: Series) -> Series:
    """Map bool / string Revenue labels to 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }
    )
    if mapped.isna().any():
        raise ValueError("Unparseable values in target column")
    return mapped.astype(int)


def clean(
    df: pd.DataFrame,
    target: str,
    drop_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Deterministic cleaning: drop columns, cast cats, normalise target to 0/1."""
    out: pd.DataFrame = df.copy()
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])

    if target not in out.columns:
        raise ValueError(f"Target column '{target}' missing from dataset")

    target_series = out.loc[:, target]
    if not isinstance(target_series, Series):
        raise ValueError(f"Target column '{target}' must be a Series")
    out = out.assign(**{target: _normalise_target(target_series).astype("int64")})

    for col in categorical_columns:
        if col in out.columns and col != target:
            out = out.assign(**{col: out[col].astype(str)})

    before: int = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.warning("Dropped %d duplicate rows", before - len(out))
    return out


def main() -> None:
    params: dict[str, Any] = config.load_params()
    seed: int = int(params["seed"])
    prepare_params: dict[str, Any] = params["prepare"]
    target: str = str(prepare_params["target"])

    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"{config.RAW_CSV} not found. Place the CSV under data/raw/ and run dvc add."
        )

    log.info("Reading %s", config.RAW_CSV)
    df: pd.DataFrame = pd.read_csv(config.RAW_CSV)
    log.info("Raw shape: %s", df.shape)

    df = clean(
        df,
        target=target,
        drop_columns=list(prepare_params.get("drop_columns", [])),
        categorical_columns=list(prepare_params.get("categorical_columns", [])),
    )

    stratify: Series | None = df.loc[:, target] if prepare_params.get("stratify", True) else None
    split = train_test_split(
        df,
        test_size=float(prepare_params["test_size"]),
        random_state=seed,
        stratify=stratify,
    )
    train_df = pd.DataFrame(split[0])
    valid_df = pd.DataFrame(split[1])

    config.ensure_dirs(config.PROCESSED_DIR)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    valid_df.to_csv(config.VALID_CSV, index=False)

    log.info(
        "train=%d valid=%d revenue_rate_train=%.4f revenue_rate_valid=%.4f",
        len(train_df),
        len(valid_df),
        float(train_df.loc[:, target].mean()),
        float(valid_df.loc[:, target].mean()),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry via `python -m` / dvc
    main()
