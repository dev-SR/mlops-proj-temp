"""DVC stage 2 -- train: fit a scikit-learn Pipeline and persist it.

Preprocessing lives INSIDE the Pipeline to avoid train/serve skew.

Run:  python -m src.shoppers.train      (or: dvc repro train)
"""

from __future__ import annotations

from datetime import datetime
from logging import Logger
from typing import Any

import joblib
import mlflow
import mlflow.data.pandas_dataset as mlflow_pandas
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.shoppers import config, scoring

log: Logger = config.get_logger(__name__)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot for object/category/bool columns, scaling for numeric ones."""
    categorical: list[str] = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric: list[str] = [c for c in X.columns if c not in categorical]
    log.info("categorical=%d numeric=%d", len(categorical), len(numeric))

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )


def build_estimator(
    model_name: str, seed: int, train_params: dict[str, Any]
) -> RandomForestClassifier | LogisticRegression:
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(train_params["n_estimators"]),
            max_depth=int(train_params["max_depth"]),
            min_samples_leaf=int(train_params["min_samples_leaf"]),
            class_weight=train_params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight=train_params["class_weight"],
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model_name!r}")


def build_pipeline(
    X: pd.DataFrame, model_name: str, seed: int, train_params: dict[str, Any]
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", build_estimator(model_name, seed, train_params)),
        ]
    )


def convert_int_to_float(df: DataFrame) -> DataFrame:
    out = df.copy()
    int_cols = out.select_dtypes(include="int").columns
    out[int_cols] = out[int_cols].astype("float64")
    return out


def main() -> str:
    """Train the model and return the MLflow run id."""
    params: dict[str, Any] = config.load_params()
    seed: int = int(params["seed"])
    target: str = str(params["prepare"]["target"])
    train_params: dict[str, Any] = params["train"]
    evaluate_params: dict[str, Any] = params["evaluate"]

    train_df: DataFrame = convert_int_to_float(pd.read_csv(config.TRAIN_CSV))
    valid_df: DataFrame = convert_int_to_float(pd.read_csv(config.VALID_CSV))

    X_train = train_df.drop(columns=[target])
    y_train = pd.Series(train_df.loc[:, target])
    X_valid = valid_df.drop(columns=[target])
    y_valid = pd.Series(valid_df.loc[:, target])

    pipeline: Pipeline = build_pipeline(
        X_train, str(train_params["model"]), seed, train_params
    )

    config.configure_mlflow()
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        dataset = mlflow_pandas.from_pandas(
            train_df,
            name=config.TRAIN_CSV.name,
            targets=target,
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_params(config.flatten(params))

        pipeline.fit(X_train, y_train)

        threshold = float(evaluate_params["threshold"])
        metrics: dict[str, float] = scoring.compute_metrics(
            pipeline, X_valid, y_valid, threshold
        )
        slice_col = str(evaluate_params.get("slice_column", "VisitorType"))
        metrics.update(
            scoring.slice_metrics(pipeline, X_valid, y_valid, slice_col, threshold)
        )
        metrics["n_valid"] = float(len(valid_df))
        mlflow.log_metrics(metrics)

        lineage: dict[str, Any] = config.lineage_tags()
        mlflow.set_tags(
            {
                "git_sha": lineage["git_sha"],
                "dvc_data_hash": lineage["dvc_data_hash"],
                "pipeline": lineage["pipeline"],
            }
        )
        dirty_label = "dirty" if lineage["dirty"] else lineage["git_sha"][:6]
        mlflow.set_tag("mlflow.runName", f"{dirty_label} {datetime.now():%m%d_%H%M}")

        config.ensure_dirs(config.MODELS_DIR)
        joblib.dump(pipeline, config.MODEL_PATH)
        log.info("Model written to %s", config.MODEL_PATH)

        registered_model: str | None = config.MLFLOW_REGISTERED_MODEL or None
        mlflow_sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=registered_model,
            input_example=X_train.head(5),
        )
        mlflow.log_artifact(str(config.MODEL_PATH))

        run_id = run.info.run_id
        config.MLFLOW_RUN_ID_PATH.write_text(run_id)
        log.info("MLflow run_id: %s", run_id)
        return run_id


if __name__ == "__main__":  # pragma: no cover - CLI entry via `python -m` / dvc
    main()
