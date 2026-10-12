"""Thin Airflow DAGs that orchestrate src.shoppers business logic."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_retrain() -> str:
    """Call prepare -> train -> evaluate (or fail if F1 gate not met)."""
    from src.shoppers import evaluate, prepare, train

    prepare.main()
    run_id = train.main()
    metrics = evaluate.main()
    min_f1 = float(metrics.get("min_f1", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    if f1 < min_f1:
        raise ValueError(f"F1 {f1} below gate {min_f1}; refusing promotion")
    return run_id


def _run_drift() -> str:
    from src.shoppers import drift

    path = drift.build_drift_report()
    return str(path)


with DAG(
    dag_id="shoppers_retrain",
    description="Retrain shoppers model and enforce min_f1 gate",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["shoppers", "mlops"],
) as retrain_dag:
    PythonOperator(
        task_id="prepare_train_evaluate",
        python_callable=_run_retrain,
    )


with DAG(
    dag_id="shoppers_drift",
    description="Build Evidently drift report from prediction logs",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["shoppers", "evidently"],
) as drift_dag:
    PythonOperator(
        task_id="build_drift_report",
        python_callable=_run_drift,
    )
