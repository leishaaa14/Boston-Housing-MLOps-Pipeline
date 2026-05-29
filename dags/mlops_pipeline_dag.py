"""
dags/mlops_pipeline_dag.py

Main Airflow DAG — orchestrates the full MLOps pipeline.
This DAG is triggered by the watcher DAG when data changes are detected.

Flow:
  inject_drift → check_drift → preprocess → train → evaluate → export_metrics
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import sys

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/boston_mlops$")
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner":            "mlops",
    "retries":          1,
    "retry_delay":      timedelta(minutes=1),
    "email_on_failure": False,
}

with DAG(
    dag_id          = "boston_mlops_pipeline",
    default_args    = default_args,
    description     = "Boston Housing MLOps Pipeline — triggered on data drift",
    schedule_interval = None,          # triggered externally by watcher DAG
    start_date      = days_ago(1),
    catchup         = False,
    max_active_runs = 1,
    tags            = ["mlops", "boston", "training"],
) as dag:

    # ── Task 1: Preprocess ────────────────────────────────────────
    preprocess = BashOperator(
        task_id      = "preprocess",
        bash_command = f"cd {PROJECT_ROOT} && python src/preprocess.py",
    )

    # ── Task 2: Train both models ─────────────────────────────────
    train = BashOperator(
        task_id      = "train",
        bash_command = f"cd {PROJECT_ROOT} && python src/train.py",
    )

    # ── Task 3: Evaluate ─────────────────────────────────────────
    evaluate = BashOperator(
        task_id      = "evaluate",
        bash_command = f"cd {PROJECT_ROOT} && python src/evaluate.py",
    )

    # ── Task 4: Export metrics to Prometheus ──────────────────────
    export_metrics = BashOperator(
        task_id      = "export_metrics",
        bash_command = f"cd {PROJECT_ROOT} && python monitoring/metrics_exporter.py --once",
    )

    # ── Task 5: DVC commit new state ──────────────────────────────
    dvc_commit = BashOperator(
        task_id      = "dvc_commit",
        bash_command = (
            f"cd {PROJECT_ROOT} && "
            f"dvc commit -f && "
            f"git add metrics/ dvc.lock && "
            f"git commit -m 'auto: pipeline run $(date +%Y%m%d_%H%M%S)' || true"
        ),
    )

    # ── DAG dependency chain ──────────────────────────────────────
    preprocess >> train >> evaluate >> export_metrics >> dvc_commit
