"""
dags/data_watcher_dag.py

Watcher DAG — runs every 3 minutes.
1. Injects drift into boston.csv (automatically — no manual script needed)
2. Detects the file hash has changed
3. Triggers the main mlops_pipeline_dag automatically

This is what makes the demo fully hands-free after the first run.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.api.common.trigger_dag import trigger_dag
from airflow.utils.dates import days_ago
from datetime import timedelta
import hashlib
import os
import sys
import json

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/boston_mlops")
sys.path.insert(0, PROJECT_ROOT)

WATCHED_FILE = os.path.join(PROJECT_ROOT, "data/raw/boston.csv")
HASH_FILE    = os.path.join(PROJECT_ROOT, ".last_data_hash")
RUN_COUNT    = os.path.join(PROJECT_ROOT, ".run_count")

default_args = {
    "owner":            "mlops",
    "retries":          0,
    "email_on_failure": False,
}


def inject_drift_task(**context):
    """
    Automatically injects feature scaling drift into boston.csv.
    Called by Airflow every 3 minutes — no manual script needed.

    Alternates between:
    - Even runs : inject drift   (LSTAT × 2.5, RM × 0.7)
    - Odd runs  : reset to clean (shows recovery)
    This gives you a clear before/after pattern in Grafana.
    """
    # Track run count to alternate drift / reset
    run_count = 0
    if os.path.exists(RUN_COUNT):
        with open(RUN_COUNT) as f:
            run_count = int(f.read().strip() or 0)

    run_count += 1
    with open(RUN_COUNT, "w") as f:
        f.write(str(run_count))

    sys.path.insert(0, PROJECT_ROOT)
    from scripts.drift_injector import inject_drift, reset_to_original

    if run_count % 2 == 1:
        # Odd run → inject drift
        print(f"[Run {run_count}] Injecting drift...")
        result = inject_drift(
            input_path  = WATCHED_FILE,
            output_path = WATCHED_FILE,
            lstat_scale = 2.5,
            rm_scale    = 0.7,
            seed        = run_count,  # vary noise each run
        )
        print(f"Drift injected: {json.dumps(result, indent=2)}")
    else:
        # Even run → reset to clean data (shows recovery)
        print(f"[Run {run_count}] Resetting to clean data...")
        reset_to_original(
            original_path = os.path.join(PROJECT_ROOT, "data/raw/boston_original.csv"),
            output_path   = WATCHED_FILE,
        )
        print("Dataset reset to original.")


def check_data_changed(**context):
    """
    Computes MD5 hash of boston.csv.
    Returns branch name: 'trigger_pipeline' or 'skip_no_change'.
    """
    if not os.path.exists(WATCHED_FILE):
        print(f"Watched file not found: {WATCHED_FILE}")
        return "skip_no_change"

    # Current hash
    with open(WATCHED_FILE, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    # Last known hash
    last_hash = ""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            last_hash = f.read().strip()

    print(f"Current hash : {current_hash}")
    print(f"Last hash    : {last_hash}")

    if current_hash != last_hash:
        # Save new hash
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
        print("Change detected! Triggering pipeline.")
        return "trigger_pipeline"
    else:
        print("No change detected. Skipping.")
        return "skip_no_change"


def trigger_pipeline(**context):
    """Triggers the main MLOps pipeline DAG."""
    print("Triggering boston_mlops_pipeline DAG...")
    trigger_dag(
        dag_id      = "boston_mlops_pipeline",
        run_id      = f"triggered_by_watcher_{context['ts_nodash']}",
        conf        = {"triggered_by": "data_watcher"},
        replace_microseconds = False,
    )
    print("Pipeline triggered successfully.")


with DAG(
    dag_id            = "data_watcher",
    default_args      = default_args,
    description       = "Watches for data drift every 3 min and triggers pipeline",
    schedule_interval = "*/15 * * * *",   # every 3 minutes
    start_date        = days_ago(1),
    catchup           = False,
    tags              = ["watcher", "drift"],
    max_active_runs   = 1,               # prevent overlapping runs
) as dag:

    # Task 1: Inject drift automatically
    inject = PythonOperator(
        task_id         = "inject_drift",
        python_callable = inject_drift_task,
    )

    # Task 2: Check if file actually changed
    check = BranchPythonOperator(
        task_id         = "check_data_changed",
        python_callable = check_data_changed,
    )

    # Task 3a: Trigger main pipeline (if changed)
    trigger = PythonOperator(
        task_id         = "trigger_pipeline",
        python_callable = trigger_pipeline,
    )

    # Task 3b: Skip (if no change)
    skip = BashOperator(
        task_id      = "skip_no_change",
        bash_command = "echo 'No data change detected. Skipping pipeline.'",
    )

    # DAG flow
    inject >> check >> [trigger, skip]
