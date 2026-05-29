"""
monitoring/metrics_exporter.py

Bridges MLflow metrics → Prometheus → Grafana.
Run this as a background process:
    python monitoring/metrics_exporter.py

Grafana scrapes Prometheus, which scrapes this exporter every 15s.
Grafana dashboard auto-refreshes — fully hands-free.

Also supports --once flag (called by Airflow after each training run).
"""

from prometheus_client import start_http_server, Gauge, Counter, Info
import mlflow
import json
import time
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("metrics_exporter")

# ── Prometheus Gauges ─────────────────────────────────────────────
# These are the metrics Grafana will plot on dashboards

# Per-model metrics
MODEL_RMSE = Gauge(
    "boston_model_rmse",
    "Root Mean Square Error of the model",
    ["model_type"]
)
MODEL_R2 = Gauge(
    "boston_model_r2",
    "R-squared score of the model",
    ["model_type"]
)
MODEL_MAE = Gauge(
    "boston_model_mae",
    "Mean Absolute Error of the model",
    ["model_type"]
)

# Dataset drift metrics
DATASET_LSTAT_MEAN = Gauge("boston_dataset_lstat_mean", "Mean of LSTAT feature")
DATASET_RM_MEAN    = Gauge("boston_dataset_rm_mean",    "Mean of RM feature")
DATASET_MEDV_MEAN  = Gauge("boston_dataset_medv_mean",  "Mean of MEDV target")
DATASET_N_ROWS     = Gauge("boston_dataset_n_rows",     "Number of rows in dataset")

# Pipeline counters
PIPELINE_RUNS = Counter("boston_pipeline_runs_total", "Total pipeline runs triggered")
DRIFT_DETECTED = Counter("boston_drift_detections_total", "Total drift detections")


def sync_from_mlflow():
    """
    Pull latest metrics from MLflow and push to Prometheus gauges.
    Called every 15 seconds by the background loop.
    """
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        experiment   = os.getenv("MLFLOW_EXPERIMENT_NAME", "boston_housing")

        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        exp    = client.get_experiment_by_name(experiment)
        if not exp:
            logger.warning(f"MLflow experiment '{experiment}' not found yet.")
            return

        runs = client.search_runs(
            experiment_ids = [exp.experiment_id],
            order_by       = ["start_time DESC"],
            max_results    = 10
        )

        if not runs:
            logger.warning("No MLflow runs found yet.")
            return

        # Group latest run per model type
        seen_models = set()
        for run in runs:
            model_type = run.data.params.get("model_type", "unknown")
            if model_type in seen_models:
                continue
            seen_models.add(model_type)

            m = run.data.metrics
            if "rmse" in m:
                MODEL_RMSE.labels(model_type=model_type).set(m["rmse"])
            if "r2" in m:
                MODEL_R2.labels(model_type=model_type).set(m["r2"])
            if "mae" in m:
                MODEL_MAE.labels(model_type=model_type).set(m["mae"])

            logger.info(
                f"Exported {model_type}: "
                f"RMSE={m.get('rmse',0):.4f} "
                f"R²={m.get('r2',0):.4f}"
            )

    except Exception as e:
        logger.error(f"MLflow sync error: {e}")


def sync_from_metrics_files():
    """
    Also read from local JSON files (faster, works offline from MLflow).
    """
    # Dataset stats
    stats_path = "metrics/dataset_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        DATASET_LSTAT_MEAN.set(stats.get("LSTAT_mean", 0))
        DATASET_RM_MEAN.set(stats.get("RM_mean", 0))
        DATASET_MEDV_MEAN.set(stats.get("MEDV_mean", 0))
        DATASET_N_ROWS.set(stats.get("n_rows", 0))

    # Eval metrics
    eval_path = "metrics/eval_metrics.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            evals = json.load(f)
        for model_type, metrics in evals.items():
            if "rmse" in metrics:
                MODEL_RMSE.labels(model_type=model_type).set(metrics["rmse"])
            if "r2" in metrics:
                MODEL_R2.labels(model_type=model_type).set(metrics["r2"])
            if "mae" in metrics:
                MODEL_MAE.labels(model_type=model_type).set(metrics["mae"])

    # Drift events
    drift_path = "metrics/drift_summary.json"
    if os.path.exists(drift_path):
        # Just reading it means drift was detected — increment counter
        # (counter only increments, so track via file mtime)
        pass

    logger.info("Metrics synced from local files.")


def run_once():
    """Called by Airflow after each pipeline run (--once flag)."""
    sync_from_metrics_files()
    sync_from_mlflow()
    PIPELINE_RUNS.inc()
    logger.info("One-shot metrics export complete.")


def run_background(port: int = 8000, interval: int = 15):
    """
    Background process: starts Prometheus HTTP server and syncs every 15s.
    Grafana's Prometheus datasource scrapes :8000/metrics.
    """
    logger.info(f"Starting Prometheus metrics server on :{port}")
    start_http_server(port)
    logger.info(f"Metrics available at http://localhost:{port}/metrics")

    while True:
        try:
            sync_from_metrics_files()
            sync_from_mlflow()
        except Exception as e:
            logger.error(f"Sync error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow → Prometheus Metrics Exporter")
    parser.add_argument("--once",     action="store_true", help="Export once and exit (for Airflow)")
    parser.add_argument("--port",     type=int, default=8000, help="Prometheus scrape port")
    parser.add_argument("--interval", type=int, default=15,   help="Sync interval in seconds")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_background(port=args.port, interval=args.interval)
