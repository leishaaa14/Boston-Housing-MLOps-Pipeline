"""
STEP 4: evaluate.py
Loads saved models, runs final evaluation, saves metrics for DVC tracking.
This is Stage 3 of the DVC pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("evaluate")


def evaluate():
    logger.info("Starting evaluation stage")

    # ── Load test data ───────────────────────────────────────────
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    results = {}

    # ── Evaluate both models ─────────────────────────────────────
    models = {
        "linear_regression": "models/linear_regression.pkl",
        "random_forest":     "models/random_forest.pkl",
    }

    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}. Skipping.")
            continue

        model  = joblib.load(model_path)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(mean_absolute_error(y_test, y_pred))

        results[model_name] = {
            "rmse": rmse,
            "r2":   r2,
            "mae":  mae
        }

        logger.info(
            f"{model_name} | RMSE: {rmse:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}"
        )

        # Drift alert — flag if RMSE is significantly worse than baseline
        if rmse > 7.0:
            logger.warning(
                f"DRIFT ALERT: {model_name} RMSE={rmse:.4f} exceeds threshold 7.0!"
            )

    # ── Save evaluation metrics (DVC tracks this file) ───────────
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also write DVC-compatible metrics format
    # DVC expects: {"metric_name": value} at top level for dvc metrics show
    dvc_metrics = {
        "lr_rmse":  results.get("linear_regression", {}).get("rmse", 0),
        "lr_r2":    results.get("linear_regression", {}).get("r2",   0),
        "rf_rmse":  results.get("random_forest",     {}).get("rmse", 0),
        "rf_r2":    results.get("random_forest",     {}).get("r2",   0),
    }
    with open("metrics/dvc_metrics.json", "w") as f:
        json.dump(dvc_metrics, f, indent=2)

    logger.info("Evaluation complete. Metrics saved.")
    logger.info(f"Final metrics: {json.dumps(dvc_metrics, indent=2)}")

    return results


if __name__ == "__main__":
    evaluate()
