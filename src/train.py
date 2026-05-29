
"""
STEP 3: train.py
Trains Linear Regression and Random Forest on processed data.
Logs everything to MLflow: params, metrics, models, artifacts.
This is Stage 2 of the DVC pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import json
import os
import sys

mlflow.sklearn.autolog(log_models=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("train")


def load_data():
    """Load preprocessed train/test data."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()
    logger.info(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, model_name):
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    logger.info(f"{model_name} → RMSE: {rmse:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}")
    return {"rmse": rmse, "r2": r2, "mae": mae}


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression and log to MLflow."""
    logger.info("Training Linear Regression...")

    params = {"fit_intercept": True, "model_type": "linear_regression"}

    with mlflow.start_run(run_name="LinearRegression") as run:
        # Log params
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "linear")
        mlflow.set_tag("dataset",      "boston_housing")

        # Train
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, "LinearRegression")
        mlflow.log_metrics(metrics)

        # Log feature importances (coefficients)
        coefs = dict(zip(X_train.columns, model.coef_.tolist()))
#        mlflow.log_dict(coefs, "feature_coefficients.json")

        # Log model
#        mlflow.sklearn.log_model(
 #           sk_model=model,
  #          artifact_path="model"
   #     )

        # Save locally too (for DVC tracking)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/linear_regression.pkl")

        run_id = run.info.run_id
        logger.info(f"LinearRegression MLflow run_id: {run_id}")

    return metrics, run_id


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest and log to MLflow."""
    logger.info("Training Random Forest...")

    params = {
        "n_estimators":     100,
        "max_depth":        10,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "random_state":     42,
        "model_type":       "random_forest"
    }

    with mlflow.start_run(run_name="RandomForest") as run:
        # Log params
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "ensemble")
        mlflow.set_tag("dataset",      "boston_housing")

        # Train
        model = RandomForestRegressor(
                   n_estimators=20,
                   max_depth=5,
                   random_state=42,
                   n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, "RandomForest")
        mlflow.log_metrics(metrics)

        # Log feature importances
        importances = dict(zip(
            X_train.columns,
            model.feature_importances_.tolist()
        ))
        #mlflow.log_dict(importances, "feature_importances.json")

        # Log model
    #    mlflow.sklearn.log_model(
     #       sk_model=model,
      #      artifact_path="model"
#        )

        # Save locally
        joblib.dump(model, "models/random_forest.pkl")

        run_id = run.info.run_id
        logger.info(f"RandomForest MLflow run_id: {run_id}")

    return metrics, run_id


def train():
    """Main training entry point — trains both models."""

    # ── MLflow setup ─────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment   = os.getenv("MLFLOW_EXPERIMENT_NAME", "boston_housing")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logger.info(f"MLflow tracking URI : {tracking_uri}")
    logger.info(f"MLflow experiment   : {experiment}")

    # ── Load data ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_data()

    # ── Train both models ────────────────────────────────────────
    lr_metrics, lr_run_id = train_linear_regression(X_train, X_test, y_train, y_test)
    rf_metrics, rf_run_id = train_random_forest(X_train, X_test, y_train, y_test)

    # ── Save combined metrics (DVC reads this) ───────────────────
    all_metrics = {
        "linear_regression": {**lr_metrics, "run_id": lr_run_id},
        "random_forest":     {**rf_metrics, "run_id": rf_run_id},
        "best_model":        "random_forest" if rf_metrics["rmse"] < lr_metrics["rmse"] else "linear_regression"
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Training complete. Best model: {all_metrics['best_model']}")
    logger.info(f"LR   → RMSE: {lr_metrics['rmse']:.4f} | R²: {lr_metrics['r2']:.4f}")
    logger.info(f"RF   → RMSE: {rf_metrics['rmse']:.4f} | R²: {rf_metrics['r2']:.4f}")

    return all_metrics


if __name__ == "__main__":
    train()
