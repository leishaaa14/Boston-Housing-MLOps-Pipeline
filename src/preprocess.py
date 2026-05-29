"""
STEP 2: preprocess.py
Reads raw boston.csv, cleans it, splits into train/test, saves processed files.
This is Stage 1 of the DVC pipeline.
Run via: dvc repro   (not directly)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("preprocess")


def preprocess():
    logger.info("Starting preprocessing stage")

    # ── Load raw data ────────────────────────────────────────────
    raw_path = "data/raw/boston.csv"
    if not os.path.exists(raw_path):
        logger.error(f"Raw data not found at {raw_path}")
        raise FileNotFoundError(f"{raw_path} not found. Run scripts/get_data.py first.")

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # ── Basic cleaning ───────────────────────────────────────────
    initial_rows = len(df)
    df = df.dropna()                          # drop nulls
    df = df.drop_duplicates()                 # drop duplicates
    df = df[df["MEDV"] > 0]                   # remove invalid targets
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

    logger.info(f"After cleaning: {len(df)} rows (removed {initial_rows - len(df)})")

    # ── Log dataset stats (for drift detection comparison) ───────
    stats = {
        "n_rows":      int(len(df)),
        "LSTAT_mean":  float(df["LSTAT"].mean()),
        "LSTAT_std":   float(df["LSTAT"].std()),
        "RM_mean":     float(df["RM"].mean()),
        "RM_std":      float(df["RM"].std()),
        "MEDV_mean":   float(df["MEDV"].mean()),
        "MEDV_std":    float(df["MEDV"].std()),
    }
    logger.info(f"Dataset stats: {json.dumps(stats, indent=2)}")

    # ── Feature / target split ───────────────────────────────────
    feature_cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM",
                    "AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
    X = df[feature_cols]
    y = df["MEDV"]

    # ── Train / test split (80/20) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Scale features ───────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols
    )

    # ── Save processed files ─────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv",  index=False)
    y_train.to_csv("data/processed/y_train.csv",        index=False)
    y_test.to_csv("data/processed/y_test.csv",          index=False)

    # Save scaler for use during serving
    joblib.dump(scaler, "models/scaler.pkl")

    # Save stats for drift monitoring
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Preprocessing complete. All files saved.")
    return stats


if __name__ == "__main__":
    preprocess()
