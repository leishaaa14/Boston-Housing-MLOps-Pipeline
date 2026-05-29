"""
scripts/drift_injector.py

Injects Feature Scaling Drift into the Boston dataset.
Called automatically by Airflow every 3 minutes — you never run this manually.

Method: Scale LSTAT × 2.5 and RM × 0.7 (most impactful features for MEDV).
This causes the model's RMSE to visibly worsen, triggering the full pipeline.
"""

import pandas as pd
import numpy as np
import os
import json
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("drift_injector")


def inject_drift(
    input_path:  str   = "data/raw/boston.csv",
    output_path: str   = "data/raw/boston.csv",
    lstat_scale: float = 2.5,
    rm_scale:    float = 0.7,
    seed:        int   = None,
):
    """
    Inject feature scaling drift.

    Args:
        input_path  : path to read the current dataset
        output_path : path to write the drifted dataset (usually same file)
        lstat_scale : multiplier for LSTAT (% lower-status population)
        rm_scale    : multiplier for RM (avg rooms per dwelling)
        seed        : random seed (None = vary each call for demo variety)
    """
    if not os.path.exists(input_path):
        logger.error(f"Dataset not found: {input_path}")
        raise FileNotFoundError(input_path)

    df_before = pd.read_csv(input_path)
    logger.info(f"Loaded dataset: {df_before.shape[0]} rows")
    logger.info(f"Before drift → LSTAT mean: {df_before['LSTAT'].mean():.4f} | RM mean: {df_before['RM'].mean():.4f}")

    df_after = df_before.copy()

    # ── Method 1: Feature Scaling Drift ──────────────────────────
    # LSTAT (% lower status population) — strongest negative predictor
    df_after["LSTAT"] = df_after["LSTAT"] * lstat_scale

    # RM (average rooms per dwelling) — strongest positive predictor
    df_after["RM"] = df_after["RM"] * rm_scale

    # Add slight noise for variability across runs (so each run looks different)
    rng = np.random.default_rng(seed)
    df_after["LSTAT"] += rng.normal(0, 0.5, len(df_after))
    df_after["RM"]    += rng.normal(0, 0.1, len(df_after))

    logger.info(f"After drift  → LSTAT mean: {df_after['LSTAT'].mean():.4f} | RM mean: {df_after['RM'].mean():.4f}")

    # ── Save ──────────────────────────────────────────────────────
    df_after.to_csv(output_path, index=False)
    logger.info(f"Drifted dataset saved to {output_path}")

    # ── Log drift summary ─────────────────────────────────────────
    drift_summary = {
        "timestamp":      datetime.now().isoformat(),
        "drift_type":     "feature_scaling",
        "lstat_scale":    lstat_scale,
        "rm_scale":       rm_scale,
        "lstat_before":   float(df_before["LSTAT"].mean()),
        "lstat_after":    float(df_after["LSTAT"].mean()),
        "rm_before":      float(df_before["RM"].mean()),
        "rm_after":       float(df_after["RM"].mean()),
        "rows":           int(len(df_after)),
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/drift_summary.json", "w") as f:
        json.dump(drift_summary, f, indent=2)

    logger.info(f"Drift summary: {json.dumps(drift_summary, indent=2)}")
    return drift_summary


def reset_to_original(
    original_path: str = "data/raw/boston_original.csv",
    output_path:   str = "data/raw/boston.csv"
):
    """
    Reset the dataset back to the original clean version.
    Call this to re-run the 'before drift' baseline.
    """
    if not os.path.exists(original_path):
        logger.error(f"Original dataset not found: {original_path}")
        raise FileNotFoundError(original_path)

    df = pd.read_csv(original_path)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset reset to original ({len(df)} rows)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing Drift Injector")
    parser.add_argument("--reset",       action="store_true", help="Reset to original dataset")
    parser.add_argument("--lstat-scale", type=float, default=2.5)
    parser.add_argument("--rm-scale",    type=float, default=0.7)
    args = parser.parse_args()

    if args.reset:
        reset_to_original()
    else:
        inject_drift(lstat_scale=args.lstat_scale, rm_scale=args.rm_scale)
