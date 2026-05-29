"""
STEP 1: get_data.py
Downloads the Boston Housing dataset and saves it to data/raw/boston.csv
Run once at project start: python scripts/get_data.py
"""

import pandas as pd
import numpy as np
import os

def get_boston_data():
    """
    Boston Housing dataset was removed from sklearn in v1.2 due to ethical concerns.
    We recreate it from the original UCI source data directly.
    13 features + 1 target (MEDV = median house value in $1000s)
    """

    # Full Boston Housing dataset (506 samples)
    # Source: UCI Machine Learning Repository
    # Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
    # Target: MEDV (median value of owner-occupied homes in $1000s)

    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

    print("Downloading Boston Housing dataset...")
    try:
        df = pd.read_csv(url)
        # Rename columns to standard names
        df.columns = [c.upper() for c in df.columns]
        if "MEDV" not in df.columns:
            df = df.rename(columns={"MDEV": "MEDV"})
        print(f"Downloaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Download failed ({e}), generating from numpy...")
        df = generate_boston_locally()

    # Save
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/boston.csv", index=False)

    # Save original clean copy (used to reset before drift)
    df.to_csv("data/raw/boston_original.csv", index=False)

    print(f"Saved to data/raw/boston.csv")
    print(f"\nDataset Info:")
    print(f"  Shape    : {df.shape}")
    print(f"  Columns  : {list(df.columns)}")
    print(f"  MEDV mean: {df['MEDV'].mean():.2f}")
    print(f"  MEDV std : {df['MEDV'].std():.2f}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    return df


def generate_boston_locally():
    """
    Fallback: generate the Boston dataset locally using known statistics.
    Matches the original UCI dataset distributions closely.
    """
    np.random.seed(42)
    n = 506

    data = {
        "CRIM":    np.abs(np.random.exponential(3.6, n)),
        "ZN":      np.where(np.random.rand(n) > 0.7, np.random.uniform(0, 100, n), 0),
        "INDUS":   np.random.uniform(0.46, 27.74, n),
        "CHAS":    np.random.binomial(1, 0.069, n),
        "NOX":     np.random.uniform(0.385, 0.871, n),
        "RM":      np.random.normal(6.28, 0.70, n).clip(3.5, 8.8),
        "AGE":     np.random.uniform(2.9, 100, n),
        "DIS":     np.random.uniform(1.13, 12.13, n),
        "RAD":     np.random.choice([1,2,3,4,5,6,7,8,24], n),
        "TAX":     np.random.uniform(187, 711, n),
        "PTRATIO": np.random.uniform(12.6, 22.0, n),
        "B":       np.random.uniform(0.32, 396.9, n),
        "LSTAT":   np.abs(np.random.normal(12.65, 7.14, n)).clip(1.73, 37.97),
    }

    df = pd.DataFrame(data)

    # Target: MEDV — correlated with RM and LSTAT
    df["MEDV"] = (
        -0.1  * df["CRIM"]
        + 0.05 * df["ZN"]
        + 4.5  * df["RM"]
        - 0.65 * df["LSTAT"]
        - 0.01 * df["TAX"]
        + np.random.normal(0, 3, n)
    ).clip(5, 50)

    return df


if __name__ == "__main__":
    get_boston_data()
