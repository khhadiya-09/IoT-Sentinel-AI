
import pandas as pd
import numpy as np
from pathlib import Path


# Locate project files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "mqtt_telemetry.csv"


def run_audit():

    print("=" * 60)
    print("       IoT SENTINEL AI - TELEMETRY DATA AUDIT")
    print("=" * 60)

    # Check whether the dataset exists
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    # Load dataset
    df = pd.read_csv(DATA_FILE)

    print("\n1. BASIC DATASET INFORMATION")
    print("-" * 40)

    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Dataset shape: {df.shape}")

    print("\n2. DEVICE INFORMATION")
    print("-" * 40)

    print(f"Unique devices: {df['device_id'].nunique()}")
    print("\nRecords per device:")
    print(df["device_id"].value_counts().sort_index())

    print("\n3. MISSING VALUE ANALYSIS")
    print("-" * 40)

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])

    print(f"Total missing values: {missing.sum()}")

    print("\n4. DUPLICATE RECORD ANALYSIS")
    print("-" * 40)

    duplicates = df.duplicated().sum()

    print(f"Duplicate records: {duplicates}")

    print("\n5. NUMERICAL FEATURE ANALYSIS")
    print("-" * 40)

    numerical_columns = [
        "temperature",
        "humidity",
        "battery"
    ]

    for column in numerical_columns:

        values = pd.to_numeric(df[column], errors="coerce")

        print(f"\n{column.upper()}")

        print(f"Minimum: {values.min()}")
        print(f"Maximum: {values.max()}")
        print(f"Mean: {values.mean():.2f}")
        print(f"Standard deviation: {values.std():.2f}")

        invalid = (
            values.isna() |
            ~np.isfinite(values)
        ).sum()

        print(f"Invalid numerical values: {invalid}")

    print("\n6. DEVICE STATUS DISTRIBUTION")
    print("-" * 40)

    print(df["status"].value_counts(dropna=False))

    print("\n7. TIMESTAMP ANALYSIS")
    print("-" * 40)

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    print(f"Invalid timestamps: {timestamps.isna().sum()}")

    if timestamps.notna().any():
        print(f"First timestamp: {timestamps.min()}")
        print(f"Last timestamp: {timestamps.max()}")

    print("\n8. MQTT TOPIC DISTRIBUTION")
    print("-" * 40)

    print(df["mqtt_topic"].value_counts(dropna=False))

    print("\n" + "=" * 60)
    print("DATA QUALITY AUDIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()
