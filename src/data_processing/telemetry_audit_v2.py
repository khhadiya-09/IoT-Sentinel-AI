
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CSV_FILE = RAW_DIR / "mqtt_telemetry_v2.csv"
JSONL_FILE = RAW_DIR / "mqtt_telemetry_v2.jsonl"


NUMERIC_COLUMNS = [
    "temperature",
    "humidity",
    "battery",
]

REQUIRED_COLUMNS = [
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
    "mqtt_topic",
    "collected_at",
]


def load_csv():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    return pd.read_csv(CSV_FILE)


def load_jsonl():
    if not JSONL_FILE.exists():
        raise FileNotFoundError(f"JSONL file not found: {JSONL_FILE}")

    records = []

    with JSONL_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Invalid JSON at line {line_number}")

    return pd.DataFrame(records)


def audit_dataset():
    print("=" * 65)
    print("       IOT SENTINEL AI - TELEMETRY DATA AUDIT")
    print("=" * 65)

    csv_data = load_csv()
    jsonl_data = load_jsonl()

    # 1. Basic information
    print("\n[1] DATASET OVERVIEW")

    print("CSV shape:", csv_data.shape)
    print("JSONL records:", len(jsonl_data))
    print("CSV columns:", csv_data.columns.tolist())

    # 2. Schema validation
    print("\n[2] SCHEMA VALIDATION")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in csv_data.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("All required columns are present.")

    # 3. Missing values
    print("\n[3] MISSING VALUE ANALYSIS")

    print(csv_data.isnull().sum().to_string())

    # 4. Duplicate records
    print("\n[4] DUPLICATE ANALYSIS")

    print("Exact duplicate CSV rows:", csv_data.duplicated().sum())

    # 5. Numeric validation
    print("\n[5] NUMERIC VALUE ANALYSIS")

    for column in NUMERIC_COLUMNS:
        if column not in csv_data.columns:
            continue

        values = pd.to_numeric(
            csv_data[column],
            errors="coerce"
        )

        invalid = (
            values.isna()
            | ~np.isfinite(values)
        )

        print(f"\n{column}:")
        print("Invalid values:", int(invalid.sum()))

        if values.notna().any():
            print("Minimum:", values.min())
            print("Maximum:", values.max())
            print("Mean:", round(values.mean(), 3))
            print("Standard deviation:", round(values.std(), 3))

    # 6. Range validation
    print("\n[6] RANGE VALIDATION")

    if "humidity" in csv_data.columns:
        humidity = pd.to_numeric(
            csv_data["humidity"],
            errors="coerce"
        )

        print(
            "Humidity outside 0-100:",
            int((~humidity.between(0, 100)).sum())
        )

    if "battery" in csv_data.columns:
        battery = pd.to_numeric(
            csv_data["battery"],
            errors="coerce"
        )

        print(
            "Battery outside 0-100:",
            int((~battery.between(0, 100)).sum())
        )

    # 7. Timestamp validation
    print("\n[7] TIMESTAMP VALIDATION")

    for column in ["timestamp", "collected_at"]:
        if column not in csv_data.columns:
            continue

        parsed = pd.to_datetime(
            csv_data[column],
            errors="coerce",
            utc=True
        )

        print(
            f"Invalid {column}:",
            int(parsed.isna().sum())
        )

    # 8. Device distribution
    print("\n[8] DEVICE DISTRIBUTION")

    if "device_id" in csv_data.columns:
        print(csv_data["device_id"].value_counts().to_string())

    # 9. Traffic label distribution
    print("\n[9] TRAFFIC LABEL DISTRIBUTION")

    if "traffic_label" in csv_data.columns:
        print(
            csv_data["traffic_label"]
            .value_counts(dropna=False)
            .to_string()
        )

    # 10. Experiment distribution
    print("\n[10] EXPERIMENT DISTRIBUTION")

    if "experiment_id" in csv_data.columns:
        print(
            csv_data["experiment_id"]
            .value_counts(dropna=False)
            .to_string()
        )

    # 11. CSV and JSONL consistency
    print("\n[11] CSV AND JSONL CONSISTENCY")

    print("CSV records:", len(csv_data))
    print("JSONL records:", len(jsonl_data))

    if len(csv_data) == len(jsonl_data):
        print("Record counts match.")
    else:
        print("WARNING: Record counts do not match.")

    # 12. Final summary
    print("\n[12] AUDIT SUMMARY")

    print("Total CSV records:", len(csv_data))
    print("Total JSONL records:", len(jsonl_data))
    print("Total devices:", csv_data["device_id"].nunique())

    print("\nAudit completed.")
    print("=" * 65)


if __name__ == "__main__":
    audit_dataset()
