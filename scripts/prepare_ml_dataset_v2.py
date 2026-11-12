
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/combined_pcap_features_v2.csv"
OUTPUT_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports"

PRIMARY_OUTPUT = OUTPUT_DIR / "ml_dataset_primary_v2.csv"
BOUNDARY_OUTPUT = OUTPUT_DIR / "ml_dataset_boundary_v2.csv"
MANIFEST_OUTPUT = REPORT_DIR / "ml_dataset_manifest_v2.json"

CANDIDATE_FEATURES = [
    "packet_rate_per_second",
    "mean_packet_size_bytes",
    "std_packet_size_bytes",
    "min_packet_size_bytes",
    "max_packet_size_bytes",
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
    "mean_temperature",
    "std_temperature",
    "mean_humidity",
    "std_humidity",
    "mean_battery",
    "std_battery",
]

GROUND_TRUTH_COLUMNS = [
    "normal_message_count",
    "attack_message_count",
]

METADATA_COLUMNS = [
    "experiment_id",
    "run_id",
    "scenario",
    "window_index",
    "window_start_epoch",
    "window_end_epoch",
    "window_start_utc",
    "window_end_utc",
    "window_start_relative_seconds",
    "window_duration_seconds",
    "phase_label",
    "target",
]


def main():
    for folder in (OUTPUT_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    for path in (PRIMARY_OUTPUT, BOUNDARY_OUTPUT, MANIFEST_OUTPUT):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = (
        CANDIDATE_FEATURES
        + GROUND_TRUTH_COLUMNS
        + METADATA_COLUMNS
    )

    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) != 126:
        raise ValueError(
            f"Expected 126 source windows, found {len(df)}"
        )

    if df[required].isna().any().any():
        raise ValueError("Missing values detected in required columns.")

    if (df["attack_message_count"] < 0).any():
        raise ValueError("Negative attack-message count detected.")

    if (df["normal_message_count"] < 0).any():
        raise ValueError("Negative normal-message count detected.")

    # Derive labels exclusively from ground-truth message counts.
    df["binary_label"] = np.where(
        df["attack_message_count"] > 0,
        "attack_present",
        "attack_free",
    )

    # Keep boundary windows for secondary analysis.
    primary = df[df["phase_label"] != "mixed_boundary"].copy()
    boundary = df[df["phase_label"] == "mixed_boundary"].copy()

    if len(primary) != 114 or len(boundary) != 12:
        raise ValueError(
            f"Unexpected split: primary={len(primary)}, "
            f"boundary={len(boundary)}"
        )

    if primary["binary_label"].value_counts().to_dict() != {
        "attack_free": 102,
        "attack_present": 12,
    }:
        raise ValueError("Unexpected primary label distribution.")

    numeric = primary[CANDIDATE_FEATURES].astype(float)

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Infinite or non-finite predictor values found.")

    # Preserve metadata for grouped evaluation and traceability.
    primary_output = primary[
        METADATA_COLUMNS + GROUND_TRUTH_COLUMNS
        + ["binary_label"] + CANDIDATE_FEATURES
    ].copy()

    boundary_output = boundary[
        METADATA_COLUMNS + GROUND_TRUTH_COLUMNS
        + ["binary_label"] + CANDIDATE_FEATURES
    ].copy()

    primary_output.to_csv(PRIMARY_OUTPUT, index=False)
    boundary_output.to_csv(BOUNDARY_OUTPUT, index=False)

    manifest = {
        "dataset_version": "v2",
        "source": str(INPUT_FILE.relative_to(ROOT)),
        "primary_output": str(PRIMARY_OUTPUT.relative_to(ROOT)),
        "boundary_output": str(BOUNDARY_OUTPUT.relative_to(ROOT)),
        "source_windows": int(len(df)),
        "primary_windows": int(len(primary)),
        "boundary_windows": int(len(boundary)),
        "primary_label_distribution": {
            str(k): int(v)
            for k, v in primary["binary_label"].value_counts().items()
        },
        "boundary_label_distribution": {
            str(k): int(v)
            for k, v in boundary["binary_label"].value_counts().items()
        },
        "candidate_predictors": CANDIDATE_FEATURES,
        "excluded_predictors": (
            METADATA_COLUMNS + GROUND_TRUTH_COLUMNS
            + ["unique_tcp_streams", "unique_devices",
               "mqtt_publish_count", "telemetry_message_count",
               "packet_count", "total_bytes",
               "publish_rate_per_second"]
        ),
        "label_definition": (
            "attack_present if attack_message_count > 0; "
            "otherwise attack_free"
        ),
        "primary_boundary_policy": (
            "mixed_boundary windows excluded from primary experiment "
            "and retained separately"
        ),
        "important_limitation": (
            "Current data represents one attack family: MQTT message "
            "flooding at three configured rates. It does not establish "
            "generalisation to unseen attack families."
        ),
    }

    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    print("\n=== ML DATASET PREPARATION ===")
    print(f"Source windows       : {len(df)}")
    print(f"Primary windows      : {len(primary)}")
    print(f"Boundary windows     : {len(boundary)}")
    print("\nPrimary label distribution:")
    print(primary["binary_label"].value_counts().to_string())

    print("\nBoundary label distribution:")
    print(boundary["binary_label"].value_counts().to_string())

    print("\nCandidate predictors:")
    for feature in CANDIDATE_FEATURES:
        print(f"- {feature}")

    print("\nCreated:")
    print(PRIMARY_OUTPUT)
    print(BOUNDARY_OUTPUT)
    print(MANIFEST_OUTPUT)

    print("\n[PASS] ML dataset preparation completed.")


if __name__ == "__main__":
    main()
