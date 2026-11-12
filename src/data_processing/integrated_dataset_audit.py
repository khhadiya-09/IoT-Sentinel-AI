from pathlib import Path
import argparse
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXPERIMENT_ID = (
    "normal_baseline_v2_20261108T130132Z_0d8534df"
)

EXPECTED_COLUMNS = [
    "timestamp_epoch_packet",
    "timestamp_utc",
    "frame_number",
    "frame_length_bytes",
    "tcp_stream_packet",
    "source_port_packet",
    "destination_port",
    "mqtt_message_id_packet",
    "mqtt_topic_packet",
    "traffic_direction",
    "inter_arrival_time",
    "global_inter_arrival_time",
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
]

EXPECTED_DEVICE_INTERVALS = {
    "sensor_001": 2.0,
    "sensor_002": 3.0,
    "sensor_003": 4.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit an integrated MQTT dataset (21-column v2 schema)."
    )
    parser.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help="Experiment/run folder name containing the integrated dataset.",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        default=None,
        help="Input integrated CSV. Defaults to the v2 dataset in the experiment folder.",
    )
    parser.add_argument(
        "--report",
        dest="report_file",
        default=None,
        help="Output JSON report. Defaults to integrated_dataset_audit_v2.json.",
    )
    return parser.parse_args()


def resolve_paths(args):
    experiment_dir = PROJECT_ROOT / "data" / "processed" / args.experiment_id

    input_file = (
        Path(args.input_file).resolve()
        if args.input_file
        else experiment_dir / "mqtt_integrated_dataset_v2.csv"
    )
    report_file = (
        Path(args.report_file).resolve()
        if args.report_file
        else experiment_dir / "integrated_dataset_audit_v2.json"
    )
    return input_file, report_file


def load_dataset(input_file):
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    df = pd.read_csv(input_file)
    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)
    return df


def structural_audit(df):
    print("\n" + "=" * 65)
    print("1. STRUCTURAL AUDIT")
    print("=" * 65)

    print("\nDataset shape:", df.shape)
    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    unexpected_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing columns: {missing_columns}; "
            f"unexpected columns: {unexpected_columns}"
        )

    if len(df.columns) != 21:
        raise ValueError(f"Expected 21 columns, found {len(df.columns)}.")

    duplicate_rows = int(df.duplicated().sum())
    duplicate_frames = int(df["frame_number"].duplicated().sum())

    print("\nData types:")
    print(df.dtypes)
    print("\nDuplicate rows:", duplicate_rows)
    print("Duplicate frame numbers:", duplicate_frames)

    if duplicate_rows or duplicate_frames:
        raise ValueError("Duplicate rows or frame numbers detected.")


def timing_missing_value_audit(df):
    print("\n" + "=" * 65)
    print("2. MISSING VALUE AND TIMING AUDIT")
    print("=" * 65)

    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    print("\nTotal missing values:", int(missing.sum()))

    allowed_missing_columns = {
        "inter_arrival_time",
        "global_inter_arrival_time",
    }
    unexpected = {
        column: int(count)
        for column, count in missing.items()
        if count > 0 and column not in allowed_missing_columns
    }
    if unexpected:
        raise ValueError(f"Unexpected missing values: {unexpected}")

    ordered = df.sort_values(
        ["timestamp_epoch_packet", "frame_number"]
    ).copy()

    # Stream-level intervals: one missing value at the first packet
    # of each publisher-to-broker TCP stream.
    stream_first_indices = (
        ordered.groupby("tcp_stream_packet", sort=False).head(1).index
    )
    stream_missing_indices = set(
        ordered.index[ordered["inter_arrival_time"].isna()]
    )
    expected_stream_missing_indices = set(stream_first_indices)

    if stream_missing_indices != expected_stream_missing_indices:
        raise ValueError(
            "Stream-level timing NaNs do not match the first packet "
            "of each TCP stream. "
            f"Expected rows: {len(expected_stream_missing_indices)}, "
            f"actual rows: {len(stream_missing_indices)}."
        )

    # Global intervals: one missing value at the first publisher packet overall.
    global_first_index = ordered.index[0]
    global_missing_indices = set(
        ordered.index[ordered["global_inter_arrival_time"].isna()]
    )
    if global_missing_indices != {global_first_index}:
        raise ValueError(
            "Global timing must contain exactly one NaN at the first "
            "publisher-to-broker packet."
        )

    for column in ["inter_arrival_time", "global_inter_arrival_time"]:
        values = pd.to_numeric(ordered[column].dropna(), errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise ValueError(f"Invalid non-positive or non-numeric values in {column}.")

    print("\nStream-level timing validation:")
    print("Publisher TCP streams:", ordered["tcp_stream_packet"].nunique())
    print("Expected missing intervals:", len(expected_stream_missing_indices))
    print("Actual missing intervals:", len(stream_missing_indices))
    print("Global timing expected missing intervals: 1")
    print("Global timing actual missing intervals:", len(global_missing_indices))
    print("Timing validation: PASS")

    print("\nMissing-value pattern matches v2 expectations.")


def device_audit(df):
    print("\n" + "=" * 65)
    print("3. DEVICE DISTRIBUTION")
    print("=" * 65)

    print("\nDevice counts:")
    print(df["device_id"].value_counts())
    print("\nDevice type counts:")
    print(df["device_type"].value_counts())
    print("\nRecords by device and type:")
    print(df.groupby(["device_id", "device_type"]).size())


def telemetry_audit(df):
    print("\n" + "=" * 65)
    print("4. TELEMETRY STATISTICS")
    print("=" * 65)

    telemetry_columns = ["temperature", "humidity", "battery"]
    for column in telemetry_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Invalid or missing telemetry values in {column}.")

    print("\nOverall telemetry statistics:")
    print(df[telemetry_columns].describe().round(4))
    print("\nDevice-wise telemetry statistics:")
    print(
        df.groupby("device_id")[telemetry_columns]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )


def network_audit(df):
    print("\n" + "=" * 65)
    print("5. NETWORK FEATURE AUDIT")
    print("=" * 65)

    print("\nPacket length statistics:")
    print(df["frame_length_bytes"].describe().round(4))

    print("\nStream-level inter-arrival time statistics:")
    print(df["inter_arrival_time"].dropna().describe().round(6))

    print("\nGlobal inter-arrival time statistics:")
    print(df["global_inter_arrival_time"].dropna().describe().round(6))

    print("\nTraffic direction distribution:")
    print(df["traffic_direction"].value_counts())
    print("\nTCP stream distribution:")
    print(df["tcp_stream_packet"].value_counts())


def device_timing_audit(df):
    print("\n" + "=" * 65)
    print("6. DEVICE-SPECIFIC PUBLISHING INTERVALS")
    print("=" * 65)

    timing_df = df[["device_id", "timestamp"]].copy()
    timing_df["timestamp"] = pd.to_datetime(
        timing_df["timestamp"], utc=True, errors="coerce"
    )
    if timing_df["timestamp"].isna().any():
        raise ValueError("Invalid telemetry timestamps detected.")

    timing_df = timing_df.sort_values(["device_id", "timestamp"])
    timing_df["device_publish_interval"] = (
        timing_df.groupby("device_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    valid_intervals = timing_df["device_publish_interval"].dropna()
    if (valid_intervals <= 0).any():
        raise ValueError("Invalid device publishing intervals detected.")

    statistics = (
        timing_df.groupby("device_id")["device_publish_interval"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(6)
    )

    print("\nDevice-specific publishing interval statistics:")
    print(statistics)

    timing_report = {}
    print("\nConfigured versus measured intervals:")

    for device_id, expected in EXPECTED_DEVICE_INTERVALS.items():
        if device_id not in statistics.index:
            raise ValueError(f"Expected device not found: {device_id}")

        measured = float(statistics.loc[device_id, "mean"])
        timing_report[device_id] = {
            "configured_interval_seconds": expected,
            "measured_mean_seconds": measured,
            "difference_seconds": measured - expected,
            "count": int(statistics.loc[device_id, "count"]),
            "standard_deviation_seconds": (
                None
                if pd.isna(statistics.loc[device_id, "std"])
                else float(statistics.loc[device_id, "std"])
            ),
            "minimum_seconds": float(statistics.loc[device_id, "min"]),
            "maximum_seconds": float(statistics.loc[device_id, "max"]),
        }
        print(
            f"{device_id}: configured={expected:.2f}s, "
            f"measured={measured:.6f}s, "
            f"difference={measured - expected:+.6f}s"
        )

    # Flood-client timing is reported but is not compared with the
    # normal simulator's configured device intervals.
    return timing_report


def timestamp_audit(df):
    print("\n" + "=" * 65)
    print("7. TIMESTAMP CONSISTENCY AUDIT")
    print("=" * 65)

    packet_time = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    payload_time = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

    if packet_time.isna().any() or payload_time.isna().any():
        raise ValueError("Invalid timestamps detected.")

    difference_ms = (packet_time - payload_time).dt.total_seconds() * 1000

    print("\nPacket capture timestamp range:")
    print(packet_time.min(), "to", packet_time.max())
    print("\nPayload timestamp range:")
    print(payload_time.min(), "to", payload_time.max())
    print("\nPacket-payload timestamp difference (ms):")
    print(difference_ms.describe().round(6))
    print("\nNegative differences:", int((difference_ms < 0).sum()))

    return difference_ms


def label_audit(df):
    print("\n" + "=" * 65)
    print("8. LABEL AND EXPERIMENT AUDIT")
    print("=" * 65)

    print("\nTraffic labels:")
    print(df["traffic_label"].value_counts())
    print("\nDevice status:")
    print(df["status"].value_counts())
    print("\nExperiment IDs:")
    print(df["experiment_id"].value_counts())

    dataset_experiment_ids = df["experiment_id"].dropna().unique()

    if df["experiment_id"].isna().any() or len(dataset_experiment_ids) != 1:
        raise ValueError(
            "Dataset must contain exactly one non-empty experiment_id."
        )

    print(
        "\nDataset experiment ID is consistent:",
        str(dataset_experiment_ids[0]),
    )


def range_audit(df):
    print("\n" + "=" * 65)
    print("9. VALUE RANGE AUDIT")
    print("=" * 65)

    numeric_columns = [
        "timestamp_epoch_packet",
        "frame_length_bytes",
        "source_port_packet",
        "destination_port",
        "temperature",
        "humidity",
        "battery",
        "inter_arrival_time",
        "global_inter_arrival_time",
    ]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    checks = {
        "numeric_features_are_finite": bool(
            numeric.drop(columns=[
                "inter_arrival_time",
                "global_inter_arrival_time",
            ]).notna().all().all()
            and numeric.drop(columns=[
                "inter_arrival_time",
                "global_inter_arrival_time",
            ]).map(lambda value: abs(value) != float("inf")).all().all()
        ),
        "humidity_between_0_and_100": bool(df["humidity"].between(0, 100).all()),
        "battery_between_0_and_100": bool(df["battery"].between(0, 100).all()),
        "frame_length_positive": bool((df["frame_length_bytes"] > 0).all()),
        "destination_port_is_1883": bool((df["destination_port"] == 1883).all()),
        "all_packets_publisher_to_broker": bool(
            (df["traffic_direction"] == "publisher_to_broker").all()
        ),
        "timing_values_positive_when_present": bool(
            all(
                pd.to_numeric(df[column], errors="coerce").dropna().gt(0).all()
                for column in ["inter_arrival_time", "global_inter_arrival_time"]
            )
        ),
    }

    for name, result in checks.items():
        print(f"{name}: {'PASS' if result else 'FAIL'}")

    failed = [name for name, result in checks.items() if not result]
    if failed:
        raise ValueError(f"Range validation failed: {failed}")

    return checks


def save_report(
    df,
    difference_ms,
    timing_report,
    checks,
    experiment_id,
    input_file,
    report_file,
):
    report = {
        "run_id": experiment_id,
        "dataset_experiment_id": str(df["experiment_id"].iloc[0]),
        "dataset_file": str(input_file),
        "dataset_schema_version": "v2",
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_frame_numbers": int(df["frame_number"].duplicated().sum()),
        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
        },
        "device_distribution": {
            str(key): int(value)
            for key, value in df["device_id"].value_counts().items()
        },
        "traffic_label_distribution": {
            str(key): int(value)
            for key, value in df["traffic_label"].value_counts().items()
        },
        "stream_inter_arrival_seconds": {
            "mean": float(df["inter_arrival_time"].mean()),
            "median": float(df["inter_arrival_time"].median()),
            "minimum": float(df["inter_arrival_time"].min()),
            "maximum": float(df["inter_arrival_time"].max()),
        },
        "global_inter_arrival_seconds": {
            "mean": float(df["global_inter_arrival_time"].mean()),
            "median": float(df["global_inter_arrival_time"].median()),
            "minimum": float(df["global_inter_arrival_time"].min()),
            "maximum": float(df["global_inter_arrival_time"].max()),
        },
        "device_publish_intervals": timing_report,
        "timestamp_difference_ms": {
            "mean": float(difference_ms.mean()),
            "median": float(difference_ms.median()),
            "minimum": float(difference_ms.min()),
            "maximum": float(difference_ms.max()),
        },
        "range_checks": checks,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, allow_nan=False)

    print("\nAudit report saved:")
    print(report_file)


def main():
    args = parse_args()
    input_file, report_file = resolve_paths(args)

    print("\nStarting integrated dataset audit (v2)...")
    print("Experiment ID:", args.experiment_id)

    df = load_dataset(input_file)

    structural_audit(df)
    timing_missing_value_audit(df)
    device_audit(df)
    telemetry_audit(df)
    network_audit(df)
    timing_report = device_timing_audit(df)
    difference_ms = timestamp_audit(df)
    label_audit(df)
    checks = range_audit(df)

    save_report(
        df,
        difference_ms,
        timing_report,
        checks,
        args.experiment_id,
        input_file,
        report_file,
    )

    print("\n" + "=" * 65)
    print("INTEGRATED DATASET V2 AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
