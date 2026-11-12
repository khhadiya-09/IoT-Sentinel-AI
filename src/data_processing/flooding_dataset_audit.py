"""Audit the integrated MQTT message-flooding experiment dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp_epoch_packet", "timestamp_utc", "frame_number",
    "frame_length_bytes", "tcp_stream_packet", "source_port_packet",
    "destination_port", "mqtt_message_id_packet", "mqtt_topic_packet",
    "traffic_direction", "inter_arrival_time", "device_id", "device_type",
    "timestamp", "temperature", "humidity", "battery", "status",
    "traffic_label", "experiment_id",
}
EXPECTED_LABELS = {"normal", "message_flooding"}
EXPECTED_DEVICES = {"sensor_001", "sensor_002", "sensor_003", "flood_client_001"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def interval_summary(frame: pd.DataFrame, group_col: str, time_col: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, group in frame.groupby(group_col, dropna=False):
        times = pd.to_datetime(group[time_col], utc=True, errors="coerce").dropna().sort_values()
        intervals = times.diff().dt.total_seconds().dropna()
        result[str(key)] = numeric_summary(intervals)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an integrated MQTT message-flooding dataset.")
    parser.add_argument("--run-id", default="message_flooding_v1_20261108T142620Z_71820c58")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    raw_dir = root / "data" / "raw" / "experiments" / args.run_id
    processed_dir = root / "data" / "processed" / args.run_id
    dataset_path = processed_dir / "mqtt_integrated_dataset.csv"
    metadata_path = raw_dir / "metadata.json"
    output_path = processed_dir / "flooding_dataset_audit.json"

    for path in (dataset_path, metadata_path):
        if not path.is_file():
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            return 1

    try:
        df = pd.read_csv(dataset_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        print(f"ERROR: Could not read input files: {exc}", file=sys.stderr)
        return 1

    checks: dict[str, bool] = {}
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    checks["required_columns_present"] = not missing_columns
    if missing_columns:
        print("ERROR: Missing required columns:", missing_columns, file=sys.stderr)
        return 1

    missing = {str(k): int(v) for k, v in df.isna().sum().items() if v}
    checks["no_duplicate_rows"] = not bool(df.duplicated().any())
    checks["no_duplicate_frame_numbers"] = not bool(df["frame_number"].duplicated().any())
    checks["experiment_id_matches_metadata"] = set(df["experiment_id"].dropna().astype(str)) == {str(metadata["experiment_id"])}
    checks["expected_labels_present"] = set(df["traffic_label"].dropna().astype(str)) == EXPECTED_LABELS
    checks["expected_devices_present"] = EXPECTED_DEVICES.issubset(set(df["device_id"].dropna().astype(str)))
    checks["all_flood_records_belong_to_flood_client"] = bool(
        (df.loc[df.traffic_label == "message_flooding", "device_id"] == "flood_client_001").all()
    )
    checks["all_flood_records_use_flood_label"] = bool(
        (df.loc[df.device_id == "flood_client_001", "traffic_label"] == "message_flooding").all()
    )

    # The packet inter-arrival feature is calculated independently per TCP stream.
    missing_iat = df[df["inter_arrival_time"].isna()]
    first_frames = df.groupby("tcp_stream_packet", dropna=False)["frame_number"].min()
    expected_first_frames = set(first_frames.tolist())
    checks["missing_intervals_only_at_first_packet_of_each_stream"] = (
        len(missing_iat) == len(first_frames)
        and set(missing_iat["frame_number"].tolist()) == expected_first_frames
    )

    phase = metadata["phase_timing"]
    attack_start = pd.to_datetime(phase["attack_started_at_utc"], utc=True)
    attack_end = pd.to_datetime(phase["attack_ended_at_utc"], utc=True)
    telemetry_times = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    flood_mask = df["traffic_label"].eq("message_flooding")
    normal_mask = df["traffic_label"].eq("normal")
    checks["flood_messages_within_attack_phase"] = bool(
        (telemetry_times.loc[flood_mask] >= attack_start).all()
        and (telemetry_times.loc[flood_mask] <= attack_end).all()
    )
    checks["normal_messages_present_outside_attack_phase"] = bool(
        ((telemetry_times.loc[normal_mask] < attack_start) | (telemetry_times.loc[normal_mask] > attack_end)).any()
    )

    # Range checks are descriptive data-quality checks, not claims about device physics.
    numeric_columns = ["frame_length_bytes", "temperature", "humidity", "battery", "inter_arrival_time"]
    finite_checks = {}
    for column in numeric_columns:
        vals = pd.to_numeric(df[column], errors="coerce")
        finite_checks[column] = bool(np.isfinite(vals.dropna()).all())
    checks["numeric_values_finite"] = all(finite_checks.values())
    checks["humidity_within_0_100"] = bool(df["humidity"].between(0, 100).all())
    checks["battery_within_0_100"] = bool(df["battery"].between(0, 100).all())
    checks["positive_frame_lengths"] = bool((df["frame_length_bytes"] > 0).all())

    # Per-device telemetry intervals, and per-device packet-level intervals.
    telemetry_interval_stats = interval_summary(df, "device_id", "timestamp")
    publisher_packets = df[df["traffic_direction"].eq("publisher_to_broker")].copy()
    packet_interval_stats = {
        str(stream): numeric_summary(group["inter_arrival_time"])
        for stream, group in publisher_packets.groupby("tcp_stream_packet")
    }

    flood_times = telemetry_times.loc[flood_mask].dropna().sort_values()
    flood_intervals = flood_times.diff().dt.total_seconds().dropna()
    observed_flood_rate = (
        float((len(flood_times) - 1) / (flood_times.iloc[-1] - flood_times.iloc[0]).total_seconds())
        if len(flood_times) > 1 and (flood_times.iloc[-1] - flood_times.iloc[0]).total_seconds() > 0
        else None
    )

    timestamp_delta_ms = (
        pd.to_numeric(df["timestamp_epoch_packet"], errors="coerce")
        - pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype("int64") / 1e9
    ) * 1000

    report = {
        "audit_version": "1.0",
        "dataset": str(dataset_path.relative_to(root)),
        "metadata": str(metadata_path.relative_to(root)),
        "experiment_id": metadata.get("experiment_id"),
        "run_id": metadata.get("run_id"),
        "scenario": metadata.get("scenario"),
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names": df.columns.tolist(),
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "all_checks_passed": all(checks.values()),
        "missing_values": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_frame_numbers": int(df["frame_number"].duplicated().sum()),
        "label_distribution": distribution(df["traffic_label"]),
        "device_distribution": distribution(df["device_id"]),
        "tcp_stream_label_distribution": {
            str(stream): distribution(group["traffic_label"])
            for stream, group in df.groupby("tcp_stream_packet")
        },
        "phase_timing": {
            "attack_start_utc": phase["attack_started_at_utc"],
            "attack_end_utc": phase["attack_ended_at_utc"],
            "configured_attack_rate_messages_per_second": float(phase["attack_rate_per_second"]),
            "pre_attack_seconds": int(phase["pre_attack_seconds"]),
            "attack_seconds": int(phase["attack_seconds"]),
            "post_attack_seconds": int(phase["post_attack_seconds"]),
        },
        "observed_flooding": {
            "message_count": int(flood_mask.sum()),
            "observed_rate_messages_per_second": observed_flood_rate,
            "inter_arrival_seconds": numeric_summary(flood_intervals),
            "first_message_utc": flood_times.iloc[0].isoformat() if len(flood_times) else None,
            "last_message_utc": flood_times.iloc[-1].isoformat() if len(flood_times) else None,
        },
        "telemetry_inter_arrival_seconds_by_device": telemetry_interval_stats,
        "packet_inter_arrival_seconds_by_publisher_tcp_stream": packet_interval_stats,
        "frame_length_bytes_by_label": {
            str(label): numeric_summary(group["frame_length_bytes"])
            for label, group in publisher_packets.groupby("traffic_label")
        },
        "packet_payload_timestamp_difference_ms": numeric_summary(timestamp_delta_ms),
        "numeric_finiteness": finite_checks,
        "notes": [
            "traffic_label is the configured experiment scenario label, not independent proof of benign or malicious behavior.",
            "frame_length_bytes is the captured frame length, not MQTT payload length.",
            "The observed flooding rate is calculated from the first and last flood telemetry timestamps.",
            "TCP stream identifiers and other connection-specific fields are metadata, not automatically suitable model features.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(report), indent=4, allow_nan=False), encoding="utf-8")

    print("\n=== MQTT FLOODING DATASET AUDIT ===")
    print(f"Dataset: {dataset_path}")
    print(f"Shape: {df.shape}")
    print(f"Checks passed: {report['checks_passed']}/{report['checks_total']}")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print("\nLabel distribution:")
    for name, count in report["label_distribution"].items():
        print(f"  {name}: {count}")
    print("\nObserved flooding rate:", observed_flood_rate, "messages/second")
    print("Missing values:", missing)
    print("\nAudit report saved to:", output_path)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
