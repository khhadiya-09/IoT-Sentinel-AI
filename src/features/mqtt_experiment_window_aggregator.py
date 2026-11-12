"""
Unified MQTT experiment window aggregator (v2).

Processes the validated v2 integrated datasets for the normal-baseline and
message-flooding experiments using consistent publisher-to-broker features.

Outputs are written beside each v2 integrated dataset:
    mqtt_window_features_1s_v2.csv
    mqtt_window_features_10s_v2.csv

Empty windows are preserved. Phase labels, traffic labels, device identifiers,
and label-derived counts are retained for analysis but are not model predictors
by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXPERIMENTS = [
    {
        "name": "normal_baseline_v2",
        "dataset": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "normal_baseline_v2_20261108T130132Z_0d8534df"
            / "mqtt_integrated_dataset_v2.csv"
        ),
        "metadata": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "experiments"
            / "normal_baseline_v2_20261108T130132Z_0d8534df"
            / "metadata.json"
        ),
    },
    {
        "name": "message_flooding_v1",
        "dataset": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "message_flooding_v1_20261108T142620Z_71820c58"
            / "mqtt_integrated_dataset_v2.csv"
        ),
        "metadata": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "experiments"
            / "message_flooding_v1_20261108T142620Z_71820c58"
            / "metadata.json"
        ),
    },
]

REQUIRED_COLUMNS = [
    "timestamp_epoch_packet",
    "frame_number",
    "frame_length_bytes",
    "tcp_stream_packet",
    "traffic_direction",
    "inter_arrival_time",
    "global_inter_arrival_time",
    "device_id",
    "temperature",
    "humidity",
    "battery",
    "traffic_label",
    "experiment_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate validated v2 MQTT experiments into time windows."
        )
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[1, 10],
        help="Window sizes in seconds. Default: 1 10",
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "normal", "flooding"],
        default="all",
        help="Choose which configured experiment to process. Default: all",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    for key in ["experiment_id", "scenario", "started_at_utc", "ended_at_utc"]:
        if key not in metadata:
            raise ValueError(f"Metadata is missing required key: {key}")

    return metadata


def to_epoch(value: str) -> float:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(
            f"Timestamp must include timezone information: {value}"
        )
    return float(timestamp.timestamp())


def load_publisher_data(path: Path, experiment_id: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Integrated v2 dataset not found: {path}")

    data = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required v2 columns: {missing}")

    if len(data.columns) != 21:
        raise ValueError(
            f"Expected the 21-column integrated v2 schema; found {len(data.columns)}."
        )

    numeric_columns = [
        "timestamp_epoch_packet",
        "frame_number",
        "frame_length_bytes",
        "tcp_stream_packet",
        "inter_arrival_time",
        "global_inter_arrival_time",
        "temperature",
        "humidity",
        "battery",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["timestamp_epoch_packet"].isna().any():
        raise ValueError("Packet timestamps contain missing or invalid values.")

    if not data["experiment_id"].eq(experiment_id).all():
        raise ValueError(
            "Dataset experiment IDs do not match the metadata experiment ID."
        )

    publisher = data[
        data["traffic_direction"].eq("publisher_to_broker")
    ].copy()

    if publisher.empty:
        raise ValueError("No publisher-to-broker records found.")

    if publisher["traffic_label"].isna().any():
        raise ValueError("Publisher records contain missing traffic labels.")

    for column in ["inter_arrival_time", "global_inter_arrival_time"]:
        invalid = publisher[column].dropna()
        if invalid.isna().any() or (invalid <= 0).any():
            raise ValueError(f"Invalid values detected in {column}.")

    return publisher.sort_values(
        ["timestamp_epoch_packet", "frame_number"]
    ).reset_index(drop=True)


def determine_phase(
    scenario: str,
    start: float,
    end: float,
    experiment_start: float,
    experiment_end: float,
    attack_start: float | None,
    attack_end: float | None,
) -> str:
    if scenario == "normal_baseline":
        return "normal"

    if scenario != "message_flooding":
        return "unknown"

    if attack_start is None or attack_end is None:
        raise ValueError(
            "Flooding metadata does not contain attack boundaries."
        )

    eps = 1e-9
    if start >= experiment_start - eps and end <= attack_start + eps:
        return "pre_attack"
    if start >= attack_start - eps and end <= attack_end + eps:
        return "attack"
    if start >= attack_end - eps and end <= experiment_end + eps:
        return "post_attack"
    return "transition"


def aggregate_windows(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    window_seconds: int,
) -> pd.DataFrame:
    if window_seconds <= 0:
        raise ValueError("Window sizes must be positive integers.")

    experiment_start = to_epoch(metadata["started_at_utc"])
    experiment_end = to_epoch(metadata["ended_at_utc"])
    scenario = metadata["scenario"]

    if experiment_end <= experiment_start:
        raise ValueError(
            "Experiment end timestamp must be after experiment start timestamp."
        )

    attack_start = None
    attack_end = None

    if scenario == "message_flooding":
        phase_timing = metadata.get("phase_timing", {})
        if not all(
            key in phase_timing
            for key in ["attack_started_at_utc", "attack_ended_at_utc"]
        ):
            raise ValueError(
                "Flooding metadata is missing attack boundary timestamps."
            )

        attack_start = to_epoch(phase_timing["attack_started_at_utc"])
        attack_end = to_epoch(phase_timing["attack_ended_at_utc"])

        if not (
            experiment_start < attack_start < attack_end < experiment_end
        ):
            raise ValueError(
                "Flooding phase timestamps are not chronological."
            )

    elapsed = data["timestamp_epoch_packet"] - experiment_start

    if (elapsed < -1).any():
        raise ValueError("Some packet timestamps precede experiment start.")
    if (data["timestamp_epoch_packet"] > experiment_end + 1).any():
        raise ValueError("Some packet timestamps exceed experiment end.")

    working = data.copy()
    working["window_index"] = np.floor(
        np.maximum(elapsed, 0) / window_seconds
    ).astype(int)

    duration = experiment_end - experiment_start
    number_of_windows = math.ceil(duration / window_seconds)
    rows: list[dict[str, Any]] = []

    for index in range(number_of_windows):
        start = experiment_start + index * window_seconds
        end = min(start + window_seconds, experiment_end)
        actual_duration = end - start
        window = working[working["window_index"] == index]
        count = len(window)

        phase = determine_phase(
            scenario,
            start,
            end,
            experiment_start,
            experiment_end,
            attack_start,
            attack_end,
        )

        row: dict[str, Any] = {
            "window_index": index,
            "window_start_utc": pd.to_datetime(
                start, unit="s", utc=True
            ).isoformat(),
            "window_end_utc": pd.to_datetime(
                end, unit="s", utc=True
            ).isoformat(),
            "window_duration_seconds": actual_duration,
            "window_size_seconds": window_seconds,
            "window_coverage_status": (
                "complete"
                if math.isclose(actual_duration, window_seconds)
                else "partial"
            ),
            "phase_label": phase,
            "message_count": int(count),
            "message_rate": count / actual_duration,
        }

        if count:
            labels = window["traffic_label"].astype(str)
            stream_intervals = window["inter_arrival_time"].dropna()
            global_intervals = window["global_inter_arrival_time"].dropna()
            lengths = window["frame_length_bytes"]
            telemetry = window[["temperature", "humidity", "battery"]]

            row.update(
                {
                    "normal_message_count": int(labels.eq("normal").sum()),
                    "flood_message_count": int(
                        labels.eq("message_flooding").sum()
                    ),
                    "unique_device_count": int(window["device_id"].nunique()),

                    # Stream-level intervals are measured within each TCP stream.
                    "stream_inter_arrival_count": int(len(stream_intervals)),
                    "mean_inter_arrival": stream_intervals.mean(),
                    "std_inter_arrival": stream_intervals.std(),
                    "min_inter_arrival": stream_intervals.min(),
                    "max_inter_arrival": stream_intervals.max(),

                    # Global intervals are measured across publisher packets,
                    # irrespective of which publisher TCP stream sent them.
                    "global_inter_arrival_count": int(len(global_intervals)),
                    "mean_global_inter_arrival": global_intervals.mean(),
                    "std_global_inter_arrival": global_intervals.std(),
                    "min_global_inter_arrival": global_intervals.min(),
                    "max_global_inter_arrival": global_intervals.max(),

                    "mean_frame_length": lengths.mean(),
                    "std_frame_length": lengths.std(),
                    "min_frame_length": lengths.min(),
                    "max_frame_length": lengths.max(),
                    "mean_temperature": telemetry["temperature"].mean(),
                    "mean_humidity": telemetry["humidity"].mean(),
                    "mean_battery": telemetry["battery"].mean(),
                    "observed_device_ids": ";".join(
                        sorted(
                            window["device_id"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    ),
                    "ground_truth_labels": ";".join(
                        sorted(labels.unique())
                    ),
                }
            )
        else:
            row.update(
                {
                    "normal_message_count": 0,
                    "flood_message_count": 0,
                    "unique_device_count": 0,
                    "stream_inter_arrival_count": 0,
                    "mean_inter_arrival": np.nan,
                    "std_inter_arrival": np.nan,
                    "min_inter_arrival": np.nan,
                    "max_inter_arrival": np.nan,
                    "global_inter_arrival_count": 0,
                    "mean_global_inter_arrival": np.nan,
                    "std_global_inter_arrival": np.nan,
                    "min_global_inter_arrival": np.nan,
                    "max_global_inter_arrival": np.nan,
                    "mean_frame_length": np.nan,
                    "std_frame_length": np.nan,
                    "min_frame_length": np.nan,
                    "max_frame_length": np.nan,
                    "mean_temperature": np.nan,
                    "mean_humidity": np.nan,
                    "mean_battery": np.nan,
                    "observed_device_ids": "",
                    "ground_truth_labels": "",
                }
            )

        rows.append(row)

    result = pd.DataFrame(rows)
    result.insert(0, "experiment_id", metadata["experiment_id"])
    result.insert(1, "scenario", scenario)
    return result


def main() -> None:
    args = parse_args()

    if any(size <= 0 for size in args.window_sizes):
        raise ValueError("All window sizes must be positive.")

    selected = DEFAULT_EXPERIMENTS
    if args.experiment == "normal":
        selected = [DEFAULT_EXPERIMENTS[0]]
    elif args.experiment == "flooding":
        selected = [DEFAULT_EXPERIMENTS[1]]

    print("\n=== UNIFIED MQTT EXPERIMENT WINDOW AGGREGATION V2 ===")

    for experiment in selected:
        metadata = load_metadata(experiment["metadata"])
        data = load_publisher_data(
            experiment["dataset"],
            metadata["experiment_id"],
        )

        print("\n" + "=" * 72)
        print(f"Experiment: {metadata['experiment_id']}")
        print(f"Scenario: {metadata['scenario']}")
        print(f"Integrated v2 dataset: {experiment['dataset']}")
        print(f"Publisher-to-broker records: {len(data)}")
        print(
            "Observed traffic labels:\n"
            f"{data['traffic_label'].value_counts().to_string()}"
        )

        output_dir = experiment["dataset"].parent

        for window_seconds in args.window_sizes:
            result = aggregate_windows(
                data, metadata, window_seconds
            )

            output_path = (
                output_dir
                / f"mqtt_window_features_{window_seconds}s_v2.csv"
            )
            result.to_csv(output_path, index=False)

            print(f"\n{window_seconds}-second windows:")
            print(f"Total windows: {len(result)}")
            print(
                "Populated windows: "
                f"{int((result['message_count'] > 0).sum())}"
            )
            print(
                "Empty windows: "
                f"{int((result['message_count'] == 0).sum())}"
            )
            print(
                "Phase counts:\n"
                f"{result['phase_label'].value_counts().to_string()}"
            )
            print(
                "Mean global inter-arrival values available: "
                f"{int(result['global_inter_arrival_count'].sum())}"
            )
            print(f"Saved: {output_path}")

    print("\nAll selected experiment aggregations completed.")
    print(
        "\nResearch caution: phase labels, ground-truth labels, device IDs, "
        "and label-derived counts are for analysis/ground truth. Do not "
        "automatically include them as model predictors. Window-level "
        "features must be selected separately for each ML experiment."
    )


if __name__ == "__main__":
    main()
