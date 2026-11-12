"""
IoT Sentinel AI — Feature Selection V1.1

Creates three experimental feature sets from the existing 1-second windows.

Binary target policy:
- normal_message_count > 0 and flood_message_count == 0 -> normal
- flood_message_count > 0 (including mixed windows) -> message_flooding

The two message-count columns are used ONLY to construct the target.
They are never included as model predictors.

Original input datasets are read-only and are not modified.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILES = {
    "normal_baseline": (
        PROJECT_ROOT
        / "data/processed/normal_baseline_v2_20261108T130132Z_0d8534df/"
        / "mqtt_window_features_1s_v2.csv"
    ),
    "message_flooding": (
        PROJECT_ROOT
        / "data/processed/message_flooding_v1_20261108T142620Z_71820c58/"
        / "mqtt_window_features_1s_v2.csv"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "data/processed/feature_selection_v1"

METADATA_COLUMNS = [
    "source_dataset",
    "experiment_id",
    "scenario",
    "window_index",
    "window_start_utc",
    "window_end_utc",
    "window_duration_seconds",
    "window_size_seconds",
    "window_coverage_status",
    "phase_label",
]

TARGET_COLUMN = "target"

FEATURE_SETS = {
    "set_A_network_timing_volume": [
        "message_rate",
        "mean_global_inter_arrival",
        "std_global_inter_arrival",
        "min_global_inter_arrival",
        "max_global_inter_arrival",
    ],
    "set_B_network_plus_packet": [
        "message_rate",
        "mean_global_inter_arrival",
        "std_global_inter_arrival",
        "min_global_inter_arrival",
        "max_global_inter_arrival",
        "mean_frame_length",
        "std_frame_length",
    ],
    "set_C_extended_hybrid": [
        "message_rate",
        "mean_global_inter_arrival",
        "std_global_inter_arrival",
        "min_global_inter_arrival",
        "max_global_inter_arrival",
        "mean_frame_length",
        "std_frame_length",
        "mean_inter_arrival",
        "std_inter_arrival",
        "min_inter_arrival",
        "max_inter_arrival",
        "mean_temperature",
        "mean_humidity",
        "mean_battery",
        "unique_device_count",
    ],
}

EXCLUDED_FROM_PREDICTORS = {
    "ground_truth_labels": "Original label field; retained only for auditing.",
    "normal_message_count": "Used only to construct the target; never a predictor.",
    "flood_message_count": "Used only to construct the target; never a predictor.",
    "experiment_id": "Experiment identity; metadata and potential leakage.",
    "scenario": "Scenario identity; metadata and potential leakage.",
    "phase_label": "Experiment phase; metadata, not a traffic-derived predictor.",
    "observed_device_ids": "Device identity list; potential identity leakage.",
    "message_count": "Redundant with message_rate for 1-second windows.",
    "global_inter_arrival_count": "Strongly redundant with traffic/interval counts.",
    "stream_inter_arrival_count": "Strongly redundant with global interval count in current data.",
    "window_index": "Ordering metadata, not a traffic predictor.",
    "window_start_utc": "Timestamp metadata; may encode run/phase information.",
    "window_end_utc": "Timestamp metadata; may encode run/phase information.",
    "window_duration_seconds": "Window metadata.",
    "window_size_seconds": "Window configuration metadata.",
    "window_coverage_status": "Window quality metadata.",
}


def load_datasets() -> pd.DataFrame:
    frames = []

    for source_name, path in INPUT_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input dataset not found: {path}")

        frame = pd.read_csv(path)
        frame["source_dataset"] = source_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    required = {
        "ground_truth_labels",
        "normal_message_count",
        "flood_message_count",
        "message_count",
        "window_coverage_status",
        "phase_label",
        *METADATA_COLUMNS,
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    return combined


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_rows = len(raw)

    # Retain only windows with observed traffic. This excludes empty windows.
    message_count = pd.to_numeric(raw["message_count"], errors="coerce")
    populated = raw.loc[message_count > 0].copy()

    if populated.empty:
        raise ValueError("No populated windows remain after filtering.")

    normal_counts = pd.to_numeric(
        populated["normal_message_count"], errors="coerce"
    )
    flood_counts = pd.to_numeric(
        populated["flood_message_count"], errors="coerce"
    )

    if normal_counts.isna().any() or flood_counts.isna().any():
        raise ValueError("Ground-truth message counts contain missing/non-numeric values.")

    if (normal_counts < 0).any() or (flood_counts < 0).any():
        raise ValueError("Ground-truth message counts cannot be negative.")

    if ((normal_counts + flood_counts) != message_count.loc[populated.index]).any():
        raise ValueError(
            "Ground-truth normal/flood counts do not sum to message_count "
            "for every populated window. Inspect the source data before proceeding."
        )

    # Binary window-level target: any flooding message makes the window positive.
    populated[TARGET_COLUMN] = [
        "message_flooding" if flood > 0 else "normal"
        for normal, flood in zip(normal_counts, flood_counts)
    ]

    # Validate the original semicolon-separated labels against the count-derived
    # binary target, allowing mixed windows to contain both labels.
    original_labels = populated["ground_truth_labels"].fillna("").astype(str)
    for index, label_text in original_labels.items():
        labels = {
            item.strip().lower()
            for item in label_text.split(";")
            if item.strip()
        }
        expected = set()
        row = populated.loc[index]
        if float(row["normal_message_count"]) > 0:
            expected.add("normal")
        if float(row["flood_message_count"]) > 0:
            expected.add("message_flooding")
        if labels != expected:
            raise ValueError(
                f"ground_truth_labels does not match message counts at row {index}: "
                f"labels={sorted(labels)}, expected={sorted(expected)}"
            )

    info = {
        "original_rows": int(original_rows),
        "populated_rows": int(len(populated)),
        "excluded_empty_rows": int(original_rows - len(populated)),
        "binary_target_policy": (
            "A window is message_flooding if flood_message_count > 0, "
            "including mixed normal/flood windows; otherwise it is normal."
        ),
        "binary_target_distribution": {
            str(k): int(v)
            for k, v in populated[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "original_ground_truth_label_combinations": {
            str(k): int(v)
            for k, v in populated["ground_truth_labels"].value_counts().to_dict().items()
        },
        "phase_distribution": {
            str(k): int(v)
            for k, v in populated["phase_label"].value_counts().to_dict().items()
        },
        "source_distribution": {
            str(k): int(v)
            for k, v in populated["source_dataset"].value_counts().to_dict().items()
        },
        "mixed_windows": int(
            ((normal_counts > 0) & (flood_counts > 0)).sum()
        ),
    }

    return populated, info


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_datasets()
    prepared, data_summary = prepare_dataset(raw)

    reference_columns = [
        column for column in METADATA_COLUMNS if column in prepared.columns
    ] + [
        "ground_truth_labels",
        "normal_message_count",
        "flood_message_count",
        TARGET_COLUMN,
    ]
    reference = prepared[reference_columns].copy()
    reference.to_csv(OUTPUT_DIR / "window_metadata_and_targets.csv", index=False)

    feature_set_summaries = {}

    for set_name, predictors in FEATURE_SETS.items():
        missing_features = sorted(set(predictors) - set(prepared.columns))
        if missing_features:
            raise ValueError(
                f"{set_name} requires missing columns: {missing_features}"
            )

        output = prepared[predictors + [TARGET_COLUMN]].copy()
        output_path = OUTPUT_DIR / f"{set_name}.csv"
        output.to_csv(output_path, index=False)

        feature_set_summaries[set_name] = {
            "file": str(output_path.relative_to(PROJECT_ROOT)),
            "predictor_count": len(predictors),
            "predictors": predictors,
            "target_column_not_a_predictor": TARGET_COLUMN,
            "rows": int(len(output)),
            "missing_values_by_predictor": {
                col: int(output[col].isna().sum()) for col in predictors
            },
        }

    report = {
        "title": "IoT Sentinel AI — Feature Selection V1.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            key: str(path.relative_to(PROJECT_ROOT))
            for key, path in INPUT_FILES.items()
        },
        "data_summary": data_summary,
        "feature_sets": feature_set_summaries,
        "excluded_from_predictors": EXCLUDED_FROM_PREDICTORS,
        "methodological_notes": [
            "Original raw and processed window datasets were not modified.",
            "The target was derived from normal_message_count and flood_message_count.",
            "A mixed window containing any flooding messages is labelled message_flooding.",
            "The message-count columns used to derive the target are excluded from every predictor set.",
            "ground_truth_labels and metadata are stored separately from predictor matrices.",
            "Empty windows are excluded; populated pre-attack, attack, post-attack and normal windows are retained.",
            "Pre-attack and post-attack windows with normal traffic are labelled normal under the stated binary policy.",
            "Normal and flooding observations come from separate experimental runs. Random window-level splitting can create misleading evaluation; design run-aware validation before model training.",
            "Only one configured flooding experiment is currently available; this does not establish generalization to unseen attack types.",
            "Feature sets are experimental hypotheses, not claims of optimality.",
            "Frame-length features may be confounded by the flooding client's additional sequence_number payload field.",
            "Telemetry and unique_device_count may encode device composition or simulator behavior; evaluate incremental value separately.",
            "Do not impute missing values blindly; fit any imputation procedure on training data only.",
        ],
    }

    report_path = OUTPUT_DIR / "feature_selection_report_v1.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n=== IOT SENTINEL AI — FEATURE SELECTION V1.1 ===")
    print(f"Combined original rows: {data_summary['original_rows']}")
    print(f"Populated rows retained: {data_summary['populated_rows']}")
    print(f"Empty rows excluded: {data_summary['excluded_empty_rows']}")
    print(f"Mixed windows: {data_summary['mixed_windows']}")
    print("Binary target distribution:", data_summary["binary_target_distribution"])
    print(
        "Original label combinations:",
        data_summary["original_ground_truth_label_combinations"],
    )
    print("Phase distribution:", data_summary["phase_distribution"])
    print("\nGenerated feature sets:")

    for set_name, details in feature_set_summaries.items():
        print(f"\n{set_name}")
        print(f"  Rows: {details['rows']}")
        print(f"  Predictor count: {details['predictor_count']}")
        print("  Predictors:", ", ".join(details["predictors"]))

    print("\nMetadata and targets:")
    print(OUTPUT_DIR / "window_metadata_and_targets.csv")
    print("\nJSON report:")
    print(report_path)
    print("\nOriginal datasets were not modified.")
    print("Feature selection V1.1 completed.")


if __name__ == "__main__":
    main()
