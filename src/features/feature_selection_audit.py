"""
IoT Sentinel AI — Feature Selection Audit V1

Audits the three experimental feature-set CSV files and the separate
metadata/target reference file. Does not modify input datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data/processed/feature_selection_v1"
OUTPUT_DIR = PROJECT_ROOT / "reports/feature_selection_audit_v1"

FEATURE_SET_FILES = {
    "set_A_network_timing_volume": INPUT_DIR / "set_A_network_timing_volume.csv",
    "set_B_network_plus_packet": INPUT_DIR / "set_B_network_plus_packet.csv",
    "set_C_extended_hybrid": INPUT_DIR / "set_C_extended_hybrid.csv",
}

REFERENCE_FILE = INPUT_DIR / "window_metadata_and_targets.csv"
TARGET_COLUMN = "target"
ALLOWED_TARGETS = {"normal", "message_flooding"}

FORBIDDEN_PREDICTOR_COLUMNS = {
    "target",
    "ground_truth_labels",
    "normal_message_count",
    "flood_message_count",
    "experiment_id",
    "scenario",
    "phase_label",
    "observed_device_ids",
    "window_index",
    "window_start_utc",
    "window_end_utc",
    "window_duration_seconds",
    "window_size_seconds",
    "window_coverage_status",
    "source_dataset",
}


def audit_feature_file(name: str, path: Path) -> tuple[dict, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature-set file not found: {path}")

    frame = pd.read_csv(path)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{name}: missing target column '{TARGET_COLUMN}'.")

    predictors = [column for column in frame.columns if column != TARGET_COLUMN]
    forbidden = sorted(set(predictors) & FORBIDDEN_PREDICTOR_COLUMNS)

    numeric_predictors = frame[predictors].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_predictors = sorted(set(predictors) - set(numeric_predictors))

    missing_by_column = {
        column: int(frame[column].isna().sum()) for column in frame.columns
    }
    infinite_by_column = {}
    for column in numeric_predictors:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        infinite_by_column[column] = int(np.isinf(values).sum())

    target_values = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    unexpected_targets = sorted(set(target_values) - ALLOWED_TARGETS)

    report = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "predictor_count": int(len(predictors)),
        "predictors": predictors,
        "numeric_predictor_count": int(len(numeric_predictors)),
        "non_numeric_predictors": non_numeric_predictors,
        "target_column": TARGET_COLUMN,
        "target_distribution": {
            str(key): int(value)
            for key, value in target_values.value_counts().to_dict().items()
        },
        "missing_values_by_column": missing_by_column,
        "total_missing_values": int(frame.isna().sum().sum()),
        "infinite_values_by_numeric_predictor": infinite_by_column,
        "total_infinite_values": int(sum(infinite_by_column.values())),
        "duplicate_rows": int(frame.duplicated().sum()),
        "forbidden_predictor_columns_found": forbidden,
        "unexpected_target_values": unexpected_targets,
        "checks": {
            "target_present": True,
            "target_values_valid": len(unexpected_targets) == 0,
            "no_forbidden_predictors": len(forbidden) == 0,
            "no_missing_predictor_values": all(
                frame[column].isna().sum() == 0 for column in predictors
            ),
            "no_infinite_predictor_values": sum(infinite_by_column.values()) == 0,
            "no_duplicate_rows": frame.duplicated().sum() == 0,
            "all_predictors_numeric": len(non_numeric_predictors) == 0,
        },
    }

    report["checks"]["overall_structural_pass"] = all(
        report["checks"][key]
        for key in [
            "target_present",
            "target_values_valid",
            "no_forbidden_predictors",
            "all_predictors_numeric",
            "no_duplicate_rows",
        ]
    )

    return report, frame


def audit_reference(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata/target reference file not found: {path}")

    frame = pd.read_csv(path)
    required = {
        "source_dataset",
        "experiment_id",
        "scenario",
        "phase_label",
        "ground_truth_labels",
        "normal_message_count",
        "flood_message_count",
        TARGET_COLUMN,
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Reference file is missing columns: {missing_columns}")

    normal_counts = pd.to_numeric(frame["normal_message_count"], errors="coerce")
    flood_counts = pd.to_numeric(frame["flood_message_count"], errors="coerce")
    count_sum_valid = (
        normal_counts.notna().all()
        and flood_counts.notna().all()
        and (normal_counts >= 0).all()
        and (flood_counts >= 0).all()
    )

    if count_sum_valid:
        expected_targets = pd.Series(
            np.where(flood_counts > 0, "message_flooding", "normal"),
            index=frame.index,
        )
        target_mapping_valid = (
            expected_targets == frame[TARGET_COLUMN].astype(str).str.lower()
        ).all()
    else:
        target_mapping_valid = False

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target_distribution": {
            str(key): int(value)
            for key, value in frame[TARGET_COLUMN].value_counts().to_dict().items()
        },
        "phase_distribution": {
            str(key): int(value)
            for key, value in frame["phase_label"].value_counts().to_dict().items()
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in frame["source_dataset"].value_counts().to_dict().items()
        },
        "original_label_combinations": {
            str(key): int(value)
            for key, value in frame["ground_truth_labels"].value_counts().to_dict().items()
        },
        "mixed_windows": int(
            ((normal_counts > 0) & (flood_counts > 0)).sum()
        ) if count_sum_valid else None,
        "target_mapping_matches_message_counts": bool(target_mapping_valid),
        "metadata_missing_values": {
            str(key): int(value)
            for key, value in frame.isna().sum().to_dict().items()
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_report = audit_reference(REFERENCE_FILE)
    feature_reports = {}
    loaded_frames = {}

    for name, path in FEATURE_SET_FILES.items():
        report, frame = audit_feature_file(name, path)
        feature_reports[name] = report
        loaded_frames[name] = frame

    # Confirm all feature sets have identical row counts and target ordering.
    alignment_checks = {}
    first_name = next(iter(loaded_frames))
    first_frame = loaded_frames[first_name]

    for name, frame in loaded_frames.items():
        alignment_checks[name] = {
            "same_row_count_as_set_A": len(frame) == len(first_frame),
            "same_target_sequence_as_set_A": (
                frame[TARGET_COLUMN].astype(str).tolist()
                == first_frame[TARGET_COLUMN].astype(str).tolist()
            ),
        }

    all_structural_pass = all(
        report["checks"]["overall_structural_pass"]
        for report in feature_reports.values()
    )
    all_alignment_pass = all(
        result["same_row_count_as_set_A"]
        and result["same_target_sequence_as_set_A"]
        for result in alignment_checks.values()
    )

    summary = {
        "title": "IoT Sentinel AI — Feature Selection Audit V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file_audit": reference_report,
        "feature_set_audits": feature_reports,
        "feature_set_alignment": alignment_checks,
        "overall": {
            "all_feature_sets_structurally_valid": all_structural_pass,
            "all_feature_sets_aligned": all_alignment_pass,
            "target_mapping_matches_message_counts": reference_report[
                "target_mapping_matches_message_counts"
            ],
        },
        "methodological_notes": [
            "This audit checks file structure and data integrity; it does not establish predictive value.",
            "Missing feature values are reported, not imputed.",
            "Transition windows are retained in the current feature sets; inspect phase distribution before deciding whether to include them in a particular experiment.",
            "The target is derived from flood_message_count > 0; message-count ground truth columns are kept only in the separate reference file.",
            "A structurally valid feature set is not necessarily free from experimental or temporal leakage.",
        ],
    }

    summary_path = OUTPUT_DIR / "feature_selection_audit_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    # Human-readable CSV summary, one row per feature set.
    rows = []
    for name, report in feature_reports.items():
        rows.append({
            "feature_set": name,
            "rows": report["rows"],
            "predictor_count": report["predictor_count"],
            "missing_values": report["total_missing_values"],
            "infinite_values": report["total_infinite_values"],
            "duplicate_rows": report["duplicate_rows"],
            "forbidden_predictors": len(report["forbidden_predictor_columns_found"]),
            "unexpected_targets": len(report["unexpected_target_values"]),
            "structural_pass": report["checks"]["overall_structural_pass"],
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "feature_selection_audit_summary_v1.csv",
        index=False,
    )

    print("\n=== IOT SENTINEL AI — FEATURE SELECTION AUDIT V1 ===")
    print("\nReference/target audit:")
    print(f"  Rows: {reference_report['rows']}")
    print(f"  Target mapping matches message counts: "
          f"{reference_report['target_mapping_matches_message_counts']}")
    print(f"  Mixed windows: {reference_report['mixed_windows']}")
    print(f"  Phase distribution: {reference_report['phase_distribution']}")

    print("\nFeature-set audit:")
    for name, report in feature_reports.items():
        print(f"\n{name}")
        print(f"  Rows: {report['rows']}")
        print(f"  Predictors: {report['predictor_count']}")
        print(f"  Missing values: {report['total_missing_values']}")
        print(f"  Infinite values: {report['total_infinite_values']}")
        print(f"  Duplicate rows: {report['duplicate_rows']}")
        print(f"  Forbidden predictors: {report['forbidden_predictor_columns_found']}")
        print(f"  Unexpected targets: {report['unexpected_target_values']}")
        print(f"  Structural checks: {report['checks']}")

    print("\nFeature-set alignment:")
    for name, result in alignment_checks.items():
        print(f"  {name}: {result}")

    print("\nOverall:")
    print(json.dumps(summary["overall"], indent=4))
    print(f"\nReports saved to: {OUTPUT_DIR}")
    print(f"JSON report: {summary_path}")
    print("\nAudit completed.")


if __name__ == "__main__":
    main()
