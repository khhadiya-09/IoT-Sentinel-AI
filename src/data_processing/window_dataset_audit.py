"""
IoT Sentinel AI — Window Dataset Audit V2

Audits the 1-second and 10-second V2 window datasets for:
- schema, row counts, missing values, duplicate rows
- numeric feature distributions and potential outliers
- phase and ground-truth label distributions
- message-count consistency where relevant columns exist
- ML leakage-risk column identification
- global inter-arrival-time statistics by phase

Run from the project root:
    python -m src.data_processing.window_dataset_audit

Optional:
    python -m src.data_processing.window_dataset_audit --processed-root data/processed
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENTS = {
    "normal": "normal_baseline_v2_20261108T130132Z_0d8534df",
    "flooding": "message_flooding_v1_20261108T142620Z_71820c58",
}

WINDOW_FILES = {
    "1s": "mqtt_window_features_1s_v2.csv",
    "10s": "mqtt_window_features_10s_v2.csv",
}

LEAKAGE_EXACT = {
    "traffic_label",
    "ground_truth_label",
    "ground_truth_labels",
    "phase_label",
    "experiment_id",
    "run_id",
    "device_id",
    "device_type",
    "observed_device_ids",
    "normal_message_count",
    "flood_message_count",
}


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-serializable Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def identify_leakage_risks(columns: list[str]) -> dict[str, list[str]]:
    exact = []
    derived = []
    for column in columns:
        name = column.lower().strip()
        if name in LEAKAGE_EXACT:
            exact.append(column)
        elif any(token in name for token in (
            "label", "ground_truth", "device_id", "experiment",
            "phase", "normal_message_count", "flood_message_count"
        )):
            derived.append(column)
    return {
        "exclude_from_default_predictors": sorted(set(exact)),
        "review_before_predictor_use": sorted(set(derived) - set(exact)),
    }


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if valid.empty:
        return {
            "valid_count": 0,
            "missing_or_non_finite_count": int(len(series)),
        }

    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((valid < lower) | (valid > upper)).sum())

    return {
        "valid_count": int(valid.size),
        "missing_or_non_finite_count": int(len(series) - valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std()) if valid.size > 1 else None,
        "min": float(valid.min()),
        "q1": q1,
        "median": float(valid.median()),
        "q3": q3,
        "max": float(valid.max()),
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_lower_bound": lower,
        "iqr_outlier_upper_bound": upper,
    }


def value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("<MISSING>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def audit_dataset(path: Path, experiment_key: str, window_size: str) -> dict[str, Any]:
    print("\n" + "=" * 76)
    print(f"AUDITING: {experiment_key.upper()} | {window_size} WINDOWS")
    print("=" * 76)
    print(f"File: {path}")

    if not path.exists():
        return {
            "status": "MISSING_FILE",
            "experiment": experiment_key,
            "window_size": window_size,
            "file": str(path),
        }

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    missing = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if int(count) > 0
    }
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = {
        column: numeric_summary(df[column])
        for column in numeric_columns
    }

    categorical_columns = [
        column for column in df.columns
        if column in {
            "phase_label", "traffic_label", "ground_truth_labels",
            "experiment_id", "device_id", "device_type"
        }
    ]
    categorical_stats = {
        column: value_counts(df[column])
        for column in categorical_columns
    }

    phase_timing = {}
    if "phase_label" in df.columns:
        phase_timing["phase_window_counts"] = value_counts(df["phase_label"])

    timing_by_phase = {}
    if "phase_label" in df.columns:
        for timing_column in (
            "global_inter_arrival_mean",
            "mean_global_inter_arrival_time",
            "global_inter_arrival_time_mean",
            "mean_global_inter_arrival",
        ):
            if timing_column in df.columns:
                timing_by_phase[timing_column] = (
                    df.groupby("phase_label", dropna=False)[timing_column]
                    .agg(["count", "mean", "median", "min", "max"])
                    .reset_index()
                    .to_dict(orient="records")
                )

    # Identify likely global inter-arrival columns even if the exact name differs.
    global_timing_columns = [
        column for column in df.columns
        if "global" in column.lower() and "inter_arrival" in column.lower()
    ]
    if "phase_label" in df.columns:
        for column in global_timing_columns:
            if column not in timing_by_phase:
                grouped = df.groupby("phase_label", dropna=False)[column].agg(
                    ["count", "mean", "median", "min", "max"]
                ).reset_index()
                timing_by_phase[column] = grouped.to_dict(orient="records")

    consistency_checks = {}
    if {"message_count", "normal_message_count", "flood_message_count"}.issubset(df.columns):
        total = (
            pd.to_numeric(df["normal_message_count"], errors="coerce").fillna(0)
            + pd.to_numeric(df["flood_message_count"], errors="coerce").fillna(0)
        )
        observed = pd.to_numeric(df["message_count"], errors="coerce")
        mismatches = int((total != observed).sum())
        consistency_checks["message_count_equals_normal_plus_flood"] = {
            "mismatch_rows": mismatches,
            "status": "PASS" if mismatches == 0 else "REVIEW",
        }

    if "message_count" in df.columns:
        counts = pd.to_numeric(df["message_count"], errors="coerce")
        consistency_checks["message_count_non_negative"] = {
            "invalid_rows": int((counts.dropna() < 0).sum()),
            "status": "PASS" if not (counts.dropna() < 0).any() else "FAIL",
        }

    leakage = identify_leakage_risks(df.columns.tolist())

    # Structural checks; empty windows are expected and are not treated as errors.
    checks = {
        "file_exists": True,
        "dataset_non_empty": len(df) > 0,
        "duplicate_rows_absent": duplicate_rows == 0,
        "no_infinite_numeric_values": all(
            not np.isinf(pd.to_numeric(df[column], errors="coerce").dropna()).any()
            for column in numeric_columns
        ),
    }

    report = {
        "status": "AUDITED",
        "experiment": experiment_key,
        "window_size": window_size,
        "file": str(path),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "duplicate_rows": duplicate_rows,
        "missing_values_by_column": missing,
        "numeric_feature_summary": numeric_stats,
        "categorical_distributions": categorical_stats,
        "phase_timing": phase_timing,
        "global_inter_arrival_columns": global_timing_columns,
        "global_inter_arrival_by_phase": timing_by_phase,
        "consistency_checks": consistency_checks,
        "ml_predictor_column_guidance": leakage,
        "structural_checks": checks,
    }

    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Columns with missing values: {len(missing)}")
    print(f"Numeric columns: {len(numeric_columns)}")
    if "phase_label" in df.columns:
        print("Phase distribution:")
        for label, count in phase_timing["phase_window_counts"].items():
            print(f"  {label}: {count}")
    if global_timing_columns:
        print(f"Global timing columns found: {', '.join(global_timing_columns)}")
    else:
        print("WARNING: No global inter-arrival feature column was detected.")

    print("ML leakage-risk columns:")
    print("  Exclude by default:", leakage["exclude_from_default_predictors"] or "None detected")
    print("  Review:", leakage["review_before_predictor_use"] or "None detected")

    print("Structural checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit IoT Sentinel AI V2 window datasets.")
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Root directory containing processed experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/dataset_audits",
        help="Directory for JSON audit reports.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    processed_root = (project_root / args.processed_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== IOT SENTINEL AI — WINDOW DATASET AUDIT V2 ===")
    print(f"Processed root: {processed_root}")

    all_reports = []
    for experiment_key, run_folder in EXPERIMENTS.items():
        for window_size, filename in WINDOW_FILES.items():
            dataset_path = processed_root / run_folder / filename
            report = audit_dataset(dataset_path, experiment_key, window_size)
            all_reports.append(report)

            if report["status"] == "AUDITED":
                report_name = f"{experiment_key}_{window_size}_window_audit_v2.json"
                report_path = output_dir / report_name
                report_path.write_text(
                    json.dumps(report, indent=4, default=json_safe),
                    encoding="utf-8",
                )
                print(f"Saved report: {report_path}")

    summary = {
        "project": "IoT Sentinel AI",
        "audit_version": "2.0",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": all_reports,
        "datasets_found": sum(r["status"] == "AUDITED" for r in all_reports),
        "datasets_missing": sum(r["status"] == "MISSING_FILE" for r in all_reports),
    }
    summary_path = output_dir / "window_dataset_audit_summary_v2.json"
    summary_path.write_text(
        json.dumps(summary, indent=4, default=json_safe),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("WINDOW DATASET AUDIT SUMMARY")
    print("=" * 76)
    print(f"Datasets audited: {summary['datasets_found']}/4")
    print(f"Datasets missing: {summary['datasets_missing']}")
    print(f"Summary saved: {summary_path}")

    return 0 if summary["datasets_found"] == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
