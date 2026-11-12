"""
Read-only audit for the nine IoT Sentinel AI V2 PCAP feature CSV files.

Run from the project root:
    python scripts/combined_feature_dataset_audit_v2.py

Reads the nine explicitly listed V2 feature files and writes:
    reports/combined_feature_dataset_audit_v2.csv
    reports/combined_feature_dataset_audit_v2.json
    reports/combined_feature_dataset_audit_v2.md
    data/processed/combined_pcap_features_v2.csv

The source feature CSVs are never modified. Existing outputs are not overwritten.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Explicit manifest prevents older V1/V2 exploratory files from being included.
MANIFEST = [
    {
        "file": "normal_original_v2.csv",
        "run_id": "normal_baseline_v2_20261108T130132Z_0d8534df",
        "scenario": "normal_baseline",
        "rate": None,
        "expected_rows": 30,
    },
    {
        "file": "normal_seed101_v2.csv",
        "run_id": "normal_baseline_v2_20261111T174556Z_d6e153b1",
        "scenario": "normal_baseline",
        "rate": None,
        "expected_rows": 30,
    },
    {
        "file": "normal_seed202_v2.csv",
        "run_id": "normal_baseline_v2_20261111T175428Z_b01e38fe",
        "scenario": "normal_baseline",
        "rate": None,
        "expected_rows": 30,
    },
    {
        "file": "flood_2rps_run01_v2.csv",
        "run_id": "flood_v2_2rps_run01_20261111T161433Z_13332d0b",
        "scenario": "message_flooding",
        "rate": 2.0,
        "expected_rows": 6,
    },
    {
        "file": "flood_2rps_run02_v2.csv",
        "run_id": "flood_v2_2rps_run02_20261111T180926Z_39e51e4c",
        "scenario": "message_flooding",
        "rate": 2.0,
        "expected_rows": 6,
    },
    {
        "file": "flood_5rps_run01_v2.csv",
        "run_id": "flood_v2_5rps_run01_20261111T164008Z_05155788",
        "scenario": "message_flooding",
        "rate": 5.0,
        "expected_rows": 6,
    },
    {
        "file": "flood_5rps_run02_v2.csv",
        "run_id": "flood_v2_5rps_run02_20261111T181124Z_80434161",
        "scenario": "message_flooding",
        "rate": 5.0,
        "expected_rows": 6,
    },
    {
        "file": "test_flood_v2.csv",
        "run_id": "flood_v2_10rps_run01_20261111T165033Z_8d2db887",
        "scenario": "message_flooding",
        "rate": 10.0,
        "expected_rows": 6,
    },
    {
        "file": "flood_10rps_run02_v2.csv",
        "run_id": "flood_v2_10rps_run02_20261111T181318Z_de9827ee",
        "scenario": "message_flooding",
        "rate": 10.0,
        "expected_rows": 6,
    },
]

EXPECTED_COLUMNS = [
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
    "packet_count",
    "mqtt_publish_count",
    "total_bytes",
    "mean_packet_size_bytes",
    "std_packet_size_bytes",
    "min_packet_size_bytes",
    "max_packet_size_bytes",
    "packet_rate_per_second",
    "publish_rate_per_second",
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
    "unique_tcp_streams",
    "telemetry_message_count",
    "mean_temperature",
    "std_temperature",
    "mean_humidity",
    "std_humidity",
    "mean_battery",
    "std_battery",
    "unique_devices",
    "normal_message_count",
    "attack_message_count",
    "target",
    "phase_label",
]

# Only these columns are candidate predictors. IDs, timestamps, labels, phases,
# and message-count ground-truth fields are deliberately excluded.
PREDICTOR_COLUMNS = [
    "packet_count",
    "mqtt_publish_count",
    "total_bytes",
    "mean_packet_size_bytes",
    "std_packet_size_bytes",
    "min_packet_size_bytes",
    "max_packet_size_bytes",
    "packet_rate_per_second",
    "publish_rate_per_second",
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
    "unique_tcp_streams",
    "telemetry_message_count",
    "mean_temperature",
    "std_temperature",
    "mean_humidity",
    "std_humidity",
    "mean_battery",
    "std_battery",
    "unique_devices",
]

TARGETS = {"normal", "mixed"}
PHASES = {
    "normal_baseline",
    "pre_attack",
    "attack",
    "mixed_boundary",
    "post_attack",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    return columns, rows


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def parse_number(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_signature(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple((row.get(column) or "").strip() for column in columns)


def predictor_signature(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(column) or "").strip() for column in PREDICTOR_COLUMNS)


def safe_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) *
                            sum(value * value for value in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def audit_file(entry: dict[str, Any]) -> dict[str, Any]:
    path = PROCESSED / entry["file"]
    result: dict[str, Any] = {
        "file": entry["file"],
        "expected_run_id": entry["run_id"],
        "scenario_expected": entry["scenario"],
        "configured_rate_msg_per_second": entry["rate"],
        "expected_rows": entry["expected_rows"],
        "exists": path.is_file(),
        "passed": False,
        "rows": 0,
        "columns": 0,
        "missing_cells": 0,
        "infinite_cells": 0,
        "exact_duplicate_rows": 0,
        "duplicate_predictor_rows": 0,
        "target_distribution": {},
        "phase_distribution": {},
        "errors": [],
        "warnings": [],
    }

    if not path.is_file():
        result["errors"].append("Feature CSV not found")
        return result

    try:
        columns, rows = read_csv(path)
        result["rows"] = len(rows)
        result["columns"] = len(columns)

        if columns != EXPECTED_COLUMNS:
            missing_columns = [c for c in EXPECTED_COLUMNS if c not in columns]
            extra_columns = [c for c in columns if c not in EXPECTED_COLUMNS]
            result["errors"].append(
                f"Column schema/order differs; missing={missing_columns}; extra={extra_columns}"
            )

        if len(rows) != entry["expected_rows"]:
            result["errors"].append(
                f"Expected {entry['expected_rows']} rows, found {len(rows)}"
            )

        missing_cells = sum(
            1 for row in rows for column in columns if is_blank(row.get(column))
        )
        result["missing_cells"] = missing_cells
        if missing_cells:
            result["errors"].append(f"Found {missing_cells} blank cells")

        infinite_cells = 0
        numeric_errors: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            for column in PREDICTOR_COLUMNS:
                if column not in columns:
                    continue
                value = row.get(column)
                if is_blank(value):
                    continue
                number = parse_number(value)
                if number is None:
                    numeric_errors.append(f"row {row_number}: {column} is not numeric")
                elif not math.isfinite(number):
                    infinite_cells += 1

        result["infinite_cells"] = infinite_cells
        if infinite_cells:
            result["errors"].append(f"Found {infinite_cells} infinite predictor values")
        if numeric_errors:
            result["errors"].extend(numeric_errors[:10])
            if len(numeric_errors) > 10:
                result["errors"].append(
                    f"{len(numeric_errors) - 10} additional numeric parsing errors omitted"
                )

        signatures = [row_signature(row, columns) for row in rows]
        result["exact_duplicate_rows"] = len(signatures) - len(set(signatures))
        predictor_signatures = [predictor_signature(row) for row in rows]
        result["duplicate_predictor_rows"] = (
            len(predictor_signatures) - len(set(predictor_signatures))
        )

        targets = Counter(row.get("target", "") for row in rows)
        phases = Counter(row.get("phase_label", "") for row in rows)
        result["target_distribution"] = dict(sorted(targets.items()))
        result["phase_distribution"] = dict(sorted(phases.items()))

        unexpected_targets = sorted(set(targets) - TARGETS)
        unexpected_phases = sorted(set(phases) - PHASES)
        if unexpected_targets:
            result["errors"].append(f"Unexpected target values: {unexpected_targets}")
        if unexpected_phases:
            result["errors"].append(f"Unexpected phase values: {unexpected_phases}")

        run_ids = {row.get("run_id", "") for row in rows}
        if run_ids != {entry["run_id"]}:
            result["errors"].append(
                f"run_id values do not match manifest: {sorted(run_ids)}"
            )

        expected_experiment_id = (
            "normal_baseline_v2"
            if entry["scenario"] == "normal_baseline"
            else entry["run_id"].split("_2026", 1)[0]
        )
        experiment_ids = {row.get("experiment_id", "") for row in rows}
        if experiment_ids != {expected_experiment_id}:
            result["errors"].append(
                f"experiment_id values differ from expected {expected_experiment_id!r}: "
                f"{sorted(experiment_ids)}"
            )

        scenarios = {row.get("scenario", "") for row in rows}
        if scenarios != {entry["scenario"]}:
            result["errors"].append(
                f"scenario values differ from expected {entry['scenario']!r}: {sorted(scenarios)}"
            )

        window_indices: list[int] = []
        for row_number, row in enumerate(rows, start=2):
            try:
                window_indices.append(int(row["window_index"]))
            except (KeyError, TypeError, ValueError):
                result["errors"].append(f"Invalid window_index at CSV row {row_number}")
        if len(window_indices) == len(rows) and len(set(window_indices)) != len(window_indices):
            result["errors"].append("Duplicate window_index values within run")
        if len(window_indices) == entry["expected_rows"]:
            if sorted(window_indices) != list(range(entry["expected_rows"])):
                result["errors"].append("window_index sequence is not the expected zero-based range")

        if entry["scenario"] == "normal_baseline":
            if targets != Counter({"normal": 30}):
                result["errors"].append("Normal baseline target distribution is not 30 normal windows")
            if phases != Counter({"normal_baseline": 30}):
                result["errors"].append("Normal baseline phase distribution is unexpected")
        else:
            if targets != Counter({"normal": 2, "mixed": 4}):
                result["errors"].append("Flooding target distribution is not 2 normal / 4 mixed")
            expected_phase_counts = Counter({
                "pre_attack": 1,
                "mixed_boundary": 2,
                "attack": 2,
                "post_attack": 1,
            })
            if phases != expected_phase_counts:
                result["errors"].append("Flooding phase distribution differs from expected window layout")

        if result["duplicate_predictor_rows"]:
            result["warnings"].append(
                "Some rows have identical predictor values; inspect before modelling."
            )

    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    result["passed"] = not result["errors"]
    return result


def main() -> int:
    output_paths = [
        PROCESSED / "combined_pcap_features_v2.csv",
        REPORTS / "combined_feature_dataset_audit_v2.csv",
        REPORTS / "combined_feature_dataset_audit_v2.json",
        REPORTS / "combined_feature_dataset_audit_v2.md",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        print("ERROR: Refusing to overwrite existing output(s):", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        print("Move or rename those outputs yourself, or choose new output names.", file=sys.stderr)
        return 2

    REPORTS.mkdir(parents=True, exist_ok=True)

    file_results = [audit_file(entry) for entry in MANIFEST]
    combined_rows: list[dict[str, str]] = []
    combined_columns: list[str] | None = None
    combined_read_errors: list[str] = []

    for entry in MANIFEST:
        path = PROCESSED / entry["file"]
        if not path.is_file():
            continue
        try:
            columns, rows = read_csv(path)
            if combined_columns is None:
                combined_columns = columns
            elif columns != combined_columns:
                combined_read_errors.append(
                    f"{entry['file']}: column schema differs from first readable file"
                )
                continue
            combined_rows.extend(rows)
        except Exception as exc:
            combined_read_errors.append(f"{entry['file']}: {type(exc).__name__}: {exc}")

    expected_total_rows = sum(entry["expected_rows"] for entry in MANIFEST)
    combined_summary: dict[str, Any] = {
        "expected_runs": len(MANIFEST),
        "readable_runs": sum(result["exists"] for result in file_results),
        "expected_rows": expected_total_rows,
        "combined_rows": len(combined_rows),
        "combined_columns": len(combined_columns or []),
        "target_distribution": dict(sorted(Counter(
            row.get("target", "") for row in combined_rows
        ).items())),
        "phase_distribution": dict(sorted(Counter(
            row.get("phase_label", "") for row in combined_rows
        ).items())),
        "scenario_distribution": dict(sorted(Counter(
            row.get("scenario", "") for row in combined_rows
        ).items())),
        "run_distribution": dict(sorted(Counter(
            row.get("run_id", "") for row in combined_rows
        ).items())),
        "exact_duplicate_rows": 0,
        "duplicate_predictor_rows": 0,
        "missing_cells": 0,
        "infinite_predictor_cells": 0,
        "predictor_statistics": {},
        "selected_feature_correlations": {},
        "errors": combined_read_errors,
        "warnings": [],
    }

    if combined_columns is not None:
        signatures = [row_signature(row, combined_columns) for row in combined_rows]
        combined_summary["exact_duplicate_rows"] = len(signatures) - len(set(signatures))
        predictor_signatures = [predictor_signature(row) for row in combined_rows]
        combined_summary["duplicate_predictor_rows"] = (
            len(predictor_signatures) - len(set(predictor_signatures))
        )
        combined_summary["missing_cells"] = sum(
            1 for row in combined_rows
            for column in combined_columns if is_blank(row.get(column))
        )

        for column in PREDICTOR_COLUMNS:
            values: list[float] = []
            for row_number, row in enumerate(combined_rows, start=2):
                number = parse_number(row.get(column))
                if number is not None:
                    if math.isfinite(number):
                        values.append(number)
                    else:
                        combined_summary["infinite_predictor_cells"] += 1
                elif not is_blank(row.get(column)):
                    combined_summary["errors"].append(
                        f"Combined predictor {column} contains a non-numeric value"
                    )
                    break
            combined_summary["predictor_statistics"][column] = safe_stats(values)

        correlation_pairs = [
            ("packet_count", "mqtt_publish_count"),
            ("packet_rate_per_second", "publish_rate_per_second"),
            ("unique_tcp_streams", "unique_devices"),
            ("packet_count", "total_bytes"),
        ]
        for first, second in correlation_pairs:
            x_values, y_values = [], []
            for row in combined_rows:
                x = parse_number(row.get(first))
                y = parse_number(row.get(second))
                if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                    x_values.append(x)
                    y_values.append(y)
            combined_summary["selected_feature_correlations"][f"{first}__{second}"] = (
                pearson_correlation(x_values, y_values)
            )

    if len(combined_rows) != expected_total_rows:
        combined_summary["errors"].append(
            f"Combined row count expected {expected_total_rows}, found {len(combined_rows)}"
        )
    if combined_summary["missing_cells"]:
        combined_summary["errors"].append(
            f"Combined dataset contains {combined_summary['missing_cells']} blank cells"
        )
    if combined_summary["infinite_predictor_cells"]:
        combined_summary["errors"].append(
            f"Combined dataset contains {combined_summary['infinite_predictor_cells']} infinite predictor values"
        )
    if combined_summary["exact_duplicate_rows"]:
        combined_summary["warnings"].append(
            "Exact duplicate rows were found in the combined dataset."
        )
    if combined_summary["duplicate_predictor_rows"]:
        combined_summary["warnings"].append(
            "Duplicate predictor vectors were found. These are not automatically errors, "
            "but they should be reviewed for train/test leakage."
        )

    all_files_passed = all(result["passed"] for result in file_results)
    combined_summary["all_files_passed"] = all_files_passed
    combined_summary["overall_passed"] = (
        all_files_passed
        and not combined_summary["errors"]
        and len(combined_rows) == expected_total_rows
    )

    # Write the combined file with exclusive creation to avoid overwriting.
    if combined_columns is not None:
        combined_path = PROCESSED / "combined_pcap_features_v2.csv"
        with combined_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=combined_columns)
            writer.writeheader()
            writer.writerows(combined_rows)

    audit_csv = REPORTS / "combined_feature_dataset_audit_v2.csv"
    audit_fields = [
        "file", "expected_run_id", "scenario_expected",
        "configured_rate_msg_per_second", "expected_rows", "exists",
        "passed", "rows", "columns", "missing_cells", "infinite_cells",
        "exact_duplicate_rows", "duplicate_predictor_rows",
        "target_distribution", "phase_distribution", "errors", "warnings",
    ]
    with audit_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for result in file_results:
            item = dict(result)
            for key in ("target_distribution", "phase_distribution", "errors", "warnings"):
                item[key] = json.dumps(item[key], ensure_ascii=False)
            writer.writerow({key: item.get(key) for key in audit_fields})

    audit_json = REPORTS / "combined_feature_dataset_audit_v2.json"
    report = {
        "audit_version": "2.0",
        "read_only_source_audit": True,
        "source_files": MANIFEST,
        "file_results": file_results,
        "combined_summary": combined_summary,
        "methodology_notes": [
            "The nine source feature CSV files are read but never modified.",
            "The combined dataset is a concatenation of the nine explicitly manifested files.",
            "target and phase_label are labels/metadata, not candidate predictors.",
            "normal_message_count and attack_message_count are ground-truth-derived fields and must not be used as predictors.",
            "IDs, timestamps, scenario, window indices, and configured attack rate are metadata, not predictors.",
            "unique_tcp_streams may encode the presence of an additional publisher and must be evaluated for scenario leakage.",
            "Duplicate predictor vectors are reported separately from exact duplicate rows.",
            "Correlation is descriptive and does not establish causation or prove feature redundancy.",
            "Mixed windows are preserved and are not silently converted into binary labels.",
        ],
    }
    with audit_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# IoT Sentinel AI — Combined V2 Feature Dataset Audit",
        "",
        f"- Overall result: **{'PASS' if combined_summary['overall_passed'] else 'REVIEW'}**",
        f"- Files passed: **{sum(r['passed'] for r in file_results)}/{len(file_results)}**",
        f"- Combined rows: **{len(combined_rows)}/{expected_total_rows} expected**",
        f"- Combined columns: **{combined_summary['combined_columns']}**",
        f"- Exact duplicate rows: **{combined_summary['exact_duplicate_rows']}**",
        f"- Duplicate predictor vectors: **{combined_summary['duplicate_predictor_rows']}**",
        f"- Blank cells: **{combined_summary['missing_cells']}**",
        f"- Infinite predictor cells: **{combined_summary['infinite_predictor_cells']}**",
        "",
        "## Per-file results",
        "",
        "| File | Rows | Columns | Result | Missing | Infinite | Exact duplicates | Predictor duplicates |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for result in file_results:
        md_lines.append(
            f"| `{result['file']}` | {result['rows']} | {result['columns']} | "
            f"{'PASS' if result['passed'] else 'REVIEW'} | {result['missing_cells']} | "
            f"{result['infinite_cells']} | {result['exact_duplicate_rows']} | "
            f"{result['duplicate_predictor_rows']} |"
        )

    md_lines += [
        "",
        "## Combined distributions",
        "",
        f"- Targets: `{json.dumps(combined_summary['target_distribution'], sort_keys=True)}`",
        f"- Phases: `{json.dumps(combined_summary['phase_distribution'], sort_keys=True)}`",
        f"- Scenarios: `{json.dumps(combined_summary['scenario_distribution'], sort_keys=True)}`",
        "",
        "## Predictor statistics",
        "",
        "| Feature | Count | Min | Max | Mean | Population SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, stats in combined_summary["predictor_statistics"].items():
        md_lines.append(
            f"| `{feature}` | {stats['count']} | {stats['min']} | {stats['max']} | "
            f"{stats['mean']} | {stats['std']} |"
        )

    md_lines += [
        "",
        "## Selected feature correlations",
        "",
        "| Feature pair | Pearson correlation |",
        "|---|---:|",
    ]
    for pair, value in combined_summary["selected_feature_correlations"].items():
        md_lines.append(f"| `{pair.replace('__', ' / ')}` | {value} |")

    md_lines += ["", "## Errors", ""]
    if combined_summary["errors"]:
        md_lines.extend(f"- {error}" for error in combined_summary["errors"])
    else:
        md_lines.append("- None reported.")

    md_lines += ["", "## Warnings", ""]
    warnings = combined_summary["warnings"] + [
        f"{result['file']}: {warning}"
        for result in file_results for warning in result["warnings"]
    ]
    if warnings:
        md_lines.extend(f"- {warning}" for warning in warnings)
    else:
        md_lines.append("- None reported.")

    md_lines += [
        "",
        "## Interpretation and safeguards",
        "",
        "- `target` and `phase_label` are labels/metadata, not candidate predictors.",
        "- `normal_message_count` and `attack_message_count` are ground-truth-derived; exclude them from predictors.",
        "- Exclude IDs, timestamps, scenario, window indices, and configured attack rate from model predictors.",
        "- Review `unique_tcp_streams`: it may reveal whether the extra flood publisher was present.",
        "- Duplicate predictor vectors need review before splitting; identical rows must not be scattered across train and test.",
        "- Mixed windows remain mixed. Define their treatment before supervised binary classification.",
        "- Correlation values are descriptive only and do not establish causation.",
        "- Passing this audit does not establish generalization, unseen-attack detection, or model performance.",
        "",
        "## Output files",
        "",
        "- `data/processed/combined_pcap_features_v2.csv`",
        "- `reports/combined_feature_dataset_audit_v2.csv`",
        "- `reports/combined_feature_dataset_audit_v2.json`",
        "- `reports/combined_feature_dataset_audit_v2.md`",
        "",
        "Source feature CSV files were not modified.",
    ]
    audit_md = REPORTS / "combined_feature_dataset_audit_v2.md"
    audit_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print("IOT SENTINEL AI — COMBINED V2 FEATURE DATASET AUDIT")
    print("=" * 68)
    print(f"Files passed       : {sum(r['passed'] for r in file_results)}/{len(file_results)}")
    print(f"Combined rows      : {len(combined_rows)}/{expected_total_rows}")
    print(f"Combined columns   : {combined_summary['combined_columns']}")
    print(f"Exact duplicates   : {combined_summary['exact_duplicate_rows']}")
    print(f"Predictor duplicates: {combined_summary['duplicate_predictor_rows']}")
    print(f"Blank cells        : {combined_summary['missing_cells']}")
    print(f"Infinite predictors: {combined_summary['infinite_predictor_cells']}")
    print(f"Overall result     : {'PASS' if combined_summary['overall_passed'] else 'REVIEW'}")

    for result in file_results:
        print(
            f"\n[{'PASS' if result['passed'] else 'REVIEW'}] {result['file']} "
            f"— rows={result['rows']}, columns={result['columns']}"
        )
        for error in result["errors"]:
            print(f"  ERROR: {error}")
        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")

    print("\nReports and combined dataset written:")
    print("  data/processed/combined_pcap_features_v2.csv")
    print("  reports/combined_feature_dataset_audit_v2.csv")
    print("  reports/combined_feature_dataset_audit_v2.json")
    print("  reports/combined_feature_dataset_audit_v2.md")
    print("Source feature CSV files were not modified.")

    return 0 if combined_summary["overall_passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
