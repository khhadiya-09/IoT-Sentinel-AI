"""Read-only consolidated audit for IoT Sentinel AI experiment runs.

Run from the project root:
    python notebooks/experiment_dataset_audit.py

The script reads experiment files and writes reports under reports/.
It does not modify raw experiment data.
"""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "data" / "raw" / "experiments"
REPORTS = ROOT / "reports"

# Primary dataset: three 300-second normal runs and two runs at each
# configured message-flooding rate. The short historical baseline is separate.
MANIFEST = [
    ("normal_baseline_v2_20261108T130132Z_0d8534df", "normal_baseline", "primary", 300, None, 325, 0),
    ("normal_baseline_v2_20261111T174556Z_d6e153b1", "normal_baseline", "primary", 300, None, 325, 0),
    ("normal_baseline_v2_20261111T175428Z_b01e38fe", "normal_baseline", "primary", 300, None, 325, 0),
    ("flood_v2_2rps_run01_20261111T161433Z_13332d0b", "message_flooding", "primary", 60, 2, 65, 60),
    ("flood_v2_2rps_run02_20261111T180926Z_39e51e4c", "message_flooding", "primary", 60, 2, 65, 60),
    ("flood_v2_5rps_run01_20261111T164008Z_05155788", "message_flooding", "primary", 60, 5, 65, 150),
    ("flood_v2_5rps_run02_20261111T181124Z_80434161", "message_flooding", "primary", 60, 5, 65, 150),
    ("flood_v2_10rps_run01_20261111T165033Z_8d2db887", "message_flooding", "primary", 60, 10, 65, 300),
    ("flood_v2_10rps_run02_20261111T181318Z_de9827ee", "message_flooding", "primary", 60, 10, 65, 300),
    ("normal_baseline_v2_20261108T125829Z_f05f1992", "normal_baseline", "pilot", 30, None, 33, 0),
]
REQUIRED_COLUMNS = {
    "device_id", "device_type", "timestamp", "temperature", "humidity",
    "battery", "status", "traffic_label", "experiment_id", "mqtt_topic",
    "collected_at",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compare_csv_jsonl(csv_rows, json_rows):
    if len(csv_rows) != len(json_rows):
        return False
    for csv_row, json_row in zip(csv_rows, json_rows):
        if set(csv_row) != set(json_row):
            return False
        for key, csv_value in csv_row.items():
            value = json_row[key]
            if value is None:
                if csv_value not in ("", "None"):
                    return False
            elif isinstance(value, bool):
                if csv_value.lower() != str(value).lower():
                    return False
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    if not math.isclose(float(csv_value), float(value), rel_tol=1e-10, abs_tol=1e-10):
                        return False
                except (TypeError, ValueError):
                    return False
            elif csv_value != str(value):
                return False
    return True


def extract_log_value(log_text, pattern, cast=str):
    match = re.search(pattern, log_text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def seed_from(metadata):
    for obj in (metadata, metadata.get("reproducibility", {})):
        if isinstance(obj, dict):
            for key in ("simulator_random_seed", "random_seed"):
                if key in obj:
                    return obj[key]
    return None


def audit(entry):
    run_id, scenario, role, duration, expected_rate, expected_normal, expected_attack = entry
    folder = EXPERIMENTS / run_id
    pcap_name = "message_flooding.pcapng" if scenario == "message_flooding" else "normal_baseline.pcapng"
    required_files = ["metadata.json", "telemetry.csv", "telemetry.jsonl", pcap_name]

    row = {
        "run_id": run_id, "scenario": scenario, "role": role, "passed": False,
        "duration_seconds": None, "random_seed": None,
        "configured_attack_rate": None, "measured_submission_rate": None,
        "publisher_messages_submitted": None, "publisher_messages_acknowledged": None,
        "collector_messages_received": None, "collector_messages_rejected": None,
        "csv_records": 0, "jsonl_records": 0, "missing_values": None,
        "label_distribution": {}, "device_distribution": {},
        "pcap_size_bytes": 0, "errors": [],
    }
    if not folder.is_dir():
        row["errors"].append("Experiment directory not found")
        return row

    for filename in required_files:
        if not (folder / filename).is_file():
            row["errors"].append(f"Missing file: {filename}")

    try:
        metadata_path = folder / "metadata.json"
        csv_path = folder / "telemetry.csv"
        jsonl_path = folder / "telemetry.jsonl"
        log_path = folder / "experiment.log"
        pcap_path = folder / pcap_name

        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        row["duration_seconds"] = metadata.get("duration_seconds")
        row["random_seed"] = seed_from(metadata)

        if metadata.get("status") != "completed":
            row["errors"].append("Metadata status is not completed")
        if metadata.get("run_id") != run_id:
            row["errors"].append("Metadata run_id mismatch")
        if metadata.get("scenario") != scenario:
            row["errors"].append("Metadata scenario mismatch")
        if row["duration_seconds"] != duration:
            row["errors"].append("Duration mismatch")

        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        if scenario == "message_flooding":
            row["configured_attack_rate"] = extract_log_value(
                log_text, r"Configured rate:\s*([0-9]+(?:\.[0-9]+)?)\s*msg/s", float
            )
            row["measured_submission_rate"] = extract_log_value(
                log_text, r"Measured submission rate:\s*([0-9]+(?:\.[0-9]+)?)\s*msg/s", float
            )
            row["publisher_messages_submitted"] = extract_log_value(
                log_text, r"Messages submitted:\s*(\d+)", int
            )
            row["publisher_messages_acknowledged"] = extract_log_value(
                log_text, r"Messages acknowledged:\s*(\d+)", int
            )
            row["collector_messages_received"] = extract_log_value(
                log_text, r"Received messages:\s*(\d+)", int
            )
            row["collector_messages_rejected"] = extract_log_value(
                log_text, r"Rejected messages:\s*(\d+)", int
            )

            if row["configured_attack_rate"] is None:
                row["errors"].append("Configured attack rate not found in experiment.log")
            elif not math.isclose(row["configured_attack_rate"], expected_rate, abs_tol=1e-9):
                row["errors"].append("Configured attack rate differs from manifest")
            if row["measured_submission_rate"] is None:
                row["errors"].append("Measured submission rate not found in experiment.log")
            if row["publisher_messages_submitted"] != expected_attack:
                row["errors"].append("Publisher submitted-message count differs from expected")
            if row["publisher_messages_acknowledged"] != expected_attack:
                row["errors"].append("Publisher acknowledgement count differs from expected")
            if row["collector_messages_rejected"] != 0:
                row["errors"].append("Collector rejected messages or rejection count is missing")

        csv_rows, json_rows = [], []
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                csv_rows = list(reader)
            row["csv_records"] = len(csv_rows)
            if not REQUIRED_COLUMNS.issubset(columns):
                row["errors"].append("Required telemetry columns missing")
            row["missing_values"] = sum(
                1 for record in csv_rows
                for value in record.values()
                if value is None or value.strip() == ""
            )
            if row["missing_values"]:
                row["errors"].append("CSV contains missing/empty values")

            row["label_distribution"] = dict(Counter(
                record.get("traffic_label", "") for record in csv_rows
            ))
            row["device_distribution"] = dict(Counter(
                record.get("device_id", "") for record in csv_rows
            ))
            experiment_ids = {record.get("experiment_id", "") for record in csv_rows}
            if experiment_ids != {metadata.get("experiment_id", "")}:
                row["errors"].append("Telemetry experiment_id mismatch")

        if jsonl_path.is_file():
            with jsonl_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"JSONL line {line_number} is not an object")
                        json_rows.append(record)
            row["jsonl_records"] = len(json_rows)

        if csv_path.is_file() and jsonl_path.is_file():
            if not compare_csv_jsonl(csv_rows, json_rows):
                row["errors"].append("CSV and JSONL records/content differ")

        if pcap_path.is_file():
            row["pcap_size_bytes"] = pcap_path.stat().st_size
            if row["pcap_size_bytes"] <= 0:
                row["errors"].append("PCAP is empty")

        normal_count = row["label_distribution"].get("normal", 0)
        attack_count = row["label_distribution"].get("message_flooding", 0)
        if (normal_count, attack_count) != (expected_normal, expected_attack):
            row["errors"].append("Telemetry label counts differ from expected")
        if set(row["label_distribution"]) - {"normal", "message_flooding"}:
            row["errors"].append("Unexpected traffic label found")
        if row["csv_records"] != expected_normal + expected_attack:
            row["errors"].append("CSV total record count differs from expected")
        if row["csv_records"] != row["jsonl_records"]:
            row["errors"].append("CSV/JSONL record counts differ")

    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")

    row["passed"] = not row["errors"]
    return row


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = [audit(entry) for entry in MANIFEST]
    primary = [result for result in results if result["role"] == "primary"]
    pilot = [result for result in results if result["role"] == "pilot"]

    fields = [
        "run_id", "scenario", "role", "passed", "duration_seconds", "random_seed",
        "configured_attack_rate", "measured_submission_rate",
        "publisher_messages_submitted", "publisher_messages_acknowledged",
        "collector_messages_received", "collector_messages_rejected",
        "csv_records", "jsonl_records", "missing_values",
        "label_distribution", "device_distribution", "pcap_size_bytes", "errors",
    ]
    csv_report = REPORTS / "experiment_dataset_audit.csv"
    with csv_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            item = dict(result)
            item["label_distribution"] = json.dumps(item["label_distribution"])
            item["device_distribution"] = json.dumps(item["device_distribution"])
            item["errors"] = " | ".join(item["errors"])
            writer.writerow({key: item.get(key) for key in fields})

    json_report = REPORTS / "experiment_dataset_audit.json"
    with json_report.open("w", encoding="utf-8") as handle:
        json.dump({
            "audit_version": "2.0",
            "primary_runs": len(primary),
            "primary_runs_passed": sum(result["passed"] for result in primary),
            "pilot_runs": len(pilot),
            "pilot_runs_passed": sum(result["passed"] for result in pilot),
            "results": results,
        }, handle, indent=4)

    summary = [
        "# IoT Sentinel AI — Consolidated Experiment Dataset Audit", "",
        "Read-only audit. Raw experiment files were not modified.", "",
        f"- Primary runs passed: {sum(r['passed'] for r in primary)}/{len(primary)}",
        f"- Primary CSV records: {sum(r['csv_records'] for r in primary)}",
        f"- Pilot runs passed: {sum(r['passed'] for r in pilot)}/{len(pilot)}",
        f"- Pilot CSV records: {sum(r['csv_records'] for r in pilot)}", "",
        "| Role | Run ID | Result | Configured rate | Measured rate | Submitted | ACKed | Received | Rejected | Issues |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        issues = "; ".join(result["errors"]) if result["errors"] else "None"
        summary.append(
            f"| {result['role']} | `{result['run_id']}` | "
            f"{'PASS' if result['passed'] else 'REVIEW'} | "
            f"{result['configured_attack_rate']} | {result['measured_submission_rate']} | "
            f"{result['publisher_messages_submitted']} | "
            f"{result['publisher_messages_acknowledged']} | "
            f"{result['collector_messages_received']} | "
            f"{result['collector_messages_rejected']} | {issues} |"
        )
    summary += [
        "", "## Interpretation", "",
        "- Configured and measured submission rates are distinct measurements.",
        "- Publisher acknowledgements and collector receipt counts are distinct measurements.",
        "- Random seeds are reported only when present in metadata; missing seeds are not inferred.",
        "- PCAP files are checked for presence and non-zero size; packet-level verification is separate.",
        "- Passing this audit does not establish model performance or dataset sufficiency.", "",
    ]
    (REPORTS / "experiment_dataset_audit_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print("IOT SENTINEL AI — CONSOLIDATED DATASET AUDIT V2")
    print("=" * 64)
    print(f"Primary runs: {sum(r['passed'] for r in primary)}/{len(primary)} passed")
    print(f"Primary CSV records: {sum(r['csv_records'] for r in primary)}")
    print(f"Pilot runs: {sum(r['passed'] for r in pilot)}/{len(pilot)} passed")
    print(f"Pilot CSV records: {sum(r['csv_records'] for r in pilot)}")

    for result in results:
        print(f"\n[{'PASS' if result['passed'] else 'REVIEW'}] {result['role']} — {result['run_id']}")
        print(
            f"  CSV={result['csv_records']} JSONL={result['jsonl_records']} "
            f"PCAP={result['pcap_size_bytes']} bytes"
        )
        print(
            f"  Seed={result['random_seed']} | "
            f"Configured rate={result['configured_attack_rate']} | "
            f"Measured rate={result['measured_submission_rate']}"
        )
        if result["scenario"] == "message_flooding":
            print(
                f"  Submitted={result['publisher_messages_submitted']} | "
                f"ACKed={result['publisher_messages_acknowledged']} | "
                f"Collector received={result['collector_messages_received']} | "
                f"Rejected={result['collector_messages_rejected']}"
            )
        for issue in result["errors"]:
            print(f"  ISSUE: {issue}")

    print("\nReports written:")
    print("  reports/experiment_dataset_audit.csv")
    print("  reports/experiment_dataset_audit.json")
    print("  reports/experiment_dataset_audit_summary.md")
    print("Raw experiment files were not modified.")


if __name__ == "__main__":
    main()
