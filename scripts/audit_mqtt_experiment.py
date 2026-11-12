"""
Phase-wise MQTT experiment audit for IoT Sentinel AI.

Reads publisher-to-broker MQTT PUBLISH packets from a PCAP using TShark,
decodes the JSON payloads, and compares phase-wise counts with the collector CSV.

Example (PowerShell, from the project root):
python scripts/audit_mqtt_experiment.py `
  --pcap "data/raw/experiments/RUN_FOLDER/message_flooding.pcapng" `
  --csv "data/raw/experiments/RUN_FOLDER/telemetry.csv" `
  --attack-start-epoch 1794413689.904 `
  --attack-end-epoch 1794413720.287
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DISPLAY_FILTER = "mqtt.msgtype == 3 && tcp.dstport == 1883"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit phase-wise MQTT PUBLISH traffic against collector CSV."
    )
    parser.add_argument("--pcap", required=True, help="Path to the experiment PCAPNG file.")
    parser.add_argument("--csv", required=True, help="Path to the collector telemetry CSV.")
    parser.add_argument(
        "--attack-start-epoch",
        required=True,
        type=float,
        help="Unix epoch timestamp at which the attack phase starts.",
    )
    parser.add_argument(
        "--attack-end-epoch",
        required=True,
        type=float,
        help="Unix epoch timestamp at which the attack phase ends.",
    )
    parser.add_argument(
        "--tshark",
        default="tshark",
        help='TShark executable; defaults to "tshark". On Windows, provide its full path if needed.',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path. Defaults to phase_audit.json beside the CSV.",
    )
    return parser.parse_args()


def phase_for_epoch(timestamp: float, attack_start: float, attack_end: float) -> str:
    if timestamp < attack_start:
        return "pre_attack"
    if timestamp < attack_end:
        return "attack"
    return "post_attack"


def decode_mqtt_payload(payload_hex: str) -> dict[str, Any]:
    cleaned = "".join(payload_hex.split()).replace(":", "")
    if not cleaned:
        raise ValueError("Empty mqtt.msg payload")
    decoded = bytes.fromhex(cleaned).decode("utf-8")
    value = json.loads(decoded)
    if not isinstance(value, dict):
        raise ValueError("MQTT payload JSON is not an object")
    return value


def parse_iso_timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value!r}")
    return parsed.timestamp()


def increment_counts(
    counts: dict[str, Counter],
    phase: str,
    label: str,
    device_id: str,
) -> None:
    counts[phase]["total"] += 1
    counts[phase][f"label:{label}"] += 1
    counts[phase][f"device:{device_id}"] += 1


def read_pcap(
    pcap_path: Path,
    tshark_path: str,
    attack_start: float,
    attack_end: float,
) -> tuple[dict[str, Counter], int]:
    command = [
        tshark_path,
        "-r",
        str(pcap_path),
        "-Y",
        DISPLAY_FILTER,
        "-T",
        "fields",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=f",
        "-e",
        "frame.time_epoch",
        "-e",
        "mqtt.msg",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TShark failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Error: {result.stderr.strip()}"
        )

    counts: dict[str, Counter] = defaultdict(Counter)
    decoded_count = 0
    skipped_count = 0

    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            skipped_count += 1
            continue
        try:
            frame_epoch = float(parts[0])
            payload = decode_mqtt_payload(parts[1])
            label = str(payload.get("traffic_label", "missing"))
            device_id = str(payload.get("device_id", "missing"))
            phase = phase_for_epoch(frame_epoch, attack_start, attack_end)
            increment_counts(counts, phase, label, device_id)
            decoded_count += 1
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            skipped_count += 1
            print(
                f"Warning: skipped PCAP output row {line_number}: {exc}",
                file=sys.stderr,
            )

    if skipped_count:
        print(f"Warning: {skipped_count} TShark output rows were skipped.", file=sys.stderr)
    return counts, decoded_count


def read_collector_csv(
    csv_path: Path,
    attack_start: float,
    attack_end: float,
) -> tuple[dict[str, Counter], int]:
    counts: dict[str, Counter] = defaultdict(Counter)
    required = {"timestamp", "traffic_label", "device_id"}
    total = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = required - available
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_iso_timestamp(row["timestamp"])
                phase = phase_for_epoch(timestamp, attack_start, attack_end)
                label = (row["traffic_label"] or "missing").strip()
                device_id = (row["device_id"] or "missing").strip()
                increment_counts(counts, phase, label, device_id)
                total += 1
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {row_number}: {exc}") from exc

    return counts, total


def normalise_counts(counts: dict[str, Counter]) -> dict[str, dict[str, int]]:
    phases = ("pre_attack", "attack", "post_attack")
    result: dict[str, dict[str, int]] = {}
    for phase in phases:
        result[phase] = dict(sorted(counts.get(phase, Counter()).items()))
    return result


def main() -> int:
    args = parse_args()
    pcap_path = Path(args.pcap).resolve()
    csv_path = Path(args.csv).resolve()

    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if args.attack_end_epoch <= args.attack_start_epoch:
        raise ValueError("--attack-end-epoch must be greater than --attack-start-epoch")

    tshark_path = shutil.which(args.tshark)
    if tshark_path is None and Path(args.tshark).is_file():
        tshark_path = str(Path(args.tshark).resolve())
    if tshark_path is None:
        raise FileNotFoundError(
            f"Could not find TShark executable: {args.tshark!r}. "
            "Pass its full path with --tshark."
        )

    pcap_counts, pcap_total = read_pcap(
        pcap_path, tshark_path, args.attack_start_epoch, args.attack_end_epoch
    )
    csv_counts, csv_total = read_collector_csv(
        csv_path, args.attack_start_epoch, args.attack_end_epoch
    )

    phases = ("pre_attack", "attack", "post_attack")
    comparison = {}
    for phase in phases:
        pcap_phase = pcap_counts.get(phase, Counter())
        csv_phase = csv_counts.get(phase, Counter())
        comparison[phase] = {
            "pcap_total": pcap_phase.get("total", 0),
            "csv_total": csv_phase.get("total", 0),
            "total_counts_match": (
                pcap_phase.get("total", 0) == csv_phase.get("total", 0)
            ),
            "pcap_labels": {
                key.removeprefix("label:"): value
                for key, value in pcap_phase.items()
                if key.startswith("label:")
            },
            "csv_labels": {
                key.removeprefix("label:"): value
                for key, value in csv_phase.items()
                if key.startswith("label:")
            },
            "pcap_devices": {
                key.removeprefix("device:"): value
                for key, value in pcap_phase.items()
                if key.startswith("device:")
            },
            "csv_devices": {
                key.removeprefix("device:"): value
                for key, value in csv_phase.items()
                if key.startswith("device:")
            },
        }

    all_phase_counts_match = all(
        comparison[phase]["total_counts_match"] for phase in phases
    )
    report = {
        "audit_version": "1.0",
        "pcap_file": str(pcap_path),
        "collector_csv": str(csv_path),
        "display_filter": DISPLAY_FILTER,
        "attack_start_epoch": args.attack_start_epoch,
        "attack_end_epoch": args.attack_end_epoch,
        "pcap_decoded_publish_messages": pcap_total,
        "collector_csv_records": csv_total,
        "overall_total_counts_match": pcap_total == csv_total,
        "all_phase_total_counts_match": all_phase_counts_match,
        "pcap_phase_counts": normalise_counts(pcap_counts),
        "csv_phase_counts": normalise_counts(csv_counts),
        "phase_comparison": comparison,
        "limitations": [
            "PCAP counts include publisher-to-broker MQTT PUBLISH packets only.",
            "CSV phases use the telemetry timestamp; PCAP phases use frame.time_epoch.",
            "Small boundary differences can occur because capture and telemetry timestamps are recorded at different points.",
            "device_id and traffic_label are used only for auditing, not as model predictors.",
        ],
    }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else csv_path.parent / "phase_audit.json"
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== IoT Sentinel AI: MQTT Phase Audit ===")
    print(f"PCAP PUBLISH messages decoded : {pcap_total}")
    print(f"Collector CSV records         : {csv_total}")
    print("\nPhase             PCAP    CSV    Match")
    for phase in phases:
        item = comparison[phase]
        print(
            f"{phase:<17} {item['pcap_total']:>4} "
            f"{item['csv_total']:>6}    "
            f"{'YES' if item['total_counts_match'] else 'NO'}"
        )
    print(f"\nOverall total counts match: {'YES' if pcap_total == csv_total else 'NO'}")
    print(
        "All phase totals match   : "
        f"{'YES' if all_phase_counts_match else 'NO'}"
    )
    print(f"JSON report saved to: {output_path}")

    return 0 if pcap_total == csv_total and all_phase_counts_match else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
