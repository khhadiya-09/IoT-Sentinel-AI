"""
IoT Sentinel AI - MQTT PCAP window feature extractor.

Extracts publisher-to-broker MQTT frames from a PCAPNG file, aggregates
network features into fixed-duration windows, and writes one CSV file.

Ground-truth labels are derived from decoded MQTT JSON payloads and the
experiment metadata. They are annotation columns, not model predictors.

Example (run from the project root):
python scripts/extract_pcap_features.py `
  --pcap "data/raw/experiments/RUN_FOLDER/message_flooding.pcapng" `
  --metadata "data/raw/experiments/RUN_FOLDER/metadata.json" `
  --output "data/processed/RUN_FOLDER_features.csv" `
  --tshark "C:/Program Files/Wireshark/tshark.exe"
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


MQTT_TYPE_NAMES = {
    1: "connect",
    2: "connack",
    3: "publish",
    4: "puback",
    5: "pubrec",
    6: "pubrel",
    7: "pubcomp",
    8: "subscribe",
    9: "suback",
    10: "unsubscribe",
    11: "unsuback",
    12: "pingreq",
    13: "pingresp",
    14: "disconnect",
    15: "auth",
}

FEATURE_COLUMNS = [
    "packet_count",
    "mqtt_publish_count",
    "mqtt_connect_count",
    "mqtt_subscribe_count",
    "mqtt_unsubscribe_count",
    "mqtt_disconnect_count",
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
    "unique_source_ports",
]

OUTPUT_COLUMNS = [
    "experiment_id",
    "window_index",
    "window_start_epoch",
    "window_end_epoch",
    "window_duration_seconds",
    "phase",
    "label",
    *FEATURE_COLUMNS,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract window-based network features from an MQTT PCAP."
    )
    parser.add_argument("--pcap", required=True, help="Input PCAP/PCAPNG file.")
    parser.add_argument("--metadata", required=True, help="Experiment metadata.json.")
    parser.add_argument("--output", required=True, help="Output feature CSV.")
    parser.add_argument(
        "--tshark",
        default=r"C:\Program Files\Wireshark\tshark.exe",
        help="Path to tshark.exe.",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=10.0,
        help="Window duration in seconds (default: 10).",
    )
    return parser.parse_args()


def load_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    experiment_id = metadata.get("experiment_id")
    timing = metadata.get("phase_timing", {})
    attack_start = timing.get("attack_started_at_utc")
    attack_end = timing.get("attack_ended_at_utc")

    if not experiment_id:
        raise ValueError("metadata.json is missing experiment_id.")
    if not attack_start or not attack_end:
        raise ValueError("metadata.json is missing phase_timing attack timestamps.")

    from datetime import datetime

    def to_epoch(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"Timestamp must include timezone: {value}")
        return parsed.timestamp()

    return experiment_id, to_epoch(attack_start), to_epoch(attack_end)


def run_tshark(tshark, pcap):
    display_filter = "tcp.dstport == 1883 && mqtt"
    command = [
        str(tshark),
        "-r", str(pcap),
        "-Y", display_filter,
        "-T", "fields",
        "-E", "separator=\t",
        "-E", "quote=d",
        "-E", "occurrence=f",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-e", "tcp.stream",
        "-e", "tcp.srcport",
        "-e", "mqtt.msgtype",
        "-e", "mqtt.msg",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TShark failed:\n" + completed.stderr.strip()
        )
    return completed.stdout


def parse_tshark_rows(output):
    records = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            fields = next(csv.reader([line], delimiter="\t", quotechar='"'))
            fields += [""] * (6 - len(fields))

            timestamp = float(fields[0])
            frame_length = int(fields[1])
            stream = fields[2] or None
            source_port = fields[3] or None
            msg_type = int(fields[4]) if fields[4] else None
            payload_hex = fields[5].replace(":", "").replace(" ", "")

            payload = {}
            if payload_hex:
                try:
                    decoded = bytes.fromhex(payload_hex).decode("utf-8")
                    candidate = json.loads(decoded)
                    if isinstance(candidate, dict):
                        payload = candidate
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    pass

            records.append({
                "timestamp": timestamp,
                "length": frame_length,
                "stream": stream,
                "source_port": source_port,
                "msg_type": msg_type,
                "payload": payload,
            })
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse TShark output line {line_number}: {line}"
            ) from exc

    return records


def calculate_features(records, window_seconds):
    lengths = [record["length"] for record in records]
    timestamps = sorted(record["timestamp"] for record in records)
    msg_types = Counter(
        record["msg_type"] for record in records
        if record["msg_type"] is not None
    )

    interarrivals = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
    ]

    def mean_or_zero(values):
        return statistics.mean(values) if values else 0.0

    def std_or_zero(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    duration = window_seconds
    return {
        "packet_count": len(records),
        "mqtt_publish_count": msg_types[3],
        "mqtt_connect_count": msg_types[1],
        "mqtt_subscribe_count": msg_types[8],
        "mqtt_unsubscribe_count": msg_types[10],
        "mqtt_disconnect_count": msg_types[14],
        "total_bytes": sum(lengths),
        "mean_packet_size_bytes": mean_or_zero(lengths),
        "std_packet_size_bytes": std_or_zero(lengths),
        "min_packet_size_bytes": min(lengths) if lengths else 0,
        "max_packet_size_bytes": max(lengths) if lengths else 0,
        "packet_rate_per_second": len(records) / duration,
        "publish_rate_per_second": msg_types[3] / duration,
        "mean_interarrival_seconds": mean_or_zero(interarrivals),
        "std_interarrival_seconds": std_or_zero(interarrivals),
        "min_interarrival_seconds": min(interarrivals) if interarrivals else 0.0,
        "max_interarrival_seconds": max(interarrivals) if interarrivals else 0.0,
        "unique_tcp_streams": len({
            record["stream"] for record in records if record["stream"] is not None
        }),
        "unique_source_ports": len({
            record["source_port"] for record in records
            if record["source_port"] is not None
        }),
    }


def main():
    args = parse_args()
    pcap = Path(args.pcap)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    tshark = Path(args.tshark)

    if not pcap.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not tshark.is_file():
        raise FileNotFoundError(f"TShark executable not found: {tshark}")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        raise ValueError("--window-seconds must be a positive number.")

    experiment_id, attack_start, attack_end = load_metadata(metadata_path)
    if attack_end <= attack_start:
        raise ValueError("Attack end timestamp must be after attack start.")

    print("Reading MQTT publisher-to-broker frames from PCAP...")
    tshark_output = run_tshark(tshark, pcap)
    records = parse_tshark_rows(tshark_output)

    if not records:
        raise RuntimeError(
            "No MQTT frames were extracted. Check the PCAP and TShark MQTT dissection."
        )

    records.sort(key=lambda item: item["timestamp"])
    capture_start = records[0]["timestamp"]
    capture_end = records[-1]["timestamp"]

    windows = defaultdict(list)
    for record in records:
        index = int((record["timestamp"] - capture_start) // args.window_seconds)
        windows[index].append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        discarded_partial_windows = 0

        for index in sorted(windows):
            window_records = windows[index]
            start = capture_start + index * args.window_seconds
            end = start + args.window_seconds

            # The last timestamp is the last observed frame, not the capture
            # stop time. Keep only windows whose full interval is observed.
            if end > capture_end + 1e-9:
                discarded_partial_windows += 1
                continue

            flooding_count = sum(
                1 for record in window_records
                if record["payload"].get("traffic_label") == "message_flooding"
            )
            label = "attack" if flooding_count > 0 else "normal"

            # Mark windows crossing a phase boundary separately so they can
            # be excluded from strict phase-based evaluation if needed.
            if end <= attack_start:
                phase = "pre_attack"
            elif start >= attack_end:
                phase = "post_attack"
            elif start >= attack_start and end <= attack_end:
                phase = "attack"
            else:
                phase = "boundary_mixed"

            row = {
                "experiment_id": experiment_id,
                "window_index": index,
                "window_start_epoch": round(start, 6),
                "window_end_epoch": round(end, 6),
                "window_duration_seconds": args.window_seconds,
                "phase": phase,
                "label": label,
            }
            row.update(calculate_features(window_records, args.window_seconds))
            writer.writerow(row)
            rows_written += 1

    print("\nFeature extraction completed.")
    print(f"Experiment ID       : {experiment_id}")
    print(f"MQTT frames parsed  : {len(records)}")
    print(f"Capture start epoch : {capture_start:.6f}")
    print(f"Capture end epoch   : {capture_end:.6f}")
    print(f"Window size         : {args.window_seconds:g} seconds")
    print(f"Feature windows     : {rows_written}")
    print(f"Partial windows discarded: {discarded_partial_windows}")
    print(f"Output CSV          : {output_path}")
    print("\nLabel counts:")
    label_counts = Counter()
    phase_counts = Counter()
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_counts[row["label"]] += 1
            phase_counts[row["phase"]] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("Phase counts:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
