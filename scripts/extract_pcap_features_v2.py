"""
IoT Sentinel AI - MQTT PCAP Feature Extractor V2

Capabilities:
1. Input and metadata validation
2. Dynamic MQTT publisher identification
3. Publisher-to-broker traffic extraction
4. MQTT JSON payload decoding and validation
5. Non-overlapping feature windows relative to the first captured MQTT frame
6. Network and IoT telemetry feature calculation
7. Payload-derived targets and metadata-derived experimental phase labels
8. CSV output with overwrite protection
"""

import argparse
import csv
import json
import math
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DEFAULT_TSHARK = r"C:\Program Files\Wireshark\tshark.exe"

BROKER_PORT = 1883

SIMULATOR_CLIENT_ID = "iot_sentinel_device_simulator"
FLOOD_CLIENT_ID = "iot_sentinel_flood_client_v2"

MQTT_PUBLISH_TYPE = 3

NORMAL_LABEL = "normal"
ATTACK_LABEL = "message_flooding"

EXPECTED_DEVICES = {
    "simulator": {
        "sensor_001",
        "sensor_002",
        "sensor_003",
    },
    "flood_publisher": {
        "flood_client_001",
    },
}

EXPECTED_LABELS = {
    "simulator": NORMAL_LABEL,
    "flood_publisher": ATTACK_LABEL,
}

REQUIRED_PAYLOAD_FIELDS = {
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
}

NUMERIC_PAYLOAD_FIELDS = (
    "temperature",
    "humidity",
    "battery",
)


# ---------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MQTT publisher traffic and generate feature windows."
    )

    parser.add_argument("--pcap", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--tshark",
        default=DEFAULT_TSHARK,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help=(
            "Optional window-size override. If omitted, the value "
            "from metadata.json is used."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

def validate_inputs(args):
    pcap = Path(args.pcap)
    metadata_path = Path(args.metadata)
    tshark = Path(args.tshark)
    output = Path(args.output)

    if not pcap.is_file():
        raise FileNotFoundError(f"PCAP not found: {pcap}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    if not tshark.is_file():
        raise FileNotFoundError(f"TShark not found: {tshark}")

    if args.window_seconds is not None:
        if (
            isinstance(args.window_seconds, bool)
            or not math.isfinite(args.window_seconds)
            or args.window_seconds <= 0
        ):
            raise ValueError("--window-seconds must be a finite positive number.")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    if not output.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output.parent}"
        )

    return pcap, metadata_path, tshark, output


# ---------------------------------------------------------
# METADATA LOADING
# ---------------------------------------------------------

def parse_utc_timestamp(value, field_name):
    if not value:
        return None

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp string.")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in {field_name}: {value}") from exc

    if parsed.tzinfo is None:
        raise ValueError(f"Timezone missing from timestamp in {field_name}: {value}")

    return parsed.astimezone(timezone.utc).timestamp()


def load_metadata(metadata_path, window_override=None):
    with metadata_path.open("r", encoding="utf-8") as file:
        raw_metadata = json.load(file)

    experiment_id = raw_metadata.get("experiment_id")
    run_id = raw_metadata.get("run_id")
    scenario = raw_metadata.get("scenario")

    if not experiment_id or not run_id:
        raise ValueError("Missing experiment_id or run_id in metadata.")

    if scenario not in {"normal_baseline", "message_flooding"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    windowing = raw_metadata.get("windowing", {})
    metadata_window_size = windowing.get("window_size_seconds")

    if window_override is not None:
        window_size = window_override
    else:
        window_size = metadata_window_size

    if (
        not isinstance(window_size, (int, float))
        or isinstance(window_size, bool)
        or not math.isfinite(window_size)
        or window_size <= 0
    ):
        raise ValueError("Invalid or missing window_size_seconds.")

    timing = raw_metadata.get("phase_timing", {})

    attack_start = parse_utc_timestamp(
        timing.get("attack_started_at_utc"),
        "phase_timing.attack_started_at_utc",
    )
    attack_end = parse_utc_timestamp(
        timing.get("attack_ended_at_utc"),
        "phase_timing.attack_ended_at_utc",
    )

    if scenario == "message_flooding":
        if attack_start is None or attack_end is None:
            raise ValueError("Flooding attack timestamps are missing.")

        if attack_end <= attack_start:
            raise ValueError("Invalid attack time interval.")

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "scenario": scenario,
        "attack_start_epoch": attack_start,
        "attack_end_epoch": attack_end,
        "window_size_seconds": float(window_size),
    }


# ---------------------------------------------------------
# TSHARK EXECUTION
# ---------------------------------------------------------

def run_tshark(tshark, pcap, display_filter, fields):
    command = [
        str(tshark),
        "-r", str(pcap),
        "-Y", display_filter,
        "-T", "fields",
        "-E", "separator=\t",
        "-E", "quote=d",
        "-E", "occurrence=f",
    ]

    for field in fields:
        command.extend(["-e", field])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"TShark failed:\n{result.stderr}")

    return result.stdout


# ---------------------------------------------------------
# MQTT PUBLISHER IDENTIFICATION
# ---------------------------------------------------------

def identify_publishers(tshark, pcap, scenario):
    client_ids = {
        "simulator": SIMULATOR_CLIENT_ID,
    }

    if scenario == "message_flooding":
        client_ids["flood_publisher"] = FLOOD_CLIENT_ID

    publishers = {}

    for role, client_id in client_ids.items():
        display_filter = (
            'mqtt.msgtype == 1 && '
            f'mqtt.clientid == "{client_id}"'
        )

        output = run_tshark(
            tshark,
            pcap,
            display_filter,
            ["tcp.stream", "mqtt.clientid"],
        )

        streams = set()

        for line in output.splitlines():
            if not line.strip():
                continue

            fields = next(
                csv.reader([line], delimiter="\t", quotechar='"')
            )

            if len(fields) < 2:
                continue

            stream_text, observed_client_id = fields[0], fields[1]

            if observed_client_id != client_id:
                continue

            if stream_text.strip():
                streams.add(int(stream_text))

        if len(streams) > 1:
            raise RuntimeError(
                f"Multiple TCP streams found for {client_id}: {streams}"
            )

        if len(streams) == 0:
            raise RuntimeError(
                f"No TCP stream found for MQTT client: {client_id}"
            )

        publishers[role] = next(iter(streams))

    return publishers


# ---------------------------------------------------------
# PUBLISHER TRAFFIC EXTRACTION
# ---------------------------------------------------------

def extract_publisher_frames(tshark, pcap, publishers):
    fields = [
        "frame.number",
        "frame.time_epoch",
        "frame.len",
        "tcp.stream",
        "tcp.srcport",
        "tcp.dstport",
        "mqtt.msgtype",
        "mqtt.msg",
    ]

    all_records = []

    for role, stream in publishers.items():
        display_filter = (
            f"tcp.stream == {stream} && "
            f"tcp.dstport == {BROKER_PORT} && "
            "mqtt"
        )

        output = run_tshark(
            tshark,
            pcap,
            display_filter,
            fields,
        )

        for line_number, line in enumerate(output.splitlines(), start=1):
            if not line.strip():
                continue

            values = next(
                csv.reader([line], delimiter="\t", quotechar='"')
            )

            if len(values) != len(fields):
                raise ValueError(
                    f"Unexpected TShark field count in {role}, "
                    f"line {line_number}: {values}"
                )

            try:
                record = {
                    "publisher_role": role,
                    "frame_number": int(values[0]),
                    "timestamp": float(values[1]),
                    "frame_length": int(values[2]),
                    "tcp_stream": int(values[3]),
                    "source_port": int(values[4]),
                    "destination_port": int(values[5]),
                    "mqtt_message_type": (
                        int(values[6]) if values[6] else None
                    ),
                    "mqtt_payload_hex": values[7],
                }
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Could not parse TShark output for {role}, "
                    f"line {line_number}: {line}"
                ) from exc

            if record["destination_port"] != BROKER_PORT:
                raise ValueError("Unexpected traffic direction detected.")

            if not math.isfinite(record["timestamp"]):
                raise ValueError("Non-finite PCAP timestamp encountered.")

            if record["frame_length"] <= 0:
                raise ValueError("Invalid captured frame length encountered.")

            all_records.append(record)

    all_records.sort(
        key=lambda record: (
            record["timestamp"],
            record["frame_number"],
        )
    )

    return all_records


# ---------------------------------------------------------
# MQTT JSON PAYLOAD DECODING AND VALIDATION
# ---------------------------------------------------------

def validate_payloads(records, metadata):
    statistics = {}

    for role in EXPECTED_DEVICES:
        statistics[role] = {
            "publish_frames": 0,
            "valid_payloads": 0,
            "invalid_payloads": 0,
            "decode_errors": 0,
            "json_errors": 0,
            "schema_errors": 0,
            "numeric_field_errors": 0,
            "timestamp_errors": 0,
            "experiment_id_mismatches": 0,
            "device_id_mismatches": 0,
            "traffic_label_mismatches": 0,
        }

    for record in records:
        if record["mqtt_message_type"] != MQTT_PUBLISH_TYPE:
            record["payload"] = None
            record["payload_valid"] = False
            record["payload_errors"] = []
            continue

        role = record["publisher_role"]
        stats = statistics[role]
        stats["publish_frames"] += 1

        errors = []
        payload = None
        payload_hex = record.get("mqtt_payload_hex") or ""

        try:
            cleaned_hex = (
                payload_hex
                .replace(":", "")
                .replace(" ", "")
                .strip()
            )

            if not cleaned_hex:
                raise ValueError("Empty MQTT payload.")

            payload_bytes = bytes.fromhex(cleaned_hex)

        except ValueError:
            stats["decode_errors"] += 1
            errors.append("decode_error")
            payload_bytes = None

        if payload_bytes is not None:
            try:
                decoded_text = payload_bytes.decode("utf-8")
                payload = json.loads(decoded_text)

                if not isinstance(payload, dict):
                    raise ValueError("JSON payload must be an object.")

            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                stats["json_errors"] += 1
                errors.append("json_error")
                payload = None

        if payload is not None:
            missing_fields = REQUIRED_PAYLOAD_FIELDS - set(payload.keys())

            if missing_fields:
                stats["schema_errors"] += 1
                errors.append(
                    "schema_error:" + ",".join(sorted(missing_fields))
                )

            if payload.get("experiment_id") != metadata["experiment_id"]:
                stats["experiment_id_mismatches"] += 1
                errors.append("experiment_id_mismatch")

            if payload.get("device_id") not in EXPECTED_DEVICES[role]:
                stats["device_id_mismatches"] += 1
                errors.append("device_id_mismatch")

            if payload.get("traffic_label") != EXPECTED_LABELS[role]:
                stats["traffic_label_mismatches"] += 1
                errors.append("traffic_label_mismatch")

            for field in ("device_id", "device_type", "status", "traffic_label",
                          "experiment_id"):
                value = payload.get(field)
                if value is not None and not isinstance(value, str):
                    stats["schema_errors"] += 1
                    errors.append(f"invalid_type:{field}")

            for field in NUMERIC_PAYLOAD_FIELDS:
                value = payload.get(field)

                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    stats["numeric_field_errors"] += 1
                    errors.append(f"invalid_numeric_field:{field}")

            humidity = payload.get("humidity")
            if (
                isinstance(humidity, (int, float))
                and not isinstance(humidity, bool)
                and math.isfinite(humidity)
                and not 0 <= humidity <= 100
            ):
                stats["numeric_field_errors"] += 1
                errors.append("humidity_out_of_range")

            battery = payload.get("battery")
            if (
                isinstance(battery, (int, float))
                and not isinstance(battery, bool)
                and math.isfinite(battery)
                and not 0 <= battery <= 100
            ):
                stats["numeric_field_errors"] += 1
                errors.append("battery_out_of_range")

            timestamp_value = payload.get("timestamp")
            try:
                if not isinstance(timestamp_value, str):
                    raise ValueError("Timestamp must be a string.")

                parsed_timestamp = datetime.fromisoformat(
                    timestamp_value.replace("Z", "+00:00")
                )

                if parsed_timestamp.tzinfo is None:
                    raise ValueError("Timestamp must include a timezone.")

            except ValueError:
                stats["timestamp_errors"] += 1
                errors.append("invalid_telemetry_timestamp")

        record["payload"] = payload
        record["payload_valid"] = not errors and payload is not None
        record["payload_errors"] = errors

        if record["payload_valid"]:
            stats["valid_payloads"] += 1
        else:
            stats["invalid_payloads"] += 1

    return statistics


# ---------------------------------------------------------
# FEATURE CALCULATION HELPERS
# ---------------------------------------------------------

def calculate_mean(values):
    return sum(values) / len(values) if values else 0.0


def calculate_population_std(values):
    if len(values) < 2:
        return 0.0

    average = calculate_mean(values)

    return math.sqrt(
        sum((value - average) ** 2 for value in values) / len(values)
    )


def get_numeric_payload_values(payloads, field):
    values = []

    for payload in payloads:
        value = payload.get(field)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            values.append(float(value))

    return values


def epoch_to_utc_iso(epoch_value):
    return datetime.fromtimestamp(
        epoch_value,
        tz=timezone.utc,
    ).isoformat()


# ---------------------------------------------------------
# FEATURE WINDOW GENERATION
# ---------------------------------------------------------

def generate_feature_windows(records, metadata):
    if not records:
        raise ValueError("Cannot generate windows from an empty record list.")

    window_size = metadata["window_size_seconds"]

    first_timestamp = min(record["timestamp"] for record in records)
    last_timestamp = max(record["timestamp"] for record in records)

    total_duration = last_timestamp - first_timestamp

    if total_duration <= 0:
        raise ValueError(
            "PCAP traffic spans zero seconds; feature windows cannot be generated."
        )

    complete_window_count = math.floor(total_duration / window_size)

    if complete_window_count < 1:
        raise ValueError(
            "The capture does not contain one complete feature window."
        )

    windows = []

    for window_index in range(complete_window_count):
        window_start = first_timestamp + window_index * window_size
        window_end = window_start + window_size

        window_records = [
            record for record in records
            if window_start <= record["timestamp"] < window_end
        ]

        if not window_records:
            continue

        publish_records = [
            record for record in window_records
            if record["mqtt_message_type"] == MQTT_PUBLISH_TYPE
        ]

        valid_payload_records = [
            record for record in publish_records
            if record.get("payload_valid", False)
        ]

        payloads = [
            record["payload"]
            for record in valid_payload_records
        ]

        frame_lengths = [
            record["frame_length"]
            for record in window_records
        ]

        timestamps = [
            record["timestamp"]
            for record in window_records
        ]

        interarrival_times = [
            timestamps[index] - timestamps[index - 1]
            for index in range(1, len(timestamps))
        ]

        temperatures = get_numeric_payload_values(payloads, "temperature")
        humidities = get_numeric_payload_values(payloads, "humidity")
        batteries = get_numeric_payload_values(payloads, "battery")

        device_ids = {
            payload["device_id"]
            for payload in payloads
        }

        tcp_streams = {
            record["tcp_stream"]
            for record in window_records
        }

        normal_count = sum(
            payload["traffic_label"] == NORMAL_LABEL
            for payload in payloads
        )

        attack_count = sum(
            payload["traffic_label"] == ATTACK_LABEL
            for payload in payloads
        )

        if normal_count > 0 and attack_count > 0:
            target = "mixed"
        elif attack_count > 0:
            target = "attack"
        elif normal_count > 0:
            target = "normal"
        else:
            target = "unlabeled"

        attack_start = metadata["attack_start_epoch"]
        attack_end = metadata["attack_end_epoch"]

        if attack_start is None or attack_end is None:
            phase_label = "normal_baseline"
        elif window_end <= attack_start:
            phase_label = "pre_attack"
        elif window_start >= attack_end:
            phase_label = "post_attack"
        elif window_start >= attack_start and window_end <= attack_end:
            phase_label = "attack"
        else:
            phase_label = "mixed_boundary"

        duration = window_size

        feature_row = {
            # Identifiers and window metadata (not model predictors)
            "experiment_id": metadata["experiment_id"],
            "run_id": metadata["run_id"],
            "scenario": metadata["scenario"],
            "window_index": window_index,
            "window_start_epoch": window_start,
            "window_end_epoch": window_end,
            "window_start_utc": epoch_to_utc_iso(window_start),
            "window_end_utc": epoch_to_utc_iso(window_end),
            "window_start_relative_seconds": window_index * window_size,
            "window_duration_seconds": duration,

            # Network traffic features
            "packet_count": len(window_records),
            "mqtt_publish_count": len(publish_records),
            "total_bytes": sum(frame_lengths),
            "mean_packet_size_bytes": calculate_mean(frame_lengths),
            "std_packet_size_bytes": calculate_population_std(frame_lengths),
            "min_packet_size_bytes": min(frame_lengths),
            "max_packet_size_bytes": max(frame_lengths),
            "packet_rate_per_second": len(window_records) / duration,
            "publish_rate_per_second": len(publish_records) / duration,
            "mean_interarrival_seconds": calculate_mean(interarrival_times),
            "std_interarrival_seconds": calculate_population_std(
                interarrival_times
            ),
            "min_interarrival_seconds": (
                min(interarrival_times) if interarrival_times else 0.0
            ),
            "max_interarrival_seconds": (
                max(interarrival_times) if interarrival_times else 0.0
            ),
            "unique_tcp_streams": len(tcp_streams),

            # IoT telemetry features
            "telemetry_message_count": len(payloads),
            "mean_temperature": calculate_mean(temperatures),
            "std_temperature": calculate_population_std(temperatures),
            "mean_humidity": calculate_mean(humidities),
            "std_humidity": calculate_population_std(humidities),
            "mean_battery": calculate_mean(batteries),
            "std_battery": calculate_population_std(batteries),
            "unique_devices": len(device_ids),

            # Label counts and targets
            "normal_message_count": normal_count,
            "attack_message_count": attack_count,
            "target": target,
            "phase_label": phase_label,
        }

        windows.append(feature_row)

    incomplete_final_duration = (
        total_duration - complete_window_count * window_size
    )

    summary = {
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "total_observed_duration": total_duration,
        "complete_window_slots": complete_window_count,
        "generated_windows": len(windows),
        "incomplete_final_duration": incomplete_final_duration,
    }

    return windows, summary


# ---------------------------------------------------------
# OUTPUT CSV
# ---------------------------------------------------------

def write_feature_csv(output_path, windows):
    if not windows:
        raise ValueError("No feature windows are available to write.")

    fieldnames = list(windows[0].keys())

    # Exclusive creation prevents accidental overwrite.
    with output_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(windows)


# ---------------------------------------------------------
# EXTRACTION SUMMARY
# ---------------------------------------------------------

def print_extraction_summary(records, publishers, payload_stats):
    print("\nMQTT Publisher Identification:")

    for role, stream in sorted(publishers.items()):
        print(f"{role}: TCP stream {stream}")

    print("\nPublisher Traffic Extraction Summary")

    for role in publishers:
        role_records = [
            record for record in records
            if record["publisher_role"] == role
        ]

        publish_count = sum(
            record["mqtt_message_type"] == MQTT_PUBLISH_TYPE
            for record in role_records
        )

        print(f"\nPublisher: {role}")
        print(f"Captured frames: {len(role_records)}")
        print(f"PUBLISH frames: {publish_count}")

        stats = payload_stats[role]

        print(f"Valid JSON payloads: {stats['valid_payloads']}")
        print(f"Invalid payloads: {stats['invalid_payloads']}")
        print(f"Decode errors: {stats['decode_errors']}")
        print(f"JSON errors: {stats['json_errors']}")
        print(f"Schema errors: {stats['schema_errors']}")
        print(f"Numeric field errors: {stats['numeric_field_errors']}")
        print(f"Timestamp errors: {stats['timestamp_errors']}")
        print(
            "Experiment ID mismatches:",
            stats["experiment_id_mismatches"],
        )
        print(
            "Device ID mismatches:",
            stats["device_id_mismatches"],
        )
        print(
            "Traffic label mismatches:",
            stats["traffic_label_mismatches"],
        )

    print(f"\nTotal extracted frames: {len(records)}")


def print_window_summary(windows, summary):
    print("\nFeature Window Summary")
    print("----------------------")
    print(f"First captured timestamp: {epoch_to_utc_iso(summary['first_timestamp'])}")
    print(f"Last captured timestamp : {epoch_to_utc_iso(summary['last_timestamp'])}")
    print(f"Observed duration       : {summary['total_observed_duration']:.3f} seconds")
    print(f"Window size             : {windows[0]['window_duration_seconds']} seconds")
    print(f"Complete window slots   : {summary['complete_window_slots']}")
    print(f"Generated windows       : {summary['generated_windows']}")
    print(
        "Excluded final duration: "
        f"{summary['incomplete_final_duration']:.3f} seconds"
    )

    target_counts = {}
    phase_counts = {}

    for row in windows:
        target_counts[row["target"]] = target_counts.get(row["target"], 0) + 1
        phase_counts[row["phase_label"]] = (
            phase_counts.get(row["phase_label"], 0) + 1
        )

    print("\nTarget distribution:")
    for label, count in sorted(target_counts.items()):
        print(f"  {label}: {count}")

    print("\nExperimental phase distribution:")
    for label, count in sorted(phase_counts.items()):
        print(f"  {label}: {count}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    pcap, metadata_path, tshark, output = validate_inputs(args)

    metadata = load_metadata(
        metadata_path,
        window_override=args.window_seconds,
    )

    publishers = identify_publishers(
        tshark,
        pcap,
        metadata["scenario"],
    )

    if "simulator" not in publishers:
        raise RuntimeError("Simulator publisher was not identified.")

    if (
        metadata["scenario"] == "message_flooding"
        and "flood_publisher" not in publishers
    ):
        raise RuntimeError("Flood publisher was not identified.")

    print("\nIoT Sentinel AI - PCAP Feature Extractor V2")
    print("--------------------------------------------")
    print(f"Experiment ID : {metadata['experiment_id']}")
    print(f"Run ID        : {metadata['run_id']}")
    print(f"Scenario      : {metadata['scenario']}")
    print(f"Window size   : {metadata['window_size_seconds']} seconds")

    print("\nExtracting publisher-to-broker traffic...")

    records = extract_publisher_frames(
        tshark,
        pcap,
        publishers,
    )

    if not records:
        raise RuntimeError("No publisher traffic was extracted.")

    payload_stats = validate_payloads(records, metadata)

    print_extraction_summary(
        records,
        publishers,
        payload_stats,
    )

    windows, window_summary = generate_feature_windows(
        records,
        metadata,
    )

    if not windows:
        raise RuntimeError("No complete feature windows were generated.")

    print_window_summary(windows, window_summary)

    write_feature_csv(output, windows)

    print("\nFeature extraction completed successfully.")
    print(f"Feature CSV created: {output}")
    print(f"Rows written: {len(windows)}")
    print(f"Columns written: {len(windows[0])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
