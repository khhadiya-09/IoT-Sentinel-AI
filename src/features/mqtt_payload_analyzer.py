
from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TSHARK_PATH = Path(
    r"C:\Program Files\Wireshark\tshark.exe"
)

EXPECTED_TOPIC = "iot-sentinel/devices/telemetry"

REQUIRED_FIELDS = [
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
]

OUTPUT_COLUMNS = [
    "frame_number",
    "timestamp_epoch",
    "tcp_stream",
    "source_port",
    "mqtt_message_id",
    "mqtt_topic",
    *REQUIRED_FIELDS,
]


# --------------------------------------------------
# ARGUMENT PARSING
# --------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract and validate MQTT JSON payloads."
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to the input PCAP/PCAPNG file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV file.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )

    return parser.parse_args()


# --------------------------------------------------
# EXTRACT RAW MQTT PAYLOADS
# --------------------------------------------------

def extract_mqtt_payloads(pcap_file):
    """Extract publisher-to-broker MQTT PUBLISH payloads."""

    if not TSHARK_PATH.exists():
        raise FileNotFoundError(
            f"TShark not found: {TSHARK_PATH}"
        )

    if not pcap_file.exists():
        raise FileNotFoundError(
            f"PCAP file not found: {pcap_file}"
        )

    command = [
        str(TSHARK_PATH),
        "-r",
        str(pcap_file),
        "-Y",
        "mqtt.msgtype == 3 && tcp.dstport == 1883",
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=/t",
        "-e",
        "frame.number",
        "-e",
        "frame.time_epoch",
        "-e",
        "tcp.stream",
        "-e",
        "tcp.srcport",
        "-e",
        "mqtt.msgid",
        "-e",
        "mqtt.topic",
        "-e",
        "mqtt.msg",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    reader = csv.DictReader(
        completed.stdout.splitlines(),
        delimiter="\t",
    )

    return list(reader)


# --------------------------------------------------
# TIMESTAMP VALIDATION
# --------------------------------------------------

def validate_timestamp(value):
    """Validate that a timestamp contains timezone information."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Invalid timestamp")

    try:
        parsed = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            f"Invalid timestamp format: {error}"
        ) from error

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "Timestamp must include timezone information"
        )

    return pd.to_datetime(value, utc=True, errors="raise")


# --------------------------------------------------
# DECODE AND VALIDATE PAYLOADS
# --------------------------------------------------

def decode_payloads(raw_records):
    """Decode hexadecimal MQTT payloads into JSON records."""

    decoded_records = []
    errors = []

    for record in raw_records:
        try:
            raw_hex = record["mqtt.msg"].strip()

            if not raw_hex:
                raise ValueError("Empty MQTT payload")

            # Convert hexadecimal into bytes.
            payload_bytes = bytes.fromhex(raw_hex)

            # Decode bytes into UTF-8 text.
            payload_text = payload_bytes.decode("utf-8")

            # Parse JSON.
            payload = json.loads(payload_text)

            if not isinstance(payload, dict):
                raise ValueError(
                    "MQTT payload is not a JSON object"
                )

            # Check required JSON fields.
            missing = [
                field
                for field in REQUIRED_FIELDS
                if field not in payload
            ]

            if missing:
                raise ValueError(
                    f"Missing JSON fields: {missing}"
                )

            # Validate text fields.
            text_fields = [
                "device_id",
                "device_type",
                "timestamp",
                "status",
                "traffic_label",
                "experiment_id",
            ]

            for field in text_fields:
                if (
                    not isinstance(payload[field], str)
                    or not payload[field].strip()
                ):
                    raise ValueError(
                        f"Invalid text field: {field}"
                    )

            # Validate numeric fields.
            for field in [
                "temperature",
                "humidity",
                "battery",
            ]:
                value = payload[field]

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Invalid numeric field: {field}"
                    )

            # Validate MQTT topic.
            if record["mqtt.topic"] != EXPECTED_TOPIC:
                raise ValueError("Unexpected MQTT topic")

            # Validate timezone-aware timestamp.
            validate_timestamp(payload["timestamp"])

            # Build structured record.
            decoded_record = {
                "frame_number": int(record["frame.number"]),
                "timestamp_epoch": float(
                    record["frame.time_epoch"]
                ),
                "tcp_stream": int(record["tcp.stream"]),
                "source_port": int(record["tcp.srcport"]),
                "mqtt_message_id": (
                    int(record["mqtt.msgid"])
                    if record["mqtt.msgid"].strip()
                    else None
                ),
                "mqtt_topic": record["mqtt.topic"],
            }

            for field in REQUIRED_FIELDS:
                decoded_record[field] = payload[field]

            decoded_records.append(decoded_record)

        except (
            ValueError,
            TypeError,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            OverflowError,
        ) as error:

            errors.append({
                "frame_number": record.get(
                    "frame.number", "unknown"
                ),
                "error": str(error),
            })

    return decoded_records, errors


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

def save_dataset(decoded_records, output_file, overwrite):
    """Save successfully decoded payloads to CSV."""

    if not decoded_records:
        print(
            "\nNo valid payloads were decoded. "
            "No output dataset was created."
        )
        return None

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            "Use --overwrite only if you intentionally "
            "want to replace it."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = pd.DataFrame(
        decoded_records,
        columns=OUTPUT_COLUMNS,
    )

    dataset.to_csv(
        output_file,
        index=False,
    )

    return dataset


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    args = parse_arguments()

    pcap_file = Path(args.pcap).resolve()
    output_file = Path(args.output).resolve()

    print("\nStarting MQTT payload analysis...")
    print("Input PCAP:", pcap_file)
    print("Output CSV:", output_file)

    raw_records = extract_mqtt_payloads(pcap_file)

    print(
        "\nRaw publisher-to-broker MQTT PUBLISH "
        "payloads extracted:",
        len(raw_records),
    )

    decoded_records, errors = decode_payloads(
        raw_records
    )

    dataset = save_dataset(
        decoded_records,
        output_file,
        args.overwrite,
    )

    print("\n" + "=" * 65)
    print("          MQTT PAYLOAD ANALYSIS SUMMARY")
    print("=" * 65)

    print("Extracted payloads:", len(raw_records))
    print("Successfully decoded:", len(decoded_records))
    print("Failed payloads:", len(errors))

    if dataset is not None:
        print("\nPayload dataset saved:")
        print(output_file)

        print("\nDevice distribution:")
        print(dataset["device_id"].value_counts())

        print("\nDevice types:")
        print(dataset["device_type"].value_counts())

        print("\nTraffic labels:")
        print(dataset["traffic_label"].value_counts())

        print("\nExperiment IDs:")
        print(dataset["experiment_id"].value_counts())

        print("\nDataset shape:", dataset.shape)

        print("\nDataset preview:")
        print(dataset.head(10).to_string(index=False))

        print("\nMissing values:")
        print(dataset.isnull().sum())

        print(
            "\nUnique frame numbers:",
            dataset["frame_number"].nunique(),
        )

    if errors:
        print("\nPayload decoding errors:")

        for error in errors:
            print(error)

    print("\nPayload analysis completed.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        FileExistsError,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
