import argparse
import csv
import math
import subprocess

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TSHARK_PATH = Path(
    r"C:\Program Files\Wireshark\tshark.exe"
)

BROKER_PORT = 1883

EXTRACT_FIELDS = [
    "frame.time_epoch",
    "frame.number",
    "frame.len",
    "tcp.stream",
    "tcp.srcport",
    "tcp.dstport",
    "mqtt.msgid",
    "mqtt.topic",
]


def extract_packets(pcap_file):
    """Extract MQTT PUBLISH packets in both communication directions."""

    if not pcap_file.is_file():
        raise FileNotFoundError(
            f"Capture file not found: {pcap_file}"
        )

    if not TSHARK_PATH.is_file():
        raise FileNotFoundError(
            f"TShark executable not found: {TSHARK_PATH}"
        )

    display_filter = (
        f"mqtt.msgtype == 3 && "
        f"(tcp.dstport == {BROKER_PORT} || "
        f"tcp.srcport == {BROKER_PORT})"
    )

    command = [
        str(TSHARK_PATH),
        "-r",
        str(pcap_file),
        "-Y",
        display_filter,
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=f",
    ]

    for field in EXTRACT_FIELDS:
        command.extend(["-e", field])

    print("\nRunning TShark packet extraction...")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )

    lines = result.stdout.splitlines()

    if len(lines) <= 1:
        return []

    headers = lines[0].split("\t")
    packets = []

    for line_number, line in enumerate(lines[1:], start=2):
        values = line.split("\t")

        if len(values) != len(headers):
            print(
                f"Skipping malformed TShark row at line {line_number}."
            )
            continue

        packets.append(dict(zip(headers, values)))

    return packets


def identify_direction(source_port, destination_port):
    """Identify MQTT packet communication direction."""

    if destination_port == BROKER_PORT:
        return "publisher_to_broker"

    if source_port == BROKER_PORT:
        return "broker_to_subscriber"

    return "other"


def build_features(packets):
    """
    Build packet features with two timing measurements.

    inter_arrival_time:
        Time since the previous packet in the same TCP stream and direction.

    global_inter_arrival_time:
        Time since the previous packet in the same traffic direction,
        regardless of TCP stream. For publisher-to-broker packets, this
        measures the global interval between consecutive publisher packets.
    """

    records = []
    previous_stream_timestamps = {}
    previous_direction_timestamps = {}

    packets.sort(
        key=lambda packet: (
            float(packet["frame.time_epoch"]),
            int(packet["frame.number"]),
        )
    )

    for packet in packets:
        try:
            timestamp_epoch = float(
                packet["frame.time_epoch"]
            )

            if not math.isfinite(timestamp_epoch):
                raise ValueError("Invalid packet timestamp.")

            frame_number = int(packet["frame.number"])
            packet_size = int(packet["frame.len"])
            tcp_stream = int(packet["tcp.stream"])
            source_port = int(packet["tcp.srcport"])
            destination_port = int(packet["tcp.dstport"])

            if packet_size <= 0:
                raise ValueError("Invalid frame length.")

            direction = identify_direction(
                source_port,
                destination_port,
            )

            if direction == "other":
                raise ValueError(
                    "Unexpected MQTT communication direction."
                )

            stream_key = (
                direction,
                tcp_stream,
                source_port,
                destination_port,
            )

            previous_stream_timestamp = (
                previous_stream_timestamps.get(stream_key)
            )

            if previous_stream_timestamp is None:
                stream_inter_arrival = None
            else:
                stream_inter_arrival = (
                    timestamp_epoch - previous_stream_timestamp
                )

                if stream_inter_arrival < 0:
                    raise ValueError(
                        "Negative stream inter-arrival time detected."
                    )

            previous_stream_timestamps[stream_key] = timestamp_epoch

            previous_direction_timestamp = (
                previous_direction_timestamps.get(direction)
            )

            if previous_direction_timestamp is None:
                global_inter_arrival = None
            else:
                global_inter_arrival = (
                    timestamp_epoch - previous_direction_timestamp
                )

                if global_inter_arrival < 0:
                    raise ValueError(
                        "Negative global inter-arrival time detected."
                    )

            previous_direction_timestamps[direction] = timestamp_epoch

            timestamp_utc = datetime.fromtimestamp(
                timestamp_epoch,
                tz=timezone.utc,
            ).isoformat()

            record = {
                "timestamp_epoch": timestamp_epoch,
                "timestamp_utc": timestamp_utc,
                "frame_number": frame_number,
                "frame_length_bytes": packet_size,
                "tcp_stream": tcp_stream,
                "source_port": source_port,
                "destination_port": destination_port,
                "mqtt_message_id": packet["mqtt.msgid"].strip(),
                "mqtt_topic": packet["mqtt.topic"].strip(),
                "traffic_direction": direction,
                # Retained for compatibility with the existing pipeline.
                "inter_arrival_time": stream_inter_arrival,
                "global_inter_arrival_time": global_inter_arrival,
            }

            records.append(record)

        except (
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
        ) as error:
            print(
                f"Skipping invalid packet record: {error}"
            )

    return records


def save_features(records, output_file, overwrite=False):
    """Save extracted packet features to CSV."""

    if not records:
        raise ValueError(
            "No valid packet features available for saving."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}\n"
            "Use --overwrite only if you intentionally want to replace it."
        )

    fieldnames = list(records[0].keys())

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nFeature dataset saved: {output_file}")


def print_interval_summary(records, field, title):
    valid_intervals = [
        record[field]
        for record in records
        if record[field] is not None
        and math.isfinite(record[field])
    ]

    print(f"\n{title}")

    if not valid_intervals:
        print("No valid intervals available.")
        return

    print("Valid intervals:", len(valid_intervals))
    print("Minimum:", round(min(valid_intervals), 6), "seconds")
    print("Maximum:", round(max(valid_intervals), 6), "seconds")
    print(
        "Average:",
        round(sum(valid_intervals) / len(valid_intervals), 6),
        "seconds",
    )


def print_summary(records, pcap_file, output_file):
    """Display a summary of extracted MQTT network features."""

    print("\n" + "=" * 70)
    print("       MQTT NETWORK FEATURE EXTRACTION SUMMARY")
    print("=" * 70)

    print("Input PCAP:", pcap_file)
    print("Total extracted PUBLISH packets:", len(records))

    direction_counts = Counter(
        record["traffic_direction"]
        for record in records
    )

    print("\nTraffic direction distribution:")
    for direction, count in sorted(direction_counts.items()):
        print(f"{direction}: {count}")

    publisher_records = [
        record
        for record in records
        if record["traffic_direction"] == "publisher_to_broker"
    ]

    subscriber_records = [
        record
        for record in records
        if record["traffic_direction"] == "broker_to_subscriber"
    ]

    print("\nPacket count verification:")
    print("Publisher-to-broker packets:", len(publisher_records))
    print("Broker-to-subscriber packets:", len(subscriber_records))

    frame_lengths = [
        record["frame_length_bytes"]
        for record in records
    ]

    print("\nCaptured frame length statistics:")
    print("Minimum:", min(frame_lengths), "bytes")
    print("Maximum:", max(frame_lengths), "bytes")
    print(
        "Average:",
        round(sum(frame_lengths) / len(frame_lengths), 2),
        "bytes",
    )

    print_interval_summary(
        publisher_records,
        "inter_arrival_time",
        "Publisher stream-level inter-arrival time:",
    )
    print_interval_summary(
        publisher_records,
        "global_inter_arrival_time",
        "Publisher global inter-arrival time:",
    )
    print_interval_summary(
        subscriber_records,
        "global_inter_arrival_time",
        "Broker-to-subscriber global inter-arrival time:",
    )

    print("\nFeature columns:")
    for field in records[0].keys():
        print("-", field)

    print("\nOutput CSV:", output_file)
    print("\nFeature extraction completed.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="IoT Sentinel AI MQTT feature extractor"
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to the input PCAPNG file",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output feature CSV",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output CSV",
    )

    args = parser.parse_args()

    pcap_file = Path(args.pcap)
    if not pcap_file.is_absolute():
        pcap_file = PROJECT_ROOT / pcap_file

    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file

    print("\nStarting MQTT feature extraction...")
    print("Input PCAP:", pcap_file)
    print("Output CSV:", output_file)

    try:
        packets = extract_packets(pcap_file)

        if not packets:
            raise ValueError(
                "No MQTT PUBLISH packets found in the capture."
            )

        print("Raw PUBLISH packet rows extracted:", len(packets))

        records = build_features(packets)

        if not records:
            raise ValueError(
                "No valid packet features generated."
            )

        save_features(
            records,
            output_file,
            overwrite=args.overwrite,
        )

        print_summary(
            records,
            pcap_file,
            output_file,
        )

    except (
        FileNotFoundError,
        FileExistsError,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
    ) as error:
        print(f"\nFeature extraction failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
