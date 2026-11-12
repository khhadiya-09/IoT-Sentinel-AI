
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TSHARK = Path(r"C:\Program Files\Wireshark\tshark.exe")
WINDOW_SIZE = 10

EXPERIMENTS = {
    "2 msg/s": (
        PROJECT_ROOT / "data/raw/experiments/"
        "flood_v2_2rps_run01_20261111T161433Z_13332d0b/"
        "message_flooding.pcapng"
    ),
    "5 msg/s": (
        PROJECT_ROOT / "data/raw/experiments/"
        "flood_v2_5rps_run01_20261111T164008Z_05155788/"
        "message_flooding.pcapng"
    ),
    "10 msg/s": (
        PROJECT_ROOT / "data/raw/experiments/"
        "flood_v2_10rps_run01_20261111T165033Z_8d2db887/"
        "message_flooding.pcapng"
    ),
}

PACKET_NAMES = {
    1: "CONNECT",
    2: "CONNACK",
    3: "PUBLISH",
    4: "PUBACK",
    5: "PUBREC",
    6: "PUBREL",
    7: "PUBCOMP",
    8: "SUBSCRIBE",
    9: "SUBACK",
    10: "UNSUBSCRIBE",
    11: "UNSUBACK",
    12: "PINGREQ",
    13: "PINGRESP",
    14: "DISCONNECT",
    15: "AUTH",
}


def decode_traffic_label(message_hex):
    """Extract traffic_label from a JSON MQTT message payload."""
    if not message_hex:
        return None

    try:
        cleaned = message_hex.replace(":", "").replace(" ", "")
        raw = bytes.fromhex(cleaned)

        start = raw.find(b"{")
        end = raw.rfind(b"}")

        if start == -1 or end == -1 or end < start:
            return None

        payload = json.loads(raw[start:end + 1].decode("utf-8"))
        return payload.get("traffic_label")

    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def extract_packets(pcap_path):
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    command = [
        str(TSHARK),
        "-r", str(pcap_path),
        "-Y", "mqtt && tcp",
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-e", "tcp.stream",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "mqtt.msgtype",
        "-e", "mqtt.msg",
        "-E", "separator=,",
        "-E", "quote=d",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    packets = []
    reader = csv.reader(result.stdout.splitlines())

    for fields in reader:
        if len(fields) < 7:
            continue

        try:
            timestamp = float(fields[0])
            frame_length = int(fields[1])
            stream_id = int(fields[2])
            source_port = int(fields[3])
            destination_port = int(fields[4])
            packet_type = int(fields[5])
        except (ValueError, IndexError):
            continue

        if destination_port == 1883:
            direction = "client_to_broker"
        elif source_port == 1883:
            direction = "broker_to_client"
        else:
            continue

        message_hex = fields[6].strip() if fields[6] else ""
        traffic_label = decode_traffic_label(message_hex)

        packets.append({
            "timestamp": timestamp,
            "frame_length": frame_length,
            "stream_id": stream_id,
            "direction": direction,
            "packet_type": packet_type,
            "traffic_label": traffic_label,
        })

    return packets


def create_windows(packets):
    if not packets:
        return [], 0

    first_timestamp = min(p["timestamp"] for p in packets)
    last_timestamp = max(p["timestamp"] for p in packets)

    complete_window_count = int(
        (last_timestamp - first_timestamp) // WINDOW_SIZE
    )

    grouped = defaultdict(list)

    for packet in packets:
        window_index = int(
            (packet["timestamp"] - first_timestamp) // WINDOW_SIZE
        )

        # Discard incomplete trailing windows consistently.
        if window_index >= complete_window_count:
            continue

        key = (packet["stream_id"], window_index)
        grouped[key].append(packet)

    return grouped, complete_window_count


def calculate_features(experiment_name, grouped):
    rows = []

    for (stream_id, window_index), packets in sorted(grouped.items()):
        packets.sort(key=lambda p: p["timestamp"])

        timestamps = [p["timestamp"] for p in packets]
        sizes = [p["frame_length"] for p in packets]

        packet_count = len(packets)
        total_bytes = sum(sizes)

        interarrival = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
        ]

        packet_types = Counter(
            p["packet_type"] for p in packets
        )

        client_packets = [
            p for p in packets
            if p["direction"] == "client_to_broker"
        ]

        broker_packets = [
            p for p in packets
            if p["direction"] == "broker_to_client"
        ]

        labels = {
            p["traffic_label"]
            for p in packets
            if p["traffic_label"]
        }

        if "message_flooding" in labels:
            label = "attack"
        elif "normal" in labels:
            label = "normal"
        else:
            label = "unknown"

        row = {
            "experiment": experiment_name,
            "tcp_stream": stream_id,
            "window_index": window_index,
            "window_start_seconds": window_index * WINDOW_SIZE,
            "window_duration_seconds": WINDOW_SIZE,
            "packet_count": packet_count,
            "total_bytes": total_bytes,
            "mean_packet_size_bytes": sum(sizes) / packet_count,
            "std_packet_size_bytes": (
                sum((x - sum(sizes) / packet_count) ** 2 for x in sizes)
                / packet_count
            ) ** 0.5,
            "min_packet_size_bytes": min(sizes),
            "max_packet_size_bytes": max(sizes),
            "packet_rate_per_second": packet_count / WINDOW_SIZE,
            "mean_interarrival_seconds": (
                sum(interarrival) / len(interarrival)
                if interarrival else 0
            ),
            "std_interarrival_seconds": (
                (
                    sum(
                        (x - sum(interarrival) / len(interarrival)) ** 2
                        for x in interarrival
                    ) / len(interarrival)
                ) ** 0.5
                if interarrival else 0
            ),
            "min_interarrival_seconds": (
                min(interarrival) if interarrival else 0
            ),
            "max_interarrival_seconds": (
                max(interarrival) if interarrival else 0
            ),
            "client_to_broker_count": len(client_packets),
            "broker_to_client_count": len(broker_packets),
            "client_to_broker_bytes": sum(
                p["frame_length"] for p in client_packets
            ),
            "broker_to_client_bytes": sum(
                p["frame_length"] for p in broker_packets
            ),
            "mqtt_publish_count": packet_types[3],
            "mqtt_connect_count": packet_types[1],
            "mqtt_connack_count": packet_types[2],
            "mqtt_puback_count": packet_types[4],
            "mqtt_subscribe_count": packet_types[8],
            "mqtt_suback_count": packet_types[9],
            "mqtt_unsubscribe_count": packet_types[10],
            "mqtt_disconnect_count": packet_types[14],
            "label": label,
        }

        rows.append(row)

    return rows


def main():
    all_rows = []

    print("\n=== TCP STREAM-LEVEL FEATURE EXTRACTION ===")

    for experiment_name, pcap_path in EXPERIMENTS.items():
        print(f"\nProcessing: {experiment_name}")

        packets = extract_packets(pcap_path)
        grouped, complete_windows = create_windows(packets)

        rows = calculate_features(experiment_name, grouped)
        all_rows.extend(rows)

        print(f"MQTT frames extracted: {len(packets)}")
        print(f"Complete windows: {complete_windows}")
        print(f"Stream-window records: {len(rows)}")

        print("Labels:")
        print(dict(Counter(row["label"] for row in rows)))

    output_path = (
        PROJECT_ROOT
        / "data/processed/mqtt_tcp_stream_features.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not all_rows:
        raise RuntimeError("No stream-level feature records were generated.")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Total stream-window records: {len(all_rows)}")
    print(f"Total features: {len(all_rows[0]) - 4}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
