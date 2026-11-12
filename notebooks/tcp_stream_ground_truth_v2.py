
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


def decode_payload_label(message_hex):
    """Extract traffic_label from an MQTT JSON payload."""
    if not message_hex:
        return None

    try:
        cleaned = message_hex.replace(":", "").replace(" ", "")
        raw = bytes.fromhex(cleaned)

        start = raw.find(b"{")
        end = raw.rfind(b"}")

        if start == -1 or end == -1 or end < start:
            return None

        payload = json.loads(
            raw[start:end + 1].decode("utf-8")
        )

        label = payload.get("traffic_label")

        if label in ("normal", "message_flooding"):
            return label

        return None

    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def identify_client_role(client_id):
    if client_id == "iot_sentinel_device_simulator":
        return "normal_simulator"

    if client_id == "iot_sentinel_flood_client_v2":
        return "flood_publisher"

    if client_id.startswith("iot_collector_"):
        return "collector"

    return "unknown"


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
        "-e", "mqtt.clientid",
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
    client_ids = {}

    reader = csv.reader(result.stdout.splitlines())

    for fields in reader:
        if len(fields) < 8:
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

        client_id = fields[7].strip()

        if client_id:
            client_ids[stream_id] = client_id

        if destination_port == 1883:
            direction = "client_to_broker"
        elif source_port == 1883:
            direction = "broker_to_client"
        else:
            continue

        message_hex = fields[6].strip() if fields[6] else ""

        traffic_label = None

        # Count labels only from MQTT PUBLISH packets.
        if packet_type == 3:
            traffic_label = decode_payload_label(message_hex)

        packets.append({
            "timestamp": timestamp,
            "frame_length": frame_length,
            "stream_id": stream_id,
            "direction": direction,
            "packet_type": packet_type,
            "traffic_label": traffic_label,
        })

    return packets, client_ids


def group_packets_into_windows(packets):
    if not packets:
        return {}, 0

    first_timestamp = min(p["timestamp"] for p in packets)
    last_timestamp = max(p["timestamp"] for p in packets)

    complete_windows = int(
        (last_timestamp - first_timestamp) // WINDOW_SIZE
    )

    grouped = defaultdict(list)

    for packet in packets:
        window_index = int(
            (packet["timestamp"] - first_timestamp) // WINDOW_SIZE
        )

        if window_index >= complete_windows:
            continue

        key = (packet["stream_id"], window_index)
        grouped[key].append(packet)

    return grouped, complete_windows


def determine_composition(normal_count, attack_count):
    if normal_count > 0 and attack_count > 0:
        return "mixed"

    if normal_count > 0:
        return "normal"

    if attack_count > 0:
        return "attack"

    return "unknown"


def build_records(experiment_name, grouped, client_ids):
    records = []

    for (stream_id, window_index), packets in sorted(grouped.items()):
        packets.sort(key=lambda p: p["timestamp"])

        client_id = client_ids.get(stream_id, "")
        client_role = identify_client_role(client_id)

        normal_count = sum(
            1 for p in packets
            if p["packet_type"] == 3
            and p["traffic_label"] == "normal"
        )

        attack_count = sum(
            1 for p in packets
            if p["packet_type"] == 3
            and p["traffic_label"] == "message_flooding"
        )

        publish_count = sum(
            1 for p in packets
            if p["packet_type"] == 3
        )

        unclassified_count = (
            publish_count - normal_count - attack_count
        )

        composition = determine_composition(
            normal_count,
            attack_count
        )

        # Preserve the original binary-label logic for comparison.
        if attack_count > 0:
            legacy_label = "attack"
        elif normal_count > 0:
            legacy_label = "normal"
        else:
            legacy_label = "unknown"

        records.append({
            "experiment": experiment_name,
            "tcp_stream": stream_id,
            "client_id": client_id,
            "client_role": client_role,
            "window_index": window_index,
            "window_start_seconds": window_index * WINDOW_SIZE,
            "normal_payload_count": normal_count,
            "attack_payload_count": attack_count,
            "unclassified_publish_count": unclassified_count,
            "total_publish_payloads": publish_count,
            "contains_normal_traffic": normal_count > 0,
            "contains_attack_traffic": attack_count > 0,
            "traffic_composition": composition,
            "legacy_label": legacy_label,
            "total_mqtt_frames": len(packets),
            "mqtt_publish_count": publish_count,
        })

    return records


def main():
    all_records = []

    print("\n=== GROUND-TRUTH LABELING V2 ===")

    for experiment_name, pcap_path in EXPERIMENTS.items():
        print(f"\nProcessing: {experiment_name}")

        packets, client_ids = extract_packets(pcap_path)

        grouped, complete_windows = group_packets_into_windows(
            packets
        )

        records = build_records(
            experiment_name,
            grouped,
            client_ids
        )

        all_records.extend(records)

        print("MQTT frames extracted:", len(packets))
        print("Complete windows:", complete_windows)
        print("Stream-window records:", len(records))

        print(
            "Client roles:",
            dict(Counter(r["client_role"] for r in records))
        )

        print(
            "Traffic compositions:",
            dict(Counter(r["traffic_composition"] for r in records))
        )

        print(
            "Unclassified PUBLISH payloads:",
            sum(r["unclassified_publish_count"] for r in records)
        )

    if not all_records:
        raise RuntimeError("No ground-truth records were generated.")

    output_path = (
        PROJECT_ROOT
        / "data/processed/mqtt_tcp_stream_ground_truth_v2.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\n=== GROUND-TRUTH SUMMARY ===")
    print("Total records:", len(all_records))

    print(
        "Traffic composition:",
        dict(Counter(
            r["traffic_composition"] for r in all_records
        ))
    )

    print(
        "Client roles:",
        dict(Counter(
            r["client_role"] for r in all_records
        ))
    )

    print(
        "Total normal payloads:",
        sum(r["normal_payload_count"] for r in all_records)
    )

    print(
        "Total attack payloads:",
        sum(r["attack_payload_count"] for r in all_records)
    )

    print(
        "Total unclassified PUBLISH payloads:",
        sum(r["unclassified_publish_count"] for r in all_records)
    )

    print(
        "\nSaved to:",
        output_path.relative_to(PROJECT_ROOT)
    )


if __name__ == "__main__":
    main()
