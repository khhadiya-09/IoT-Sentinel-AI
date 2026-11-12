
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TSHARK = Path(r"C:\Program Files\Wireshark\tshark.exe")

EXPERIMENTS = {
    "2 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_2rps_run01_20261111T161433Z_13332d0b"
        / "message_flooding.pcapng"
    ),
    "5 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_5rps_run01_20261111T164008Z_05155788"
        / "message_flooding.pcapng"
    ),
    "10 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_10rps_run01_20261111T165033Z_8d2db887"
        / "message_flooding.pcapng"
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


def analyse_experiment(name, pcap_path):
    if not pcap_path.exists():
        raise FileNotFoundError(
            f"PCAP file not found for {name}: {pcap_path}"
        )

    command = [
        str(TSHARK),
        "-r", str(pcap_path),
        "-Y", "mqtt && tcp",
        "-T", "fields",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "mqtt.msgtype",
        "-E", "separator=,",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    counts = defaultdict(Counter)
    total_frames = 0

    for line in result.stdout.splitlines():
        fields = line.split(",")

        if len(fields) < 3:
            continue

        source_port, destination_port, packet_type = fields[:3]

        if not packet_type.strip():
            continue

        try:
            source_port = int(source_port)
            destination_port = int(destination_port)
            packet_type = int(packet_type)
        except ValueError:
            continue

        if destination_port == 1883:
            direction = "Client_to_Broker"
        elif source_port == 1883:
            direction = "Broker_to_Client"
        else:
            direction = "Other"

        packet_name = PACKET_NAMES.get(
            packet_type,
            f"UNKNOWN_{packet_type}"
        )

        counts[direction][packet_name] += 1
        total_frames += 1

    return counts, total_frames


def main():
    all_rows = []
    grand_total = 0

    print("\n=== MQTT DIRECTION-WISE PACKET ANALYSIS ===")

    for experiment_name, pcap_path in EXPERIMENTS.items():
        counts, total = analyse_experiment(
            experiment_name,
            pcap_path
        )

        grand_total += total

        print(f"\n--- {experiment_name} ---")
        print(f"Total MQTT frames: {total}")

        for direction in (
            "Client_to_Broker",
            "Broker_to_Client",
            "Other",
        ):
            print(f"\n{direction}:")

            if not counts[direction]:
                print("  No packets found.")

            for packet_name, count in sorted(
                counts[direction].items()
            ):
                print(f"  {packet_name}: {count}")

        for direction, packet_counts in counts.items():
            for packet_name, count in packet_counts.items():
                all_rows.append({
                    "experiment": experiment_name,
                    "direction": direction,
                    "packet_type": packet_name,
                    "count": count,
                })

    output_path = (
        PROJECT_ROOT
        / "data/processed/mqtt_direction_analysis.csv"
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
            fieldnames=[
                "experiment",
                "direction",
                "packet_type",
                "count",
            ]
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Total MQTT frames analysed: {grand_total}")
    print(f"Results saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

