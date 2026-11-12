
import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TSHARK = Path(r"C:\Program Files\Wireshark\tshark.exe")

EXPERIMENTS = {
    "2 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_2rps_run01_20261111T161433Z_13332d0b/message_flooding.pcapng"
    ),
    "5 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_5rps_run01_20261111T164008Z_05155788/message_flooding.pcapng"
    ),
    "10 msg/s": (
        PROJECT_ROOT
        / "data/raw/experiments/flood_v2_10rps_run01_20261111T165033Z_8d2db887/message_flooding.pcapng"
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


def analyse_pcap(experiment_name, pcap_path):
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    command = [
        str(TSHARK),
        "-r", str(pcap_path),
        "-Y", "mqtt && tcp",
        "-T", "fields",
        "-e", "tcp.stream",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "mqtt.msgtype",
        "-e", "mqtt.clientid",
        "-e", "mqtt.msgid",
        "-E", "separator=,",
        "-E", "quote=d",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    streams = defaultdict(lambda: {
        "client_ip": "",
        "client_port": "",
        "broker_ip": "",
        "client_id": "",
        "client_to_broker": Counter(),
        "broker_to_client": Counter(),
        "message_ids": set(),
        "frames": 0,
    })

    reader = csv.reader(result.stdout.splitlines())

    for fields in reader:
        if len(fields) < 8:
            continue

        stream_id, src_ip, dst_ip, src_port, dst_port, msg_type, client_id, msg_id = (
            fields[:8]
        )

        try:
            src_port = int(src_port)
            dst_port = int(dst_port)
            msg_type = int(msg_type)
        except ValueError:
            continue

        stream = streams[stream_id]
        stream["frames"] += 1

        packet_name = PACKET_NAMES.get(
            msg_type, f"UNKNOWN_{msg_type}"
        )

        if dst_port == 1883:
            direction = "client_to_broker"
            stream["client_ip"] = src_ip
            stream["client_port"] = str(src_port)
            stream["broker_ip"] = dst_ip
            stream["client_to_broker"][packet_name] += 1

        elif src_port == 1883:
            direction = "broker_to_client"
            stream["client_ip"] = dst_ip
            stream["client_port"] = str(dst_port)
            stream["broker_ip"] = src_ip
            stream["broker_to_client"][packet_name] += 1

        else:
            continue

        if client_id.strip():
            stream["client_id"] = client_id.strip()

        if msg_id.strip():
            stream["message_ids"].add(msg_id.strip())

    return streams


def main():
    all_rows = []

    print("\n=== MQTT TCP STREAM ANALYSIS ===")

    for experiment_name, pcap_path in EXPERIMENTS.items():
        streams = analyse_pcap(experiment_name, pcap_path)

        print(f"\n{'=' * 55}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"TCP streams identified: {len(streams)}")

        for stream_id, data in sorted(
            streams.items(),
            key=lambda item: int(item[0])
        ):
            print(f"\nTCP Stream: {stream_id}")
            print(f"Client ID: {data['client_id'] or 'Not identified'}")
            print(
                f"Client endpoint: "
                f"{data['client_ip']}:{data['client_port']}"
            )
            print(f"Broker IP: {data['broker_ip']}")
            print(f"Total MQTT frames: {data['frames']}")
            print(
                f"Distinct MQTT message IDs observed: "
                f"{len(data['message_ids'])}"
            )

            print("Client -> Broker:")
            for name, count in sorted(
                data["client_to_broker"].items()
            ):
                print(f"  {name}: {count}")

            print("Broker -> Client:")
            for name, count in sorted(
                data["broker_to_client"].items()
            ):
                print(f"  {name}: {count}")

            for direction in (
                "client_to_broker",
                "broker_to_client",
            ):
                for packet_name, count in data[direction].items():
                    all_rows.append({
                        "experiment": experiment_name,
                        "tcp_stream": stream_id,
                        "client_id": data["client_id"],
                        "client_ip": data["client_ip"],
                        "client_port": data["client_port"],
                        "broker_ip": data["broker_ip"],
                        "direction": direction,
                        "packet_type": packet_name,
                        "count": count,
                        "total_stream_frames": data["frames"],
                        "distinct_message_ids": len(
                            data["message_ids"]
                        ),
                    })

    output_path = (
        PROJECT_ROOT
        / "data/processed/mqtt_tcp_stream_analysis.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment",
        "tcp_stream",
        "client_id",
        "client_ip",
        "client_port",
        "broker_ip",
        "direction",
        "packet_type",
        "count",
        "total_stream_frames",
        "distinct_message_ids",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Total stream records: {len(all_rows)}")
    print(
        "Results saved to: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
