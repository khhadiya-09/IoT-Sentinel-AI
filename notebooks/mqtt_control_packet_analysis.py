
import subprocess
import pandas as pd
from pathlib import Path

TSHARK = r"C:\Program Files\Wireshark\tshark.exe"

experiments = {
    "2 msg/s": Path(
        "data/raw/experiments/flood_v2_2rps_run01_20261111T161433Z_13332d0b/message_flooding.pcapng"
    ),
    "5 msg/s": Path(
        "data/raw/experiments/flood_v2_5rps_run01_20261111T164008Z_05155788/message_flooding.pcapng"
    ),
    "10 msg/s": Path(
        "data/raw/experiments/flood_v2_10rps_run01_20261111T165033Z_8d2db887/message_flooding.pcapng"
    ),
}

results = []

print("=== MQTT CONTROL PACKET ANALYSIS ===")

for experiment, pcap in experiments.items():

    if not pcap.exists():
        raise FileNotFoundError(pcap)

    command = [
        TSHARK,
        "-r", str(pcap),
        "-Y", "tcp.dstport == 1883 && mqtt",
        "-T", "fields",
        "-e", "mqtt.msgtype"
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    counts = {}

    for value in completed.stdout.splitlines():
        if not value.strip():
            continue

        packet_type = int(value.strip())
        counts[packet_type] = counts.get(packet_type, 0) + 1

    for packet_type, count in sorted(counts.items()):
        results.append({
            "experiment": experiment,
            "mqtt_packet_type": packet_type,
            "count": count
        })

df = pd.DataFrame(results)

packet_names = {
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
    15: "AUTH"
}

df["packet_name"] = df["mqtt_packet_type"].map(packet_names)

print("\n=== PACKET TYPE COUNTS ===")
print(
    df.pivot_table(
        index="packet_name",
        columns="experiment",
        values="count",
        aggfunc="sum",
        fill_value=0
    ).to_string()
)

output = Path("data/processed/mqtt_control_packet_analysis.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)

print("\nAnalysis saved to:", output)
print("Total MQTT frames analysed:", df["count"].sum())
