
import pandas as pd
from pathlib import Path

path = Path("data/processed/mqtt_flooding_features_master.csv")
df = pd.read_csv(path)

pairs = [
    ("packet_count", "mqtt_publish_count"),
    ("packet_count", "total_bytes"),
    ("packet_rate_per_second", "publish_rate_per_second"),
    ("unique_tcp_streams", "unique_source_ports"),
]

print("=== EXACT FEATURE DEPENDENCY CHECK ===")

for first, second in pairs:
    difference = df[first] - df[second]

    print(f"\n{first} vs {second}")
    print("Correlation:", round(df[first].corr(df[second]), 6))
    print("Identical values:", (difference == 0).all())
    print("Maximum absolute difference:", abs(difference).max())

print("\n=== PACKET COUNT VS PUBLISH COUNT ===")

df["non_publish_mqtt_frames"] = (
    df["packet_count"] - df["mqtt_publish_count"]
)

print(
    df[[
        "experiment_id",
        "window_index",
        "packet_count",
        "mqtt_publish_count",
        "non_publish_mqtt_frames"
    ]].to_string(index=False)
)

print("\n=== DEPENDENCY CHECK COMPLETED ===")
