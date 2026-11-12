
import pandas as pd
from pathlib import Path

# Load master dataset
DATA_PATH = Path("data/processed/mqtt_flooding_features_master.csv")

df = pd.read_csv(DATA_PATH)

# Select numerical network features
features = [
    "packet_count",
    "mqtt_publish_count",
    "total_bytes",
    "mean_packet_size_bytes",
    "std_packet_size_bytes",
    "packet_rate_per_second",
    "publish_rate_per_second",
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "unique_tcp_streams",
    "unique_source_ports"
]

# Create a separate analysis category
df["traffic_category"] = df["phase"].map({
    "pre_attack": "normal",
    "post_attack": "normal",
    "attack": "pure_attack",
    "boundary_mixed": "boundary_mixed"
})

print("=== TRAFFIC CATEGORY DISTRIBUTION ===")
print(df["traffic_category"].value_counts())

print("\n=== MEAN FEATURE VALUES BY CATEGORY ===")
category_means = (
    df.groupby("traffic_category")[features]
      .mean()
      .round(4)
      .T
)

print(category_means.to_string())

print("\n=== FEATURE CORRELATION ===")
correlation = df[features].corr()

print(correlation.round(3).to_string())

print("\n=== HIGHLY CORRELATED FEATURE PAIRS ===")

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        value = correlation.iloc[i, j]

        if abs(value) >= 0.90:
            print(
                f"{features[i]} <-> {features[j]}: "
                f"{value:.3f}"
            )

print("\nFeature analysis completed.")
