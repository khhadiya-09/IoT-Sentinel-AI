import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load master dataset
DATA_PATH = Path("data/processed/mqtt_flooding_features_master.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Research features
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

# 1. PUBLISH rate across windows
plt.figure(figsize=(10, 5))

for experiment_id, group in df.groupby("experiment_id"):
    plt.plot(
        group["window_index"],
        group["publish_rate_per_second"],
        marker="o",
        label=experiment_id
    )

plt.xlabel("Window Index")
plt.ylabel("MQTT PUBLISH Rate (messages/second)")
plt.title("MQTT Publishing Rate Across Experiments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "publish_rate_across_experiments.png", dpi=300)
plt.close()

# 2. Packet count by configured flooding rate
plt.figure(figsize=(8, 5))

df.groupby("configured_attack_rate_per_second")[
    "packet_count"
].mean().plot(kind="bar")

plt.xlabel("Configured Flooding Rate (messages/second)")
plt.ylabel("Average Packet Count per Window")
plt.title("Network Traffic Volume vs Flooding Intensity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "packet_count_vs_flooding_rate.png", dpi=300)
plt.close()

# 3. Inter-arrival time by flooding rate
plt.figure(figsize=(8, 5))

df.groupby("configured_attack_rate_per_second")[
    "mean_interarrival_seconds"
].mean().plot(kind="bar")

plt.xlabel("Configured Flooding Rate (messages/second)")
plt.ylabel("Mean Inter-arrival Time (seconds)")
plt.title("Packet Inter-arrival Time vs Flooding Intensity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "interarrival_vs_flooding_rate.png", dpi=300)
plt.close()

# 4. Feature correlation heatmap
correlation = df[features].corr()

fig, ax = plt.subplots(figsize=(12, 9))

image = ax.imshow(correlation, aspect="auto", vmin=-1, vmax=1)

ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))

ax.set_xticklabels(features, rotation=90)
ax.set_yticklabels(features)

ax.set_title("Network Feature Correlation Matrix")

fig.colorbar(image, ax=ax, label="Pearson Correlation")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "feature_correlation_heatmap.png",
    dpi=300
)

plt.close(fig)

print("EDA completed successfully.")
print("Figures saved to:", OUTPUT_DIR.resolve())

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file.name)
    