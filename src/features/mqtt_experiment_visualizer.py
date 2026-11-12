"""
Generate research figures comparing the IoT Sentinel AI normal baseline
and controlled MQTT message-flooding experiment.

Inputs are existing 1-second window-feature CSVs and experiment metadata.
This script does not modify any source datasets.

Outputs:
    reports/figures/mqtt_message_rate_comparison.png
    reports/figures/mqtt_inter_arrival_comparison.png
    reports/figures/mqtt_frame_length_comparison.png
    reports/figures/mqtt_flooding_timeline.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

NORMAL_DIR = (
    ROOT / "data" / "processed"
    / "normal_baseline_v2_20261108T130132Z_0d8534df"
)
FLOOD_DIR = (
    ROOT / "data" / "processed"
    / "message_flooding_v1_20261108T142620Z_71820c58"
)
FLOOD_METADATA = (
    ROOT / "data" / "raw" / "experiments"
    / "message_flooding_v1_20261108T142620Z_71820c58"
    / "metadata.json"
)
OUTPUT_DIR = ROOT / "reports" / "figures"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature file not found: {path}")
    return pd.read_csv(path)


def load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def add_elapsed_seconds(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["elapsed_seconds"] = (
        pd.to_datetime(result["window_start_utc"], utc=True)
        - pd.to_datetime(result["window_start_utc"], utc=True).iloc[0]
    ).dt.total_seconds()
    return result


def save_message_rate_plot(normal: pd.DataFrame, flood: pd.DataFrame) -> None:
    normal = add_elapsed_seconds(normal)
    flood = add_elapsed_seconds(flood)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(
        normal["elapsed_seconds"],
        normal["message_rate"],
        linewidth=1.2,
    )
    axes[0].set_title("Normal baseline: publisher message rate")
    axes[0].set_xlabel("Elapsed time (seconds)")
    axes[0].set_ylabel("Messages per second")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        flood["elapsed_seconds"],
        flood["message_rate"],
        linewidth=1.2,
    )
    axes[1].set_title("Message-flooding experiment: publisher message rate")
    axes[1].set_xlabel("Elapsed time (seconds)")
    axes[1].set_ylabel("Messages per second")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("MQTT publisher message rate using 1-second windows")
    fig.savefig(OUTPUT_DIR / "mqtt_message_rate_comparison.png", dpi=200)
    plt.close(fig)


def save_inter_arrival_plot(normal: pd.DataFrame, flood: pd.DataFrame) -> None:
    normal_values = normal.loc[
        normal["message_count"] > 0, "mean_inter_arrival"
    ].dropna()
    flood_attack = flood[
        (flood["phase_label"] == "attack")
        & (flood["message_count"] > 0)
    ]["mean_inter_arrival"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.boxplot(
        [normal_values, flood_attack],
        tick_labels=["Normal baseline", "Flooding attack"],
        showmeans=True,
    )
    ax.set_title("Window-level mean inter-arrival time")
    ax.set_ylabel("Mean inter-arrival time (seconds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUTPUT_DIR / "mqtt_inter_arrival_comparison.png", dpi=200)
    plt.close(fig)


def save_frame_length_plot(normal: pd.DataFrame, flood: pd.DataFrame) -> None:
    normal_values = normal.loc[
        normal["message_count"] > 0, "mean_frame_length"
    ].dropna()
    flood_attack = flood[
        (flood["phase_label"] == "attack")
        & (flood["message_count"] > 0)
    ]["mean_frame_length"].dropna()

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.boxplot(
        [normal_values, flood_attack],
        tick_labels=["Normal baseline", "Flooding attack"],
        showmeans=True,
    )
    ax.set_title("Window-level mean captured frame length")
    ax.set_ylabel("Mean frame length (bytes)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUTPUT_DIR / "mqtt_frame_length_comparison.png", dpi=200)
    plt.close(fig)


def save_flooding_timeline(
    flood: pd.DataFrame,
    metadata: dict,
) -> None:
    flood = add_elapsed_seconds(flood)

    experiment_start = pd.Timestamp(metadata["started_at_utc"])
    attack_start = pd.Timestamp(metadata["phase_timing"]["attack_started_at_utc"])
    attack_end = pd.Timestamp(metadata["phase_timing"]["attack_ended_at_utc"])

    attack_start_elapsed = (attack_start - experiment_start).total_seconds()
    attack_end_elapsed = (attack_end - experiment_start).total_seconds()

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(
        flood["elapsed_seconds"],
        flood["normal_message_count"],
        width=0.9,
        label="Normal messages",
    )
    ax.bar(
        flood["elapsed_seconds"],
        flood["flood_message_count"],
        width=0.9,
        bottom=flood["normal_message_count"],
        label="Flooding messages",
    )
    ax.axvline(
        attack_start_elapsed,
        linestyle="--",
        linewidth=1.2,
        label="Attack start",
    )
    ax.axvline(
        attack_end_elapsed,
        linestyle="--",
        linewidth=1.2,
        label="Attack end",
    )
    ax.set_title("MQTT message counts across the flooding experiment")
    ax.set_xlabel("Elapsed time from experiment start (seconds)")
    ax.set_ylabel("Messages per 1-second window")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUTPUT_DIR / "mqtt_flooding_timeline.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal = load_csv(NORMAL_DIR / "mqtt_window_features_1s.csv")
    flood = load_csv(FLOOD_DIR / "mqtt_window_features_1s.csv")
    metadata = load_metadata(FLOOD_METADATA)

    required = [
        "window_start_utc",
        "message_count",
        "message_rate",
        "mean_inter_arrival",
        "mean_frame_length",
        "normal_message_count",
        "flood_message_count",
        "phase_label",
    ]
    for name, data in [("normal", normal), ("flooding", flood)]:
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"{name} window dataset is missing columns: {missing}")

    save_message_rate_plot(normal, flood)
    save_inter_arrival_plot(normal, flood)
    save_frame_length_plot(normal, flood)
    save_flooding_timeline(flood, metadata)

    print("\n=== MQTT EXPERIMENT VISUALIZATION ===")
    print(f"Normal 1-second windows: {len(normal)}")
    print(f"Flooding 1-second windows: {len(flood)}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    for filename in [
        "mqtt_message_rate_comparison.png",
        "mqtt_inter_arrival_comparison.png",
        "mqtt_frame_length_comparison.png",
        "mqtt_flooding_timeline.png",
    ]:
        print(f" - {filename}")
    print("\nNote: inter-arrival values are window aggregates of stream-level intervals.")
    print("Frame length is captured frame length, not MQTT payload length.")


if __name__ == "__main__":
    main()
