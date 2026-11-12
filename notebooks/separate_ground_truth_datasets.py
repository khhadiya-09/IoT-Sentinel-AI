
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/mqtt_tcp_stream_ground_truth_v2.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed"


def main():
    print("\n=== GROUND-TRUTH DATASET SEPARATION ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Dataset A: Original source-level traffic.
    source_df = df[
        df["client_role"].isin([
            "normal_simulator",
            "flood_publisher"
        ])
    ].copy()

    # Assign labels from the observed payload composition.
    source_df["source_label"] = source_df[
        "traffic_composition"
    ]

    # Only pure normal and pure attack source windows are accepted.
    if not source_df["source_label"].isin(
        ["normal", "attack"]
    ).all():
        raise ValueError(
            "Unexpected mixed or unknown source-level records."
        )

    # Dataset B: Collector observations.
    collector_df = df[
        df["client_role"] == "collector"
    ].copy()

    source_path = (
        OUTPUT_DIR / "mqtt_source_level_ground_truth_v2.csv"
    )

    collector_path = (
        OUTPUT_DIR / "mqtt_collector_observations_v2.csv"
    )

    source_df.to_csv(source_path, index=False)
    collector_df.to_csv(collector_path, index=False)

    print("\n=== SOURCE-LEVEL DATASET ===")
    print("Shape:", source_df.shape)
    print("Source labels:")
    print(source_df["source_label"].value_counts())
    print("\nClient roles:")
    print(source_df["client_role"].value_counts())

    print("\n=== COLLECTOR OBSERVATION DATASET ===")
    print("Shape:", collector_df.shape)
    print("Traffic composition:")
    print(collector_df["traffic_composition"].value_counts())

    print("\n=== INTEGRITY CHECK ===")
    print("Original records:", len(df))
    print(
        "Separated records:",
        len(source_df) + len(collector_df)
    )

    print(
        "All original records accounted for:",
        len(source_df) + len(collector_df) == len(df)
    )

    print("\n=== FILES SAVED ===")
    print(source_path.relative_to(PROJECT_ROOT))
    print(collector_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
