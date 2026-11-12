"""
IoT Sentinel AI — Missing-Value Diagnostics V2

Prints exact missing-value counts and percentages for all four V2 window
datasets, along with examples of affected rows and relevant window counts.

Run from the project root:
    python -m src.data_processing.window_missingness_diagnostic
"""

from pathlib import Path
import pandas as pd

EXPERIMENTS = {
    "NORMAL": "normal_baseline_v2_20261108T130132Z_0d8534df",
    "FLOODING": "message_flooding_v1_20261108T142620Z_71820c58",
}

FILES = {
    "1-second": "mqtt_window_features_1s_v2.csv",
    "10-second": "mqtt_window_features_10s_v2.csv",
}


def main():
    root = Path("data/processed")
    print("\n=== IOT SENTINEL AI — MISSING-VALUE DIAGNOSTIC V2 ===")

    for experiment, folder in EXPERIMENTS.items():
        for window_label, filename in FILES.items():
            path = root / folder / filename
            print("\n" + "=" * 78)
            print(f"{experiment} | {window_label} WINDOWS")
            print("=" * 78)

            if not path.exists():
                print(f"ERROR: File not found: {path}")
                continue

            df = pd.read_csv(path, low_memory=False)
            print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"Completely empty rows: {int(df.isna().all(axis=1).sum())}")

            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            if missing.empty:
                print("No missing values found.")
                continue

            print("\nMissing-value summary:")
            summary = pd.DataFrame({
                "missing_count": missing.astype(int),
                "missing_percent": (missing / len(df) * 100).round(2),
                "non_missing_count": (len(df) - missing).astype(int),
            })
            print(summary.to_string())

            print("\nMissing values by phase:")
            if "phase_label" in df.columns:
                for column in missing.index:
                    phase_counts = (
                        df.assign(_missing=df[column].isna())
                        .groupby("phase_label", dropna=False)["_missing"]
                        .agg(["sum", "count"])
                    )
                    phase_counts["missing_percent"] = (
                        phase_counts["sum"] / phase_counts["count"] * 100
                    ).round(2)
                    print(f"\n{column}:")
                    print(phase_counts.to_string())

            print("\nExamples of rows containing missing values (up to 5):")
            display_columns = [
                column for column in [
                    "window_start", "window_end", "phase_label",
                    "message_count", "global_inter_arrival_count",
                    "mean_global_inter_arrival", "std_global_inter_arrival",
                    "inter_arrival_count", "mean_inter_arrival",
                    "std_inter_arrival",
                ] if column in df.columns
            ]
            affected = df.loc[df[missing.index].isna().any(axis=1), display_columns]
            print(affected.head(5).to_string(index=False))

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("No dataset values were modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
