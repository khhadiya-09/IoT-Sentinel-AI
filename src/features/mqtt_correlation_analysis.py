"""
IoT Sentinel AI — Feature Correlation and Redundancy Analysis V1

Analyzes numeric candidate features from the normal and flooding 1-second
window datasets. Excludes label-derived fields and identifiers.

Outputs:
    reports/correlation_analysis_1s_v1/pearson_correlation.csv
    reports/correlation_analysis_1s_v1/spearman_correlation.csv
    reports/correlation_analysis_1s_v1/high_correlation_pairs.csv
    reports/correlation_analysis_1s_v1/correlation_analysis.json
    reports/correlation_analysis_1s_v1/*.png

Run from project root:
    python -m src.features.mqtt_correlation_analysis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NORMAL_RUN = "normal_baseline_v2_20261108T130132Z_0d8534df"
FLOOD_RUN = "message_flooding_v1_20261108T142620Z_71820c58"
WINDOW_FILE = "mqtt_window_features_1s_v2.csv"

EXCLUDED_COLUMNS = {
    "traffic_label",
    "ground_truth_label",
    "ground_truth_labels",
    "phase_label",
    "experiment_id",
    "run_id",
    "device_id",
    "device_type",
    "observed_device_ids",
    "normal_message_count",
    "flood_message_count",
    "window_start",
    "window_end",
    "window_id",
}

CANDIDATE_COLUMNS = [
    "message_count",
    "message_rate",
    "global_inter_arrival_count",
    "mean_global_inter_arrival",
    "std_global_inter_arrival",
    "min_global_inter_arrival",
    "max_global_inter_arrival",
    "stream_inter_arrival_count",
    "mean_inter_arrival",
    "std_inter_arrival",
    "min_inter_arrival",
    "max_inter_arrival",
    "mean_frame_length",
    "std_frame_length",
    "min_frame_length",
    "max_frame_length",
    "mean_temperature",
    "mean_humidity",
    "mean_battery",
    "unique_device_count",
]

CORRELATION_THRESHOLD = 0.85


def load_data(root: Path) -> pd.DataFrame:
    normal_path = root / "data" / "processed" / NORMAL_RUN / WINDOW_FILE
    flood_path = root / "data" / "processed" / FLOOD_RUN / WINDOW_FILE

    for path in (normal_path, flood_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    normal = pd.read_csv(normal_path, low_memory=False)
    flooding = pd.read_csv(flood_path, low_memory=False)

    normal["analysis_group"] = "normal_baseline"
    flooding["analysis_group"] = flooding["phase_label"].astype(str)

    combined = pd.concat([normal, flooding], ignore_index=True, sort=False)
    combined = combined[combined["analysis_group"] != "transition"].copy()

    # Exclude empty windows from correlation calculations because their
    # absent measurements are not observed zero-valued feature measurements.
    combined["message_count"] = pd.to_numeric(
        combined["message_count"], errors="coerce"
    )
    populated = combined[combined["message_count"] > 0].copy()

    return populated


def get_features(df: pd.DataFrame) -> list[str]:
    features = []
    for column in CANDIDATE_COLUMNS:
        if column not in df.columns or column in EXCLUDED_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 3 and values.nunique(dropna=True) > 1:
            features.append(column)
    return features


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str):
    if matrix.empty:
        return
    width = max(10, min(18, len(matrix.columns) * 0.72))
    height = max(8, min(16, len(matrix.index) * 0.62))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Correlation coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def correlation_pairs(matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    columns = list(matrix.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = matrix.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append({
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": float(value),
                    "absolute_correlation": abs(float(value)),
                    "interpretation": (
                        "strong_positive" if value > 0 else "strong_negative"
                    ),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "absolute_correlation", ascending=False
        ).reset_index(drop=True)
    return result


def main():
    root = Path.cwd()
    output_dir = root / "reports" / "correlation_analysis_1s_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== IOT SENTINEL AI — FEATURE CORRELATION ANALYSIS V1 ===")
    print("Window size: 1 second")
    print("Correlation methods: Pearson and Spearman")
    print(f"High-correlation threshold: |r| >= {CORRELATION_THRESHOLD}")
    print("Empty windows and transition windows are excluded from this analysis.")

    df = load_data(root)
    features = get_features(df)

    print(f"\nPopulated windows analyzed: {len(df)}")
    print("Analysis groups:")
    for group, count in df["analysis_group"].value_counts().items():
        print(f"  {group}: {int(count)}")
    print(f"\nCandidate numeric features: {len(features)}")
    for feature in features:
        print(f"  {feature}")

    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson", min_periods=3)
    spearman = numeric.corr(method="spearman", min_periods=3)

    pearson_path = output_dir / "pearson_correlation.csv"
    spearman_path = output_dir / "spearman_correlation.csv"
    pearson.to_csv(pearson_path)
    spearman.to_csv(spearman_path)

    save_heatmap(
        pearson,
        output_dir / "pearson_correlation_heatmap.png",
        "Pearson Feature Correlation — 1-second Windows",
    )
    save_heatmap(
        spearman,
        output_dir / "spearman_correlation_heatmap.png",
        "Spearman Feature Correlation — 1-second Windows",
    )

    pearson_pairs = correlation_pairs(pearson, CORRELATION_THRESHOLD)
    spearman_pairs = correlation_pairs(spearman, CORRELATION_THRESHOLD)

    pairs_path = output_dir / "high_correlation_pairs.csv"
    combined_pairs = []
    for method, pairs in (
        ("pearson", pearson_pairs),
        ("spearman", spearman_pairs),
    ):
        if not pairs.empty:
            item = pairs.copy()
            item.insert(0, "method", method)
            combined_pairs.append(item)

    if combined_pairs:
        pd.concat(combined_pairs, ignore_index=True).to_csv(pairs_path, index=False)
    else:
        pd.DataFrame(columns=[
            "method", "feature_1", "feature_2", "correlation",
            "absolute_correlation", "interpretation"
        ]).to_csv(pairs_path, index=False)

    print("\nStrong Pearson correlations:")
    if pearson_pairs.empty:
        print("  None found at the selected threshold.")
    else:
        for _, row in pearson_pairs.iterrows():
            print(
                f"  {row['feature_1']} <-> {row['feature_2']}: "
                f"{row['correlation']:.4f}"
            )

    print("\nStrong Spearman correlations:")
    if spearman_pairs.empty:
        print("  None found at the selected threshold.")
    else:
        for _, row in spearman_pairs.iterrows():
            print(
                f"  {row['feature_1']} <-> {row['feature_2']}: "
                f"{row['correlation']:.4f}"
            )

    # Also calculate correlations within groups where enough rows exist.
    group_results = {}
    for group in ("normal_baseline", "pre_attack", "attack", "post_attack"):
        subset = df[df["analysis_group"] == group]
        if len(subset) < 5:
            group_results[group] = {
                "rows": int(len(subset)),
                "status": "insufficient_rows_for_reliable_group_correlation",
            }
            continue
        group_numeric = subset[features].apply(pd.to_numeric, errors="coerce")
        group_corr = group_numeric.corr(method="spearman", min_periods=3)
        group_corr.to_csv(output_dir / f"{group}_spearman_correlation.csv")
        group_results[group] = {
            "rows": int(len(subset)),
            "status": "exploratory_correlations_saved",
            "file": f"{group}_spearman_correlation.csv",
        }

    report = {
        "project": "IoT Sentinel AI",
        "analysis_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": "1 second",
        "rows_analyzed": int(len(df)),
        "excluded_empty_windows": int(
            pd.concat([
                pd.read_csv(root / "data" / "processed" / NORMAL_RUN / WINDOW_FILE),
                pd.read_csv(root / "data" / "processed" / FLOOD_RUN / WINDOW_FILE),
            ], ignore_index=True)["message_count"].eq(0).sum()
        ),
        "excluded_transition_windows": 2,
        "candidate_features": features,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "pearson_high_correlation_pairs": pearson_pairs.to_dict(orient="records"),
        "spearman_high_correlation_pairs": spearman_pairs.to_dict(orient="records"),
        "group_specific_analysis": group_results,
        "limitations": [
            "Correlation does not establish causation or predictive value.",
            "Pooled correlations can be driven by differences between normal and attack groups.",
            "Within-group correlations are exploratory and have limited sample sizes.",
            "Pairwise correlations use available observations; missingness can differ by feature pair.",
            "Features derived from the same packet counts or timing values may be mathematically redundant.",
            "This analysis does not select final model features or evaluate detection performance.",
        ],
        "outputs": {
            "pearson_csv": str(pearson_path),
            "spearman_csv": str(spearman_path),
            "high_correlation_pairs_csv": str(pairs_path),
        },
    }

    report_path = output_dir / "correlation_analysis.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  Pearson matrix: {pearson_path}")
    print(f"  Spearman matrix: {spearman_path}")
    print(f"  High-correlation pairs: {pairs_path}")
    print(f"  JSON report: {report_path}")
    print(f"  Heatmaps: {output_dir}")
    print("\nAnalysis completed. Correlation is not proof of predictive value.")


if __name__ == "__main__":
    main()
