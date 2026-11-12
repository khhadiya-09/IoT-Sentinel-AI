"""
IoT Sentinel AI — Comparative Feature Analysis V1 (10-second windows)

Exploratory comparison of normal baseline windows against pre-attack,
attack, transition, and post-attack windows in the flooding experiment.

Run from the project root:
    python -m src.features.mqtt_feature_comparison_10s

Outputs:
    reports/feature_analysis_10s_v1/phase_feature_summary.csv
    reports/feature_analysis_10s_v1/feature_comparison.json
    reports/feature_analysis_10s_v1/*.png
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NORMAL_RUN = "normal_baseline_v2_20261108T130132Z_0d8534df"
FLOOD_RUN = "message_flooding_v1_20261108T142620Z_71820c58"
WINDOW_FILE = "mqtt_window_features_10s_v2.csv"

PHASE_ORDER = [
    "normal_baseline",
    "pre_attack",
    "attack",
    "transition",
    "post_attack",
]

CANDIDATE_FEATURES = [
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


def safe_float(value: Any):
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def load_windows(project_root: Path) -> pd.DataFrame:
    normal_path = project_root / "data" / "processed" / NORMAL_RUN / WINDOW_FILE
    flood_path = project_root / "data" / "processed" / FLOOD_RUN / WINDOW_FILE

    for path in (normal_path, flood_path):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    normal = pd.read_csv(normal_path, low_memory=False)
    flooding = pd.read_csv(flood_path, low_memory=False)

    normal["analysis_group"] = "normal_baseline"
    flooding["analysis_group"] = flooding["phase_label"].astype(str)

    combined = pd.concat([normal, flooding], ignore_index=True, sort=False)
    required = ["message_count", "phase_label"]
    missing_required = [column for column in required if column not in combined.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from window data: {missing_required}")

    combined["analysis_group"] = pd.Categorical(
        combined["analysis_group"],
        categories=PHASE_ORDER,
        ordered=True,
    )
    return combined


def build_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for group in PHASE_ORDER:
        subset = df[df["analysis_group"] == group]
        for feature in features:
            values = pd.to_numeric(subset[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            rows.append({
                "analysis_group": group,
                "feature": feature,
                "window_count": int(len(subset)),
                "valid_count": int(values.size),
                "missing_count": int(len(subset) - values.size),
                "missing_percent": (
                    round((len(subset) - values.size) / len(subset) * 100, 2)
                    if len(subset) else None
                ),
                "mean": safe_float(values.mean()) if len(values) else None,
                "std": safe_float(values.std()) if len(values) > 1 else None,
                "median": safe_float(values.median()) if len(values) else None,
                "q1": safe_float(values.quantile(0.25)) if len(values) else None,
                "q3": safe_float(values.quantile(0.75)) if len(values) else None,
                "min": safe_float(values.min()) if len(values) else None,
                "max": safe_float(values.max()) if len(values) else None,
            })
    return pd.DataFrame(rows)


def plot_feature(df: pd.DataFrame, feature: str, output_dir: Path):
    groups, labels = [], []
    for group in PHASE_ORDER:
        values = pd.to_numeric(
            df.loc[df["analysis_group"] == group, feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(values):
            groups.append(values.to_numpy())
            labels.append(group.replace("_", "\n"))

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_title(f"{feature.replace('_', ' ').title()} by Traffic Phase (10-second windows)")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_xlabel("Analysis group")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{feature}_by_phase.png", dpi=160)
    plt.close(fig)


def main():
    project_root = Path.cwd()
    output_dir = project_root / "reports" / "feature_analysis_10s_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== IOT SENTINEL AI — COMPARATIVE FEATURE ANALYSIS V1 ===")
    print("Window size: 10 seconds")
    print("Analysis type: exploratory; no model training")

    combined = load_windows(project_root)
    available_features = [
        feature for feature in CANDIDATE_FEATURES if feature in combined.columns
    ]
    unavailable_features = [
        feature for feature in CANDIDATE_FEATURES if feature not in combined.columns
    ]

    counts = combined["analysis_group"].value_counts(sort=False)
    print(f"\nCombined windows: {len(combined)}")
    print("Windows by analysis group:")
    for group in PHASE_ORDER:
        print(f"  {group}: {int(counts.get(group, 0))}")

    if unavailable_features:
        print("\nCandidate columns not present (skipped):")
        for feature in unavailable_features:
            print(f"  {feature}")

    summary = build_summary(combined, available_features)
    summary_path = output_dir / "phase_feature_summary.csv"
    summary.to_csv(summary_path, index=False)

    priority_features = [
        "message_count",
        "message_rate",
        "mean_global_inter_arrival",
        "std_global_inter_arrival",
        "mean_frame_length",
        "mean_temperature",
        "mean_humidity",
        "mean_battery",
    ]

    print("\nSelected feature comparison (mean / median / valid n):")
    for feature in priority_features:
        if feature not in available_features:
            continue
        print(f"\n{feature}:")
        rows = summary[summary["feature"] == feature]
        for _, row in rows.iterrows():
            if row["valid_count"] == 0:
                print(f"  {row['analysis_group']}: no valid observations")
            else:
                print(
                    f"  {row['analysis_group']}: "
                    f"mean={row['mean']:.6g}, "
                    f"median={row['median']:.6g}, "
                    f"valid_n={int(row['valid_count'])}"
                )

    plot_features = [
        "message_count",
        "message_rate",
        "mean_global_inter_arrival",
        "std_global_inter_arrival",
        "mean_frame_length",
        "mean_temperature",
        "mean_humidity",
        "mean_battery",
    ]
    for feature in plot_features:
        if feature in available_features:
            plot_feature(combined, feature, output_dir)

    report = {
        "project": "IoT Sentinel AI",
        "analysis_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": "10 seconds",
        "analysis_type": "exploratory_descriptive_comparison",
        "source_datasets": {
            "normal": str(project_root / "data" / "processed" / NORMAL_RUN / WINDOW_FILE),
            "flooding": str(project_root / "data" / "processed" / FLOOD_RUN / WINDOW_FILE),
        },
        "group_counts": {
            group: int(counts.get(group, 0)) for group in PHASE_ORDER
        },
        "candidate_features_analyzed": available_features,
        "candidate_features_skipped": unavailable_features,
        "predictor_exclusions": [
            "traffic_label",
            "ground_truth_labels",
            "phase_label",
            "experiment_id",
            "normal_message_count",
            "flood_message_count",
            "observed_device_ids",
            "device identifiers",
        ],
        "important_limitations": [
            "This is descriptive analysis, not a model evaluation.",
            "There are only 31 normal and 10 flooding-experiment windows in total.",
            "The flooding experiment has only five 10-second attack windows.",
            "The normal baseline and flooding experiment are separate runs; run-specific effects may be confounded with traffic class.",
            "Frame length may be confounded by the additional sequence_number payload field in flooding messages.",
            "The normal label is simulator-configured and is not independent proof of benign traffic.",
            "Transition windows are retained separately and should not be merged silently.",
            "Do not infer robust detection or unseen-attack generalization from these descriptive comparisons.",
        ],
        "summary_csv": str(summary_path),
        "plots_directory": str(output_dir),
    }

    report_path = output_dir / "feature_comparison.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  Summary CSV: {summary_path}")
    print(f"  JSON report: {report_path}")
    print(f"  Comparison plots: {output_dir}")
    print("\nCaution: The small number of windows limits conclusions.")
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()
