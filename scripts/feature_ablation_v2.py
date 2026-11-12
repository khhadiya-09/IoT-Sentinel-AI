
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/ml_dataset_primary_v2.csv"
OUTPUT = ROOT / "reports/feature_ablation_v2.csv"

FEATURE_GROUPS = {
    "packet_rate_only": [
        "packet_rate_per_second",
    ],
    "interarrival_only": [
        "mean_interarrival_seconds",
        "std_interarrival_seconds",
        "min_interarrival_seconds",
        "max_interarrival_seconds",
    ],
    "packet_size_only": [
        "mean_packet_size_bytes",
        "std_packet_size_bytes",
        "min_packet_size_bytes",
        "max_packet_size_bytes",
    ],
    "all_network_features": [
        "packet_rate_per_second",
        "mean_packet_size_bytes",
        "std_packet_size_bytes",
        "min_packet_size_bytes",
        "max_packet_size_bytes",
        "mean_interarrival_seconds",
        "std_interarrival_seconds",
        "min_interarrival_seconds",
        "max_interarrival_seconds",
    ],
}


def get_rate(run_id):
    for rate in (2, 5, 10):
        if f"_{rate}rps_" in run_id:
            return rate
    return None


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    df = df[
        df["scenario"] == "message_flooding"
    ].copy()

    df["rate"] = df["run_id"].map(get_rate)
    df["y"] = (
        df["binary_label"] == "attack_present"
    ).astype(int)

    if len(df) != 24 or df["rate"].isna().any():
        raise ValueError("Unexpected flooding dataset.")

    results = []

    for group_name, features in FEATURE_GROUPS.items():
        for held_out_rate in (2, 5, 10):
            train = df[df["rate"] != held_out_rate]
            test = df[df["rate"] == held_out_rate]

            model = RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                train[features],
                train["y"],
            )

            predictions = model.predict(test[features])
            actual = test["y"].to_numpy()

            tn, fp, fn, tp = confusion_matrix(
                actual,
                predictions,
                labels=[0, 1],
            ).ravel()

            results.append({
                "feature_group": group_name,
                "held_out_rate": held_out_rate,
                "feature_count": len(features),
                "test_windows": len(test),
                "test_runs": test["run_id"].nunique(),
                "accuracy": accuracy_score(
                    actual, predictions
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    actual, predictions
                ),
                "precision": precision_score(
                    actual, predictions, zero_division=0
                ),
                "recall": recall_score(
                    actual, predictions, zero_division=0
                ),
                "f1": f1_score(
                    actual, predictions, zero_division=0
                ),
                "false_positive_rate": (
                    fp / (fp + tn) if tn + fp else 0
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            })

    result = pd.DataFrame(results)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\n=== FEATURE ABLATION RESULTS ===")
    print(result.round(4).to_string(index=False))

    print("\n=== AGGREGATED RESULTS ===")
    print(
        result.groupby("feature_group")[
            ["accuracy", "balanced_accuracy", "precision",
             "recall", "f1", "false_positive_rate"]
        ].mean().round(4).to_string()
    )

    print(f"\nSaved: {OUTPUT}")
    print("\n[PASS] Feature ablation completed.")


if __name__ == "__main__":
    main()
