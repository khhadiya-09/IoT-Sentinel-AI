
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
OUTPUT = ROOT / "reports/unseen_flood_rate_evaluation_v2.csv"

FEATURES = [
    "packet_rate_per_second",
    "mean_packet_size_bytes",
    "std_packet_size_bytes",
    "min_packet_size_bytes",
    "max_packet_size_bytes",
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
]

RATES = [2, 5, 10]


def get_rate(run_id):
    for rate in RATES:
        if f"_{rate}rps_" in run_id:
            return rate
    return None


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT}"
        )

    df = pd.read_csv(INPUT)

    flooding = df[
        df["scenario"] == "message_flooding"
    ].copy()

    flooding["rate"] = flooding["run_id"].apply(get_rate)

    if flooding["rate"].isna().any():
        raise ValueError("Unknown flooding rate detected.")

    if len(flooding) != 24:
        raise ValueError(
            f"Expected 24 primary flooding windows, got {len(flooding)}"
        )

    flooding["y"] = (
        flooding["binary_label"] == "attack_present"
    ).astype(int)

    results = []

    for held_out_rate in RATES:
        train = flooding[
            flooding["rate"] != held_out_rate
        ]

        test = flooding[
            flooding["rate"] == held_out_rate
        ]

        X_train = train[FEATURES].astype(float)
        X_test = test[FEATURES].astype(float)

        y_train = train["y"].to_numpy()
        y_test = test["y"].to_numpy()

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Training classes missing for rate {held_out_rate}"
            )

        if len(np.unique(y_test)) != 2:
            raise ValueError(
                f"Test classes missing for rate {held_out_rate}"
            )

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "held_out_rate": held_out_rate,
            "training_rates": ",".join(
                str(r) for r in RATES
                if r != held_out_rate
            ),
            "training_windows": len(train),
            "test_windows": len(test),
            "test_runs": test["run_id"].nunique(),
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "false_positive_rate": (
                fp / (fp + tn) if (fp + tn) else 0.0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    result_df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT, index=False)

    print("\n=== UNSEEN FLOOD-RATE EVALUATION ===")
    print(result_df.round(4).to_string(index=False))

    print("\n=== INTERPRETATION ===")
    print(
        "Each test uses both independent runs at the withheld rate."
    )
    print(
        "This evaluates unseen flooding intensity, "
        "not unseen attack families."
    )

    print(f"\nSaved: {OUTPUT}")
    print("\n[PASS] Unseen-rate evaluation completed.")


if __name__ == "__main__":
    main()
