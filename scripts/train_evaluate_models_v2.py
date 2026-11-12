
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/ml_dataset_primary_v2.csv"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

RESULTS_FILE = REPORT_DIR / "model_evaluation_v2.csv"
PREDICTIONS_FILE = REPORT_DIR / "model_oof_predictions_v2.csv"
SUMMARY_FILE = REPORT_DIR / "model_evaluation_summary_v2.json"

MODEL_FILES = [
    MODEL_DIR / "random_forest_network_v2.joblib",
    MODEL_DIR / "random_forest_combined_v2.joblib",
    MODEL_DIR / "isolation_forest_network_v2.joblib",
]

NETWORK_FEATURES = [
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

TELEMETRY_FEATURES = [
    "mean_temperature",
    "std_temperature",
    "mean_humidity",
    "std_humidity",
    "mean_battery",
    "std_battery",
]

LABEL_MAP = {
    "attack_free": 0,
    "attack_present": 1,
}

N_SPLITS = 3
RANDOM_STATE = 42


def calculate_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_score)
        result["average_precision"] = average_precision_score(
            y_true, y_score
        )
    else:
        result["roc_auc"] = None
        result["average_precision"] = None

    return result


def check_outputs():
    paths = [
        RESULTS_FILE,
        PREDICTIONS_FILE,
        SUMMARY_FILE,
        *MODEL_FILES,
    ]

    existing = [str(path) for path in paths if path.exists()]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(existing)
        )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    check_outputs()

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required = (
        NETWORK_FEATURES
        + TELEMETRY_FEATURES
        + ["binary_label", "run_id"]
    )

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    X_network = df[NETWORK_FEATURES].astype(float)
    X_combined = df[
        NETWORK_FEATURES + TELEMETRY_FEATURES
    ].astype(float)

    if not np.isfinite(X_network.to_numpy()).all():
        raise ValueError("Invalid network feature values.")

    if not np.isfinite(X_combined.to_numpy()).all():
        raise ValueError("Invalid combined feature values.")

    if not df["binary_label"].isin(LABEL_MAP).all():
        raise ValueError("Unexpected target label.")

    y = df["binary_label"].map(LABEL_MAP).to_numpy()
    groups = df["run_id"].astype(str).to_numpy()

    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError("Not enough independent runs.")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Materialise splits once so every model uses identical folds.
    splits = list(cv.split(X_network, y, groups))

    for fold_number, (train_idx, test_idx) in enumerate(
        splits, start=1
    ):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Run leakage detected in fold {fold_number}."
            )

        if len(np.unique(y[train_idx])) != 2:
            raise RuntimeError(
                f"Training fold {fold_number} lacks a class."
            )

        if len(np.unique(y[test_idx])) != 2:
            raise RuntimeError(
                f"Test fold {fold_number} lacks a class. "
                "Grouped stratification needs adjustment."
            )

    model_specs = [
        ("Dummy_baseline", "network"),
        ("RandomForest_network", "network"),
        ("RandomForest_combined", "combined"),
        ("IsolationForest_network", "network"),
    ]

    prediction_records = []

    for model_name, feature_group in model_specs:
        print(f"\nTraining: {model_name}")

        X = (
            X_combined
            if feature_group == "combined"
            else X_network
        )

        for fold_number, (train_idx, test_idx) in enumerate(
            splits, start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            if model_name == "Dummy_baseline":
                model = DummyClassifier(
                    strategy="most_frequent"
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                scores = None

            elif model_name.startswith("RandomForest"):
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                scores = model.predict_proba(X_test)[:, 1]

            else:
                # Unsupervised training uses attack-free rows only.
                attack_free_mask = y_train == 0
                X_train_normal = X_train.iloc[
                    np.flatnonzero(attack_free_mask)
                ]

                if len(X_train_normal) < 5:
                    raise RuntimeError(
                        "Insufficient attack-free training windows "
                        f"in fold {fold_number}."
                    )

                model = IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X_train_normal)

                raw_predictions = model.predict(X_test)
                predictions = np.where(
                    raw_predictions == -1, 1, 0
                )

                # Larger scores indicate more anomalous behaviour.
                scores = -model.score_samples(X_test)

            fold_metrics = calculate_metrics(
                y_test,
                predictions,
                scores,
            )

            for local_index, global_index in enumerate(test_idx):
                prediction_records.append({
                    "model": model_name,
                    "feature_group": feature_group,
                    "fold": fold_number,
                    "run_id": groups[global_index],
                    "actual": int(y[global_index]),
                    "predicted": int(predictions[local_index]),
                    "score": (
                        float(scores[local_index])
                        if scores is not None else None
                    ),
                })

            print(
                f"  Fold {fold_number}: "
                f"F1={fold_metrics['f1']:.3f}, "
                f"Recall={fold_metrics['recall']:.3f}, "
                f"FPR={fold_metrics['false_positive_rate']:.3f}"
            )

    predictions_df = pd.DataFrame(prediction_records)

    # Aggregate out-of-fold predictions across all independent folds.
    results = []

    for model_name, feature_group in model_specs:
        subset = predictions_df[
            predictions_df["model"] == model_name
        ]

        y_true = subset["actual"].to_numpy()
        y_pred = subset["predicted"].to_numpy()

        score_values = subset["score"].to_numpy(dtype=float)
        scores = (
            score_values
            if np.isfinite(score_values).all()
            else None
        )

        metrics = calculate_metrics(
            y_true,
            y_pred,
            scores,
        )

        results.append({
            "model": model_name,
            "feature_group": feature_group,
            "folds": N_SPLITS,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    # Fit final models on the complete primary dataset for later inference.
    final_network_model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_network_model.fit(X_network, y)

    final_combined_model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_combined_model.fit(X_combined, y)

    final_if = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_if.fit(X_network.iloc[np.flatnonzero(y == 0)])

    joblib.dump(
        {
            "model": final_network_model,
            "features": NETWORK_FEATURES,
            "label_map": LABEL_MAP,
            "model_name": "RandomForest_network",
        },
        MODEL_FILES[0],
    )

    joblib.dump(
        {
            "model": final_combined_model,
            "features": NETWORK_FEATURES + TELEMETRY_FEATURES,
            "label_map": LABEL_MAP,
            "model_name": "RandomForest_combined",
        },
        MODEL_FILES[1],
    )

    joblib.dump(
        {
            "model": final_if,
            "features": NETWORK_FEATURES,
            "label_map": LABEL_MAP,
            "model_name": "IsolationForest_network",
            "training_data": "attack_free_only",
        },
        MODEL_FILES[2],
    )

    results_df.to_csv(RESULTS_FILE, index=False)
    predictions_df.to_csv(PREDICTIONS_FILE, index=False)

    summary = {
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "samples": int(len(df)),
        "independent_runs": int(df["run_id"].nunique()),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "cross_validation": {
            "method": "StratifiedGroupKFold",
            "splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "group_column": "run_id",
            "run_overlap_between_train_and_test": False,
        },
        "feature_groups": {
            "network": NETWORK_FEATURES,
            "combined": NETWORK_FEATURES + TELEMETRY_FEATURES,
        },
        "metrics": results,
        "limitations": [
            "Only MQTT message flooding is represented.",
            "Only three configured flooding rates are represented.",
            "The dataset contains a limited number of independent runs.",
            "Cross-validation results do not establish generalisation "
            "to unseen attack families.",
            "Final models are fitted on all primary data; their "
            "training performance is not independent test performance.",
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\n=== CROSS-VALIDATED MODEL RESULTS ===")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "roc_auc",
                "average_precision",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== SAVED OUTPUTS ===")
    for path in [
        RESULTS_FILE,
        PREDICTIONS_FILE,
        SUMMARY_FILE,
        *MODEL_FILES,
    ]:
        print(path)

    print("\n[PASS] Model training and evaluation completed.")


if __name__ == "__main__":
    main()
