
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/ml_dataset_primary_v2.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_FILE = MODEL_DIR / "random_forest_interarrival_v2.joblib"
SUMMARY_FILE = REPORT_DIR / "final_detector_manifest_v2.json"

FEATURES = [
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
]

LABEL_MAP = {
    "attack_free": 0,
    "attack_present": 1,
}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (MODEL_FILE, SUMMARY_FILE):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + ["binary_label", "run_id"]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected.")

    X = df[FEATURES].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite feature values detected.")

    y = df["binary_label"].map(LABEL_MAP)

    if y.isna().any():
        raise ValueError("Unexpected labels detected.")

    y = y.astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("Both classes must be present.")

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    artifact = {
        "model": model,
        "model_name": "RandomForest_interarrival",
        "model_version": "v2",
        "features": FEATURES,
        "label_map": LABEL_MAP,
        "prediction_labels": {
            0: "attack_free",
            1: "attack_present",
        },
        "training_samples": len(df),
        "training_runs": int(df["run_id"].nunique()),
        "training_class_distribution": {
            str(k): int(v)
            for k, v in df["binary_label"].value_counts().items()
        },
        "random_state": 42,
        "n_estimators": 300,
    }

    joblib.dump(artifact, MODEL_FILE)

    manifest = {
        "model_file": str(MODEL_FILE.relative_to(ROOT)),
        "model_name": artifact["model_name"],
        "model_version": artifact["model_version"],
        "features": FEATURES,
        "training_samples": len(df),
        "training_runs": int(df["run_id"].nunique()),
        "training_class_distribution":
            artifact["training_class_distribution"],
        "cross_validation_reference":
            "reports/unseen_flood_rate_evaluation_v2.csv",
        "feature_ablation_reference":
            "reports/feature_ablation_v2.csv",
        "important_limitation": (
            "The final model is fitted on all primary windows. "
            "Its training performance is not independent test "
            "performance. Generalisation to unseen attack families "
            "has not been established."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    print("\n=== FINAL DETECTOR TRAINING ===")
    print(f"Training samples : {len(df)}")
    print(f"Independent runs : {df['run_id'].nunique()}")
    print(f"Features         : {len(FEATURES)}")
    print("\nClass distribution:")
    print(df["binary_label"].value_counts().to_string())

    print("\nSaved model:")
    print(MODEL_FILE)

    print("\nSaved manifest:")
    print(SUMMARY_FILE)

    print("\n[PASS] Final detector saved successfully.")


if __name__ == "__main__":
    main()
