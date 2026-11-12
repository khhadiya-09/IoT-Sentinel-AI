
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT
    / "data/external/MQTTEEB-D A Real-World IoT Cybersecurity Dataset f"
    / "MQTTEEB-D_Final_Dataset/Raw_RealTime_Data"
)

OUTPUT_DIR = ROOT / "reports/mqtteeb/multiclass_v1"
MODEL_DIR = ROOT / "models/mqtteeb"

PREDICTIONS_FILE = OUTPUT_DIR / "oof_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "fold_metrics.csv"
CLASS_METRICS_FILE = OUTPUT_DIR / "class_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix.csv"
SUMMARY_FILE = OUTPUT_DIR / "experiment_summary.json"
MODEL_FILE = MODEL_DIR / "random_forest_mqtteeb_multiclass_v1.joblib"

LOOPS = [
    "MQTTEEB-D_dataset_loop_1.csv",
    "MQTTEEB-D_dataset_loop_2.csv",
    "MQTTEEB-D_dataset_loop_3.csv",
    "MQTTEEB-D_dataset_loop_4.csv",
]

TARGET = "label"
GROUP = "source_loop"

CLASS_NAMES = [
    "legitimate",
    "dos",
    "flood",
    "bruteforce",
    "malformed",
    "slowite",
]

NUMERIC_FEATURES = [
    "tcp_len",
    "mqtt_conflag_cleansess",
    "mqtt_dupflag",
    "mqtt_qos",
]

CATEGORICAL_FEATURES = [
    "tcp_flags",
    "mqtt_conack_flags",
    "mqtt_conflags",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
N_ESTIMATORS = 300


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

def load_dataset():

    frames = []

    for filename in LOOPS:

        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        df = pd.read_csv(
            path,
            low_memory=False,
            dtype={
                "tcp_flags": "string",
                "mqtt_conack_flags": "string",
                "mqtt_conflags": "string",
                "label": "string",
            },
        )

        df[GROUP] = filename.replace(
            "MQTTEEB-D_dataset_", ""
        ).replace(".csv", "")

        frames.append(df)

        print(
            f"Loaded {filename}: {len(df):,} records"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    print("\nOriginal records:", f"{len(data):,}")

    # Missing labels are unknown, not legitimate.
    missing_labels = int(data[TARGET].isna().sum())

    data = data.dropna(
        subset=[TARGET]
    ).copy()

    data[TARGET] = (
        data[TARGET]
        .str.strip()
        .str.lower()
    )

    unexpected = sorted(
        set(data[TARGET].unique()) - set(CLASS_NAMES)
    )

    if unexpected:
        raise ValueError(
            f"Unexpected labels found: {unexpected}"
        )

    print("Missing-label records excluded:", missing_labels)
    print("Usable labelled records:", f"{len(data):,}")

    return data, missing_labels


# ============================================================
# 3. VALIDATE FEATURES
# ============================================================

def validate_dataset(data):

    required = FEATURES + [TARGET, GROUP]

    missing = sorted(
        set(required) - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column in NUMERIC_FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data[required].isna().any().any():
        raise ValueError(
            "Missing values found in selected features."
        )

    numeric_values = data[
        NUMERIC_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite numeric feature values detected."
        )

    if set(data[TARGET].unique()) != set(CLASS_NAMES):
        raise ValueError(
            "Not all six classes are present."
        )

    print("\nClass distribution:")

    print(
        data[TARGET]
        .value_counts()
        .reindex(CLASS_NAMES)
        .to_string()
    )

    print("\nLoop distribution:")

    print(
        data.groupby(
            [GROUP, TARGET],
            observed=True,
        ).size().unstack(fill_value=0)
    )


# ============================================================
# 4. BUILD TRAINING PIPELINE
# ============================================================

def build_pipeline():

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    return pipeline


# ============================================================
# 5. FOUR-FOLD LEAVE-ONE-LOOP-OUT EVALUATION
# ============================================================

def evaluate_model(data):

    predictions = []
    fold_metrics = []
    class_metrics = []

    all_labels = np.array(CLASS_NAMES)

    for fold_number, test_loop in enumerate(
        sorted(data[GROUP].unique()),
        start=1,
    ):

        print("\n" + "=" * 65)
        print(f"FOLD {fold_number}: HOLDING OUT {test_loop}")
        print("=" * 65)

        train_data = data[
            data[GROUP] != test_loop
        ].copy()

        test_data = data[
            data[GROUP] == test_loop
        ].copy()

        X_train = train_data[FEATURES]
        y_train = train_data[TARGET]

        X_test = test_data[FEATURES]
        y_test = test_data[TARGET]

        # Confirm that every class is represented
        # in both training and test partitions.
        missing_train = (
            set(CLASS_NAMES) - set(y_train.unique())
        )

        missing_test = (
            set(CLASS_NAMES) - set(y_test.unique())
        )

        if missing_train or missing_test:
            raise ValueError(
                f"Class coverage failure in {test_loop}. "
                f"Missing training: {missing_train}; "
                f"missing testing: {missing_test}"
            )

        # No source-loop identifier enters the model.
        if set(train_data[GROUP]).intersection(
            set(test_data[GROUP])
        ):
            raise RuntimeError("Loop leakage detected.")

        model = build_pipeline()

        print("Training records:", f"{len(train_data):,}")
        print("Testing records :", f"{len(test_data):,}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "fold": fold_number,
            "held_out_loop": test_loop,
            "training_records": len(train_data),
            "testing_records": len(test_data),
            "accuracy": accuracy_score(
                y_test, y_pred
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, y_pred
            ),
            "macro_precision": precision_score(
                y_test,
                y_pred,
                labels=all_labels,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y_test,
                y_pred,
                labels=all_labels,
                average="macro",
                zero_division=0,
            ),
            "macro_f1": f1_score(
                y_test,
                y_pred,
                labels=all_labels,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_test,
                y_pred,
                labels=all_labels,
                average="weighted",
                zero_division=0,
            ),
        }

        fold_metrics.append(metrics)

        print(
            f"Accuracy          : {metrics['accuracy']:.4f}"
        )
        print(
            f"Balanced accuracy : {metrics['balanced_accuracy']:.4f}"
        )
        print(
            f"Macro F1          : {metrics['macro_f1']:.4f}"
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            output_dict=True,
            zero_division=0,
        )

        for class_name in CLASS_NAMES:

            class_metrics.append({
                "fold": fold_number,
                "held_out_loop": test_loop,
                "class": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

        for index in range(len(test_data)):

            predictions.append({
                "source_loop": test_loop,
                "actual": y_test.iloc[index],
                "predicted": y_pred[index],
                "correct": (
                    y_test.iloc[index] == y_pred[index]
                ),
            })

    return predictions, fold_metrics, class_metrics


# ============================================================
# 6. FINAL OUT-OF-FOLD CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(predictions):

    prediction_df = pd.DataFrame(predictions)

    matrix = confusion_matrix(
        prediction_df["actual"],
        prediction_df["predicted"],
        labels=CLASS_NAMES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"

    return matrix_df


# ============================================================
# 7. SAVE FINAL MODEL AND RESULTS
# ============================================================

def save_outputs(
    data,
    predictions,
    fold_metrics,
    class_metrics,
    missing_labels,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        PREDICTIONS_FILE,
        METRICS_FILE,
        CLASS_METRICS_FILE,
        CONFUSION_FILE,
        SUMMARY_FILE,
        MODEL_FILE,
    ]

    existing = [
        str(path)
        for path in output_files
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n"
            + "\n".join(existing)
        )

    predictions_df = pd.DataFrame(predictions)

    metrics_df = pd.DataFrame(fold_metrics)

    class_metrics_df = pd.DataFrame(class_metrics)

    matrix_df = create_confusion_matrix(predictions)

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    class_metrics_df.to_csv(
        CLASS_METRICS_FILE,
        index=False,
    )

    matrix_df.to_csv(
        CONFUSION_FILE,
    )

    # Train a final model on all labelled records.
    # This model is for later inference, not an
    # independent evaluation result.
    final_model = build_pipeline()

    final_model.fit(
        data[FEATURES],
        data[TARGET],
    )

    artifact = {
        "model": final_model,
        "model_name": "RandomForest_MQTTEEB_D_Multiclass",
        "model_version": "v1",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": CLASS_NAMES,
        "training_records": len(data),
        "training_class_distribution": {
            str(key): int(value)
            for key, value in (
                data[TARGET].value_counts().items()
            )
        },
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    summary = {
        "dataset": "MQTTEEB-D original raw capture loops",
        "total_labelled_records": int(len(data)),
        "missing_label_records_excluded": missing_labels,
        "capture_loops": sorted(
            data[GROUP].unique().tolist()
        ),
        "classes": CLASS_NAMES,
        "features": FEATURES,
        "excluded_features": [
            "timestamp",
            "tcp_time_delta",
            "mqtt_hdrflags",
            "mqtt_kalive",
            "mqtt_msg",
            GROUP,
        ],
        "model": "RandomForestClassifier",
        "evaluation": {
            "method": "Leave-One-Loop-Out",
            "folds": len(fold_metrics),
            "training_loop_overlap": False,
        },
        "fold_metrics": fold_metrics,
        "mean_fold_metrics": {
            key: float(
                metrics_df[key].mean()
            )
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ]
        },
        "limitations": [
            "LOLO evaluates generalization across capture loops, not unseen attack families.",
            "The external benchmark evaluates known attack categories.",
            "The final model is fitted on all labelled records; its training performance is not independent test performance.",
            "The selected features and their extraction quality influence the results.",
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL MULTI-CLASS EVALUATION")
    print("=" * 65)

    print(
        metrics_df[
            [
                "held_out_loop",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nMean fold metrics:")

    for key, value in summary["mean_fold_metrics"].items():
        print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(matrix_df.to_string())

    print("\nSaved outputs:")

    for path in output_files:
        print(path)

    print("\n[PASS] Multi-class evaluation completed.")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MQTTEEB-D MULTI-CLASS MQTT ATTACK DETECTION")
    print("=" * 65)

    data, missing_labels = load_dataset()

    validate_dataset(data)

    predictions, fold_metrics, class_metrics = (
        evaluate_model(data)
    )

    save_outputs(
        data,
        predictions,
        fold_metrics,
        class_metrics,
        missing_labels,
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n[ERROR]", error)
        sys.exit(1)
