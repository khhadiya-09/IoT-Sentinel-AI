
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "baseline_models_v1"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "baseline_v1"
)

FEATURE_SETS = {
    "set_A": "set_A_network_timing_volume",
    "set_B": "set_B_network_plus_packet",
    "set_C": "set_C_extended_hybrid",
}

RANDOM_STATE = 42

# Explicit positive class for binary evaluation.
POSITIVE_CLASS = "message_flooding"


def build_models():
    return {
        "dummy": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                DummyClassifier(strategy="most_frequent"),
            ),
        ]),

        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
    }


def calculate_metrics(y_true, y_pred, y_probability):
    labels = ["normal", POSITIVE_CLASS]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "roc_auc": roc_auc_score(
            (y_true == POSITIVE_CLASS).astype(int),
            y_probability,
        ),
        "pr_auc": average_precision_score(
            (y_true == POSITIVE_CLASS).astype(int),
            y_probability,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return {
        key: float(value)
        if isinstance(value, (np.floating, float))
        else int(value)
        for key, value in metrics.items()
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for feature_name, file_prefix in FEATURE_SETS.items():

        print("\n" + "=" * 65)
        print(f"FEATURE SET: {feature_name}")
        print("=" * 65)

        train_path = DATA_DIR / f"{file_prefix}_train.csv"
        test_path = DATA_DIR / f"{file_prefix}_test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Training or testing file missing for {feature_name}"
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if list(train_df.columns) != list(test_df.columns):
            raise ValueError(
                f"Feature mismatch in {feature_name}"
            )

        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"]

        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        if set(y_train.unique()) != {
            "normal",
            POSITIVE_CLASS,
        }:
            raise ValueError(
                f"Unexpected training classes in {feature_name}"
            )

        if set(y_test.unique()) != {
            "normal",
            POSITIVE_CLASS,
        }:
            raise ValueError(
                f"Unexpected testing classes in {feature_name}"
            )

        feature_results = []

        for model_name, model in build_models().items():

            print(f"\nTraining: {model_name}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            classifier = model.named_steps["classifier"]

            positive_index = list(
                classifier.classes_
            ).index(POSITIVE_CLASS)

            probabilities = model.predict_proba(
                X_test
            )[:, positive_index]

            metrics = calculate_metrics(
                y_test,
                predictions,
                probabilities,
            )

            result = {
                "feature_set": feature_name,
                "model": model_name,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "predictor_count": len(X_train.columns),
                **metrics,
            }

            feature_results.append(result)
            all_results.append(result)

            print(
                f"Accuracy: {metrics['accuracy']:.4f}"
            )
            print(
                f"Precision: {metrics['precision']:.4f}"
            )
            print(
                f"Recall: {metrics['recall']:.4f}"
            )
            print(
                f"F1-score: {metrics['f1_score']:.4f}"
            )
            print(
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )
            print(
                f"PR-AUC: {metrics['pr_auc']:.4f}"
            )
            print(
                f"False Positive Rate: "
                f"{metrics['false_positive_rate']:.4f}"
            )

            report = classification_report(
                y_test,
                predictions,
                labels=["normal", POSITIVE_CLASS],
                output_dict=True,
                zero_division=0,
            )

            report_path = OUTPUT_DIR / (
                f"{feature_name}_{model_name}_classification_report.json"
            )

            with report_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(report, file, indent=4)

            matrix = confusion_matrix(
                y_test,
                predictions,
                labels=["normal", POSITIVE_CLASS],
            )

            matrix_df = pd.DataFrame(
                matrix,
                index=["actual_normal", "actual_flooding"],
                columns=["predicted_normal", "predicted_flooding"],
            )

            matrix_df.to_csv(
                OUTPUT_DIR
                / f"{feature_name}_{model_name}_confusion_matrix.csv"
            )

            model_path = MODEL_DIR / (
                f"{feature_name}_{model_name}.joblib"
            )

            joblib.dump(model, model_path)

        pd.DataFrame(feature_results).to_csv(
            OUTPUT_DIR
            / f"{feature_name}_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_DIR / "all_model_results_v1.csv",
        index=False,
    )

    results_df.to_json(
        OUTPUT_DIR / "all_model_results_v1.json",
        orient="records",
        indent=4,
    )

    print("\n" + "=" * 65)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 65)

    print("\nMODEL COMPARISON:")
    print(
        results_df[
            [
                "feature_set",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
                "false_positive_rate",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nResults directory: {OUTPUT_DIR}")
    print(f"Saved models directory: {MODEL_DIR}")


if __name__ == "__main__":
    main()
