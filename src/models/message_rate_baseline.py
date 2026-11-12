
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "temporal_split_v1"
OUTPUT_DIR = ROOT / "reports" / "baseline_models_v1"

TARGET = "target"
FEATURE = "message_rate"
POSITIVE_CLASS = "message_flooding"


def evaluate_model(train_file, test_file):

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df[[FEATURE]]
    y_train = train_df[TARGET]

    X_test = test_df[[FEATURE]]
    y_test = test_df[TARGET]

    y_test_binary = (y_test == POSITIVE_CLASS).astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(POSITIVE_CLASS)

    probabilities = model.predict_proba(X_test)[:, positive_index]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=["normal", POSITIVE_CLASS]
    ).ravel()

    return {
        "model": "Logistic Regression",
        "feature": FEATURE,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test_binary, probabilities
        ),
        "pr_auc": average_precision_score(
            y_test_binary, probabilities
        ),
        "false_positive_rate": fp / (fp + tn),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():

    train_file = DATA_DIR / "set_A_network_timing_volume_train.csv"
    test_file = DATA_DIR / "set_A_network_timing_volume_test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Required files not found in: {DATA_DIR}"
        )

    result = evaluate_model(train_file, test_file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "message_rate_only_results_v1.csv"

    pd.DataFrame([result]).to_csv(output_file, index=False)

    print("\nMESSAGE-RATE-ONLY BASELINE")
    print("=" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
