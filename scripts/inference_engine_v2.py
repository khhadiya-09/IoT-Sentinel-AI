
from pathlib import Path
import argparse
import time

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    ROOT / "models/random_forest_interarrival_v2.joblib"
)

FEATURES = [
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
]

LABELS = {
    0: "attack_free",
    1: "attack_present",
}


class InputValidationError(ValueError):
    """Raised when an inference input is invalid."""


def validate_features(data):
    missing = [name for name in FEATURES if name not in data]

    if missing:
        raise InputValidationError(
            f"Missing required features: {missing}"
        )

    values = {}

    for name in FEATURES:
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise InputValidationError(
                f"{name} must be numeric."
            )

        if not np.isfinite(value):
            raise InputValidationError(
                f"{name} must be finite."
            )

        if value < 0:
            raise InputValidationError(
                f"{name} cannot be negative."
            )

        values[name] = value

    minimum = values["min_interarrival_seconds"]
    maximum = values["max_interarrival_seconds"]
    mean = values["mean_interarrival_seconds"]

    if minimum > maximum:
        raise InputValidationError(
            "Minimum interarrival cannot exceed maximum."
        )

    if not minimum <= mean <= maximum:
        raise InputValidationError(
            "Mean interarrival must be between minimum "
            "and maximum interarrival."
        )

    return pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES,
    )


class IoTSentinelDetector:

    def __init__(self, model_path=DEFAULT_MODEL):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        artifact = joblib.load(model_path)

        required_keys = {"model", "features", "label_map"}

        if not required_keys.issubset(artifact):
            raise ValueError(
                "The model artifact has an invalid structure."
            )

        if artifact["features"] != FEATURES:
            raise ValueError(
                "Model feature schema does not match "
                "the inference engine."
            )

        self.model = artifact["model"]
        self.model_path = model_path

        if set(self.model.classes_) != {0, 1}:
            raise ValueError(
                "The model must support both prediction classes."
            )

    def predict_one(self, data):
        features = validate_features(data)

        start = time.perf_counter()

        prediction = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        probability_map = {
            int(label): float(probability)
            for label, probability
            in zip(self.model.classes_, probabilities)
        }

        return {
            "prediction": LABELS[prediction],
            "predicted_class": prediction,
            "attack_free_probability": probability_map[0],
            "attack_present_probability": probability_map[1],
            "inference_time_ms": elapsed_ms,
        }

    def predict_csv(self, input_path, output_path):
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output_path}"
            )

        df = pd.read_csv(input_path)

        if df.empty:
            raise ValueError("Input CSV contains no rows.")

        missing = [
            feature for feature in FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise InputValidationError(
                f"Missing CSV columns: {missing}"
            )

        results = []

        for index, row in df.iterrows():
            try:
                result = self.predict_one(row.to_dict())
            except (InputValidationError, ValueError) as exc:
                raise InputValidationError(
                    f"Invalid input at CSV row {index + 2}: {exc}"
                ) from exc

            results.append(result)

        result_df = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(results)],
            axis=1,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_df.to_csv(output_path, index=False)

        print("\n=== INFERENCE RESULTS ===")
        print(f"Input rows       : {len(df)}")
        print(f"Predicted normal : "
              f"{(result_df['predicted_class'] == 0).sum()}")
        print(f"Predicted attack : "
              f"{(result_df['predicted_class'] == 1).sum()}")
        print(
            "Mean inference time: "
            f"{result_df['inference_time_ms'].mean():.4f} ms"
        )
        print(f"\nOutput saved: {output_path}")

        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="IoT Sentinel AI inference engine"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing the four required features.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for predictions.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained model artifact.",
    )

    args = parser.parse_args()

    detector = IoTSentinelDetector(args.model)

    print("IoT Sentinel AI detector loaded.")
    print(f"Model: {args.model}")

    detector.predict_csv(
        args.input,
        args.output,
    )

    print("\n[PASS] Inference completed successfully.")


if __name__ == "__main__":
    main()
