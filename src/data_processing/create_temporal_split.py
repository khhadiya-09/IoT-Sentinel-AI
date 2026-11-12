
from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_baseline_v1"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection_v1"
    / "window_metadata_and_targets.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split_v1"
)

FEATURE_FILES = [
    "set_A_network_timing_volume_clean.csv",
    "set_B_network_plus_packet_clean.csv",
    "set_C_extended_hybrid_clean.csv",
]

TRAIN_FRACTION = 0.70

# Five actual 1-second window indices before each test boundary.
EMBARGO_WINDOW_INDICES = 5

EXPECTED_SOURCES = {
    "normal_baseline",
    "message_flooding",
}

EXPECTED_TARGETS = {
    "normal",
    "message_flooding",
}


def class_distribution(series):
    return {
        str(label): int(count)
        for label, count in series.value_counts().items()
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH).reset_index(drop=True)

    required_columns = {
        "source_dataset",
        "window_index",
        "window_start_utc",
        "phase_label",
        "target",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_columns)}"
        )

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError("Required metadata contains missing values.")

    if set(metadata["source_dataset"].unique()) != EXPECTED_SOURCES:
        raise ValueError("Unexpected source experiment distribution.")

    if not metadata["window_index"].map(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Window indices must be integers.")

    metadata["window_index"] = metadata["window_index"].astype(int)

    # Each retained observation must represent one unique source window.
    if metadata.duplicated(
        subset=["source_dataset", "window_index"]
    ).any():
        raise ValueError("Duplicate source/window identities found.")

    train_positions = []
    test_positions = []
    embargo_positions = []

    source_audit = {}

    # Create chronological partitions independently for each experiment.
    for source in sorted(EXPECTED_SOURCES):
        source_rows = metadata[
            metadata["source_dataset"] == source
        ].sort_values("window_index", kind="stable")

        ordered_positions = source_rows.index.to_list()

        number_of_rows = len(ordered_positions)
        split_position = int(number_of_rows * TRAIN_FRACTION)

        if split_position <= 0 or split_position >= number_of_rows:
            raise ValueError(
                f"Invalid split boundary for source: {source}"
            )

        candidate_train_positions = ordered_positions[:split_position]
        source_test_positions = ordered_positions[split_position:]

        test_start_index = int(
            metadata.loc[
                source_test_positions[0], "window_index"
            ]
        )

        embargo_cutoff = (
            test_start_index - EMBARGO_WINDOW_INDICES
        )

        source_embargo_positions = [
            position
            for position in candidate_train_positions
            if metadata.loc[position, "window_index"] >= embargo_cutoff
        ]

        source_train_positions = [
            position
            for position in candidate_train_positions
            if metadata.loc[position, "window_index"] < embargo_cutoff
        ]

        if not source_train_positions or not source_test_positions:
            raise ValueError(
                f"Empty train or test partition for {source}"
            )

        train_positions.extend(source_train_positions)
        test_positions.extend(source_test_positions)
        embargo_positions.extend(source_embargo_positions)

        source_train = metadata.loc[source_train_positions]
        source_test = metadata.loc[source_test_positions]
        source_embargo = metadata.loc[source_embargo_positions]

        # Verify chronological separation and embargo.
        last_train_index = int(source_train["window_index"].max())
        first_test_index = int(source_test["window_index"].min())

        if first_test_index - last_train_index <= EMBARGO_WINDOW_INDICES:
            raise ValueError(
                f"Temporal embargo validation failed for {source}"
            )

        source_audit[source] = {
            "available_rows": number_of_rows,
            "candidate_training_rows": len(candidate_train_positions),
            "training_rows_after_embargo": len(source_train),
            "testing_rows": len(source_test),
            "embargoed_rows": len(source_embargo),
            "last_training_window_index": last_train_index,
            "first_testing_window_index": first_test_index,
            "window_index_gap": first_test_index - last_train_index,
            "training_target_distribution": class_distribution(
                source_train["target"]
            ),
            "testing_target_distribution": class_distribution(
                source_test["target"]
            ),
        }

    # Keep each output in chronological order within its source.
    train_positions = sorted(
        train_positions,
        key=lambda position: (
            metadata.loc[position, "source_dataset"],
            metadata.loc[position, "window_index"],
        ),
    )

    test_positions = sorted(
        test_positions,
        key=lambda position: (
            metadata.loc[position, "source_dataset"],
            metadata.loc[position, "window_index"],
        ),
    )

    embargo_positions = sorted(
        embargo_positions,
        key=lambda position: (
            metadata.loc[position, "source_dataset"],
            metadata.loc[position, "window_index"],
        ),
    )

    train_metadata = metadata.iloc[train_positions].copy()
    test_metadata = metadata.iloc[test_positions].copy()
    embargo_metadata = metadata.iloc[embargo_positions].copy()

    # Verify that the combined partitions contain both target classes.
    if set(train_metadata["target"].unique()) != EXPECTED_TARGETS:
        raise ValueError(
            "Training partition does not contain both target classes."
        )

    if set(test_metadata["target"].unique()) != EXPECTED_TARGETS:
        raise ValueError(
            "Testing partition does not contain both target classes."
        )

    if set(train_positions) & set(test_positions):
        raise ValueError("Training and testing observations overlap.")

    if set(train_positions) & set(embargo_positions):
        raise ValueError("Training and embargo observations overlap.")

    if set(test_positions) & set(embargo_positions):
        raise ValueError("Testing and embargo observations overlap.")

    audit = {
        "project": "IoT Sentinel AI",
        "stage": "Chronological train/test split",
        "version": "1.0",
        "split_method": "Independent chronological split per source experiment",
        "training_fraction_before_embargo": TRAIN_FRACTION,
        "embargo_window_indices": EMBARGO_WINDOW_INDICES,
        "metadata_rows": len(metadata),
        "training_rows": len(train_metadata),
        "testing_rows": len(test_metadata),
        "embargoed_rows": len(embargo_metadata),
        "training_target_distribution": class_distribution(
            train_metadata["target"]
        ),
        "testing_target_distribution": class_distribution(
            test_metadata["target"]
        ),
        "source_details": source_audit,
        "validation_passed": True,
        "limitations": [
            "Only one normal baseline run and one message flooding run are available.",
            "Training and testing observations originate from the same two experimental runs.",
            "This split does not establish independent-run generalization.",
            "This split does not establish unseen-attack detection.",
            "The embargo is based on original window indices, not retained-row positions.",
        ],
    }

    # Save metadata independently from model predictors.
    metadata_outputs = {
        "train_metadata.csv": train_metadata,
        "test_metadata.csv": test_metadata,
        "embargo_metadata.csv": embargo_metadata,
    }

    for filename, dataframe in metadata_outputs.items():
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output_path}"
            )

        dataframe.to_csv(output_path, index=False)

    # Apply exactly the same row positions to every feature set.
    for filename in FEATURE_FILES:
        input_path = INPUT_DIR / filename

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        features = pd.read_csv(input_path)

        if len(features) != len(metadata):
            raise ValueError(
                f"Row count mismatch between metadata and {filename}"
            )

        if not features["target"].equals(metadata["target"]):
            raise ValueError(
                f"Target alignment failed for {filename}"
            )

        train_features = features.iloc[train_positions].copy()
        test_features = features.iloc[test_positions].copy()

        if train_features.isna().any().any():
            raise ValueError(
                f"Missing values in training data: {filename}"
            )

        if test_features.isna().any().any():
            raise ValueError(
                f"Missing values in testing data: {filename}"
            )

        numeric_train = train_features.drop(
            columns=["target"]
        ).select_dtypes(include=[np.number])

        numeric_test = test_features.drop(
            columns=["target"]
        ).select_dtypes(include=[np.number])

        if np.isinf(numeric_train.to_numpy()).any():
            raise ValueError(
                f"Infinite values in training data: {filename}"
            )

        if np.isinf(numeric_test.to_numpy()).any():
            raise ValueError(
                f"Infinite values in testing data: {filename}"
            )

        train_name = filename.replace(
            "_clean.csv", "_train.csv"
        )

        test_name = filename.replace(
            "_clean.csv", "_test.csv"
        )

        for output_name, dataframe in [
            (train_name, train_features),
            (test_name, test_features),
        ]:
            output_path = OUTPUT_DIR / output_name

            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite: {output_path}"
                )

            dataframe.to_csv(output_path, index=False)

        audit.setdefault("feature_sets", {})[filename] = {
            "predictor_count": len(
                train_features.columns
            ) - 1,
            "training_rows": len(train_features),
            "testing_rows": len(test_features),
            "training_missing_values": int(
                train_features.isna().sum().sum()
            ),
            "testing_missing_values": int(
                test_features.isna().sum().sum()
            ),
            "training_targets_aligned": train_features[
                "target"
            ].reset_index(drop=True).equals(
                train_metadata["target"].reset_index(drop=True)
            ),
            "testing_targets_aligned": test_features[
                "target"
            ].reset_index(drop=True).equals(
                test_metadata["target"].reset_index(drop=True)
            ),
        }

    audit_path = OUTPUT_DIR / "temporal_split_audit_v1.json"

    if audit_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {audit_path}"
        )

    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT V1 COMPLETED")
    print("=" * 60)

    print(f"\nTraining rows: {len(train_metadata)}")
    print(f"Testing rows: {len(test_metadata)}")
    print(f"Embargoed rows: {len(embargo_metadata)}")

    print("\nTraining target distribution:")
    print(
        train_metadata["target"].value_counts().to_string()
    )

    print("\nTesting target distribution:")
    print(
        test_metadata["target"].value_counts().to_string()
    )

    print("\nSOURCE-WISE DETAILS:")
    for source, details in source_audit.items():
        print(f"\n{source}")
        for key, value in details.items():
            print(f"{key}: {value}")

    print("\nFeature sets saved successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
