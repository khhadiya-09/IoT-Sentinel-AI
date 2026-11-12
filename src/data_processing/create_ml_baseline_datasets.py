
from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_selection_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_baseline_v1"
)

EXCLUDED_FEATURES = [
    "std_global_inter_arrival",
    "std_inter_arrival",
    "std_frame_length",
]

FEATURE_FILES = [
    "set_A_network_timing_volume.csv",
    "set_B_network_plus_packet.csv",
    "set_C_extended_hybrid.csv",
]

EXPECTED_TARGETS = {"normal", "message_flooding"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = {
        "project": "IoT Sentinel AI",
        "stage": "ML baseline dataset preparation",
        "version": "1.0",
        "excluded_features": EXCLUDED_FEATURES,
        "datasets": {},
        "all_datasets_passed": True,
    }

    reference_targets = None
    reference_rows = None

    for filename in FEATURE_FILES:
        input_path = INPUT_DIR / filename

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {input_path}"
            )

        df = pd.read_csv(input_path)

        if "target" not in df.columns:
            raise ValueError(
                f"Target column missing in {filename}"
            )

        original_shape = df.shape
        original_targets = df["target"].copy()

        excluded_present = [
            feature
            for feature in EXCLUDED_FEATURES
            if feature in df.columns
        ]

        clean_df = df.drop(
            columns=excluded_present
        ).copy()

        predictor_columns = [
            column
            for column in clean_df.columns
            if column != "target"
        ]

        missing_values = int(
            clean_df.isna().sum().sum()
        )

        numeric_df = clean_df[
            predictor_columns
        ].select_dtypes(include=[np.number])

        infinite_values = int(
            np.isinf(numeric_df.to_numpy()).sum()
        )

        unexpected_targets = sorted(
            set(clean_df["target"].dropna().unique())
            - EXPECTED_TARGETS
        )

        target_missing = int(
            clean_df["target"].isna().sum()
        )

        target_alignment = (
            clean_df["target"].equals(original_targets)
        )

        if reference_rows is None:
            reference_rows = len(clean_df)
            reference_targets = clean_df["target"].reset_index(
                drop=True
            )
        else:
            row_alignment = (
                len(clean_df) == reference_rows
                and clean_df["target"].reset_index(
                    drop=True
                ).equals(reference_targets)
            )

            if not row_alignment:
                raise ValueError(
                    f"Row or target alignment failed: {filename}"
                )

        passed = (
            missing_values == 0
            and infinite_values == 0
            and target_missing == 0
            and len(unexpected_targets) == 0
            and target_alignment
            and len(clean_df) == 280
        )

        output_name = (
            filename.replace(".csv", "_clean.csv")
        )

        output_path = OUTPUT_DIR / output_name

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {output_path}"
            )

        clean_df.to_csv(output_path, index=False)

        audit["datasets"][filename] = {
            "output_file": output_name,
            "original_rows": original_shape[0],
            "original_columns": original_shape[1],
            "clean_rows": len(clean_df),
            "clean_predictor_count": len(predictor_columns),
            "excluded_features": excluded_present,
            "missing_values": missing_values,
            "infinite_values": infinite_values,
            "missing_targets": target_missing,
            "unexpected_targets": unexpected_targets,
            "target_alignment": target_alignment,
            "validation_passed": passed,
            "target_distribution": {
                str(key): int(value)
                for key, value in clean_df[
                    "target"
                ].value_counts().items()
            },
        }

        if not passed:
            audit["all_datasets_passed"] = False

        print(f"\nDataset: {filename}")
        print(f"Original shape: {original_shape}")
        print(f"Clean shape: {clean_df.shape}")
        print(f"Predictors: {len(predictor_columns)}")
        print(f"Missing values: {missing_values}")
        print(f"Infinite values: {infinite_values}")
        print(f"Target alignment: {target_alignment}")
        print(f"Validation: {'PASS' if passed else 'FAIL'}")

    audit_path = OUTPUT_DIR / "ml_baseline_audit_v1.json"

    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=4)

    print("\n" + "=" * 55)
    print("ML BASELINE DATASET PREPARATION COMPLETE")
    print("=" * 55)
    print(
        "Overall result:",
        "PASS" if audit["all_datasets_passed"] else "FAIL"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Audit report: {audit_path}")

    if not audit["all_datasets_passed"]:
        raise SystemExit(
            "One or more datasets failed validation."
        )


if __name__ == "__main__":
    main()
