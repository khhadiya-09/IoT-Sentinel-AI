
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.inference_engine_v2 import (
    FEATURES,
    InputValidationError,
    IoTSentinelDetector,
    validate_features,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_interarrival_v2.joblib"


@pytest.fixture(scope="module")
def detector():
    """Load the actual trained IoT Sentinel AI model."""
    return IoTSentinelDetector(MODEL_PATH)


@pytest.fixture
def valid_features():
    """Provide a valid interarrival feature sample."""
    return {
        "mean_interarrival_seconds": 0.5,
        "std_interarrival_seconds": 0.2,
        "min_interarrival_seconds": 0.1,
        "max_interarrival_seconds": 1.0,
    }


# --------------------------------------------------
# Feature validation tests
# --------------------------------------------------

def test_valid_features(valid_features):
    result = validate_features(valid_features)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == FEATURES
    assert len(result) == 1


def test_missing_feature():
    data = {
        "mean_interarrival_seconds": 0.5,
        "std_interarrival_seconds": 0.2,
    }

    with pytest.raises(InputValidationError, match="Missing required features"):
        validate_features(data)


@pytest.mark.parametrize(
    "invalid_value",
    ["invalid", None, "abc"],
)
def test_nonnumeric_features(valid_features, invalid_value):
    valid_features["mean_interarrival_seconds"] = invalid_value

    with pytest.raises(InputValidationError, match="must be numeric"):
        validate_features(valid_features)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_features(valid_features, invalid_value):
    valid_features["mean_interarrival_seconds"] = invalid_value

    with pytest.raises(InputValidationError, match="must be finite"):
        validate_features(valid_features)


def test_negative_feature(valid_features):
    valid_features["std_interarrival_seconds"] = -0.1

    with pytest.raises(InputValidationError, match="cannot be negative"):
        validate_features(valid_features)


def test_minimum_greater_than_maximum(valid_features):
    valid_features["min_interarrival_seconds"] = 2.0
    valid_features["max_interarrival_seconds"] = 1.0
    valid_features["mean_interarrival_seconds"] = 1.5

    with pytest.raises(
        InputValidationError,
        match="Minimum interarrival cannot exceed maximum",
    ):
        validate_features(valid_features)


def test_mean_below_minimum(valid_features):
    valid_features["mean_interarrival_seconds"] = 0.05

    with pytest.raises(
        InputValidationError,
        match="Mean interarrival must be between",
    ):
        validate_features(valid_features)


def test_mean_above_maximum(valid_features):
    valid_features["mean_interarrival_seconds"] = 1.5

    with pytest.raises(
        InputValidationError,
        match="Mean interarrival must be between",
    ):
        validate_features(valid_features)


# --------------------------------------------------
# Model loading tests
# --------------------------------------------------

def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_detector_loads_model(detector):
    assert detector.model is not None
    assert set(detector.model.classes_) == {0, 1}


# --------------------------------------------------
# Single prediction tests
# --------------------------------------------------

def test_predict_one_returns_valid_result(detector, valid_features):
    result = detector.predict_one(valid_features)

    expected_keys = {
        "prediction",
        "predicted_class",
        "attack_free_probability",
        "attack_present_probability",
        "inference_time_ms",
    }

    assert expected_keys.issubset(result.keys())

    assert result["prediction"] in {
        "attack_free",
        "attack_present",
    }

    assert result["predicted_class"] in {0, 1}

    assert 0.0 <= result["attack_free_probability"] <= 1.0
    assert 0.0 <= result["attack_present_probability"] <= 1.0

    assert (
        result["attack_free_probability"]
        + result["attack_present_probability"]
    ) == pytest.approx(1.0)

    assert result["inference_time_ms"] >= 0


# --------------------------------------------------
# CSV inference tests
# --------------------------------------------------

def test_predict_csv_creates_output(detector, tmp_path):
    source = pd.read_csv(
        ROOT / "data" / "processed" / "ml_dataset_primary_v2.csv"
    )

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output" / "predictions.csv"

    source[FEATURES].head(3).to_csv(input_file, index=False)

    result = detector.predict_csv(input_file, output_file)

    assert output_file.exists()
    assert len(result) == 3

    saved = pd.read_csv(output_file)

    assert len(saved) == 3
    assert "prediction" in saved.columns
    assert "predicted_class" in saved.columns
    assert "inference_time_ms" in saved.columns


def test_predict_csv_refuses_to_overwrite(detector, tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "existing.csv"

    pd.read_csv(
        ROOT / "data" / "processed" / "ml_dataset_primary_v2.csv"
    )[FEATURES].head(1).to_csv(input_file, index=False)

    output_file.write_text("existing data")

    with pytest.raises(FileExistsError):
        detector.predict_csv(input_file, output_file)

    assert output_file.read_text() == "existing data"


def test_predict_csv_missing_input(detector, tmp_path):
    missing_file = tmp_path / "missing.csv"
    output_file = tmp_path / "output.csv"

    with pytest.raises(FileNotFoundError):
        detector.predict_csv(missing_file, output_file)


def test_predict_csv_empty_input(detector, tmp_path):
    input_file = tmp_path / "empty.csv"
    output_file = tmp_path / "output.csv"

    pd.DataFrame(columns=FEATURES).to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="contains no rows"):
        detector.predict_csv(input_file, output_file)


def test_predict_csv_missing_column(detector, tmp_path):
    input_file = tmp_path / "missing_column.csv"
    output_file = tmp_path / "output.csv"

    pd.DataFrame(
        [{"mean_interarrival_seconds": 0.5}]
    ).to_csv(input_file, index=False)

    with pytest.raises(InputValidationError, match="Missing CSV columns"):
        detector.predict_csv(input_file, output_file)


def test_predict_csv_invalid_row(detector, tmp_path):
    input_file = tmp_path / "invalid_row.csv"
    output_file = tmp_path / "output.csv"

    data = {
        "mean_interarrival_seconds": [0.5],
        "std_interarrival_seconds": [0.2],
        "min_interarrival_seconds": [0.1],
        "max_interarrival_seconds": [float("nan")],
    }

    pd.DataFrame(data).to_csv(input_file, index=False)

    with pytest.raises(InputValidationError, match="Invalid input at CSV row"):
        detector.predict_csv(input_file, output_file)

    assert not output_file.exists()
