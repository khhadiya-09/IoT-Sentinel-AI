
from pathlib import Path
import argparse
import json
import platform
import time
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_engine_v2 import (
    FEATURES,
    DEFAULT_MODEL,
    IoTSentinelDetector,
)


DEFAULT_INPUT = ROOT / "data" / "processed" / "ml_dataset_primary_v2.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "inference_benchmark_v2.json"


def benchmark(detector, sample, warmup, iterations):
    """Measure repeated single-row inference performance."""

    print("\nWarming up the detector...")

    for _ in range(warmup):
        detector.predict_one(sample)

    print(f"Warm-up completed: {warmup} iterations")
    print("\nRunning benchmark...")

    latencies_ms = []
    total_start = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter_ns()
        detector.predict_one(sample)
        end = time.perf_counter_ns()

        latencies_ms.append((end - start) / 1_000_000)

    total_elapsed = time.perf_counter() - total_start
    latencies = np.array(latencies_ms)

    return {
        "iterations": iterations,
        "warmup_iterations": warmup,
        "total_benchmark_time_seconds": total_elapsed,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "minimum_latency_ms": float(np.min(latencies)),
        "maximum_latency_ms": float(np.max(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "latency_std_ms": float(np.std(latencies)),
        "throughput_predictions_per_second": (
            iterations / total_elapsed
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="IoT Sentinel AI inference performance benchmark"
    )

    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)

    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero.")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {output_path}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Benchmark input dataset is empty.")

    missing = [feature for feature in FEATURES if feature not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required benchmark features: {missing}"
        )

    sample = df.iloc[0][FEATURES].to_dict()
    detector = IoTSentinelDetector(model_path)

    print("IoT Sentinel AI benchmark loaded.")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Warm-up: {args.warmup}")

    results = benchmark(
        detector,
        sample,
        args.warmup,
        args.iterations,
    )

    report = {
        "benchmark_name": "IoT Sentinel AI Inference Benchmark",
        "benchmark_version": "2.0",
        "model": str(model_path),
        "input_dataset": str(input_path),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "benchmark_method": (
            "Repeated sequential single-row predict_one calls. "
            "Latency includes input validation, model prediction, "
            "probability calculation and Python call overhead. "
            "Model loading and warm-up are excluded."
        ),
        "environment": {
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n=== INFERENCE BENCHMARK RESULTS ===")
    print(f"Iterations       : {results['iterations']}")
    print(f"Warm-up          : {results['warmup_iterations']}")
    print(f"Mean latency     : {results['mean_latency_ms']:.4f} ms")
    print(f"Median latency   : {results['median_latency_ms']:.4f} ms")
    print(f"Minimum latency  : {results['minimum_latency_ms']:.4f} ms")
    print(f"Maximum latency  : {results['maximum_latency_ms']:.4f} ms")
    print(f"P95 latency      : {results['p95_latency_ms']:.4f} ms")
    print(f"P99 latency      : {results['p99_latency_ms']:.4f} ms")
    print(f"Latency std      : {results['latency_std_ms']:.4f} ms")
    print(
        f"Throughput       : "
        f"{results['throughput_predictions_per_second']:.2f} "
        "predictions/second"
    )

    print(f"\nReport saved: {output_path}")
    print("\n[PASS] Benchmark completed successfully.")


if __name__ == "__main__":
    main()
