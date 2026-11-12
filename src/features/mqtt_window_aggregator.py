
from pathlib import Path
import subprocess

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mqtt_publish_features_v1.csv"
)

PCAP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "normal_baseline_v2.pcapng"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mqtt_window_features_v3.csv"
)

TSHARK_PATH = Path(
    r"C:\Program Files\Wireshark\tshark.exe"
)

WINDOW_SIZE_SECONDS = 10


# --------------------------------------------------
# PCAP METADATA
# --------------------------------------------------

def get_capture_timestamp_bounds():
    """Retrieve earliest and latest frame timestamps from the PCAP."""

    if not TSHARK_PATH.exists():
        raise FileNotFoundError(
            f"TShark executable not found: {TSHARK_PATH}"
        )

    if not PCAP_FILE.exists():
        raise FileNotFoundError(
            f"PCAP file not found: {PCAP_FILE}"
        )

    command = [
        str(TSHARK_PATH),
        "-r",
        str(PCAP_FILE),
        "-T",
        "fields",
        "-e",
        "frame.time_epoch",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    timestamps = [
        float(value)
        for value in completed.stdout.splitlines()
        if value.strip()
    ]

    if not timestamps:
        raise ValueError(
            "No frame timestamps found in the PCAP."
        )

    capture_start = min(timestamps)
    capture_end = max(timestamps)

    return capture_start, capture_end, len(timestamps)


# --------------------------------------------------
# LOAD PACKET FEATURES
# --------------------------------------------------

def load_packet_features():
    """Load and validate packet-level MQTT features."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    required_columns = [
        "timestamp_epoch",
        "frame_length_bytes",
        "traffic_direction",
        "inter_arrival_time",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data["timestamp_epoch"] = pd.to_numeric(
        data["timestamp_epoch"],
        errors="raise",
    )

    data["frame_length_bytes"] = pd.to_numeric(
        data["frame_length_bytes"],
        errors="raise",
    )

    return data


# --------------------------------------------------
# AGGREGATE MQTT TRAFFIC
# --------------------------------------------------

def aggregate_windows(
    data,
    capture_start,
    capture_end,
):
    """Aggregate publisher-originated packets into fixed windows."""

    data = data[
        data["traffic_direction"] == "publisher_to_broker"
    ].copy()

    if data.empty:
        raise ValueError(
            "No publisher-to-broker packets found."
        )

    # Confirm that all selected packets belong to the PCAP span.
    if (
        (data["timestamp_epoch"] < capture_start).any()
        or (data["timestamp_epoch"] > capture_end).any()
    ):
        raise ValueError(
            "Packet timestamps fall outside the PCAP timestamp span."
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp_epoch"],
        unit="s",
        utc=True,
    )

    data = data.sort_values("timestamp")
    data = data.set_index("timestamp")

    grouped = data.resample(
        f"{WINDOW_SIZE_SECONDS}s"
    )

    result = grouped.agg(
        packet_count=("frame_length_bytes", "count"),
        mean_frame_length=("frame_length_bytes", "mean"),
        std_frame_length=("frame_length_bytes", "std"),
        mean_inter_arrival=("inter_arrival_time", "mean"),
        std_inter_arrival=("inter_arrival_time", "std"),
        first_packet_epoch=("timestamp_epoch", "min"),
        last_packet_epoch=("timestamp_epoch", "max"),
    )

    # Retain only windows containing publisher packets.
    result = result[
        result["packet_count"] > 0
    ].copy()

    result = result.reset_index()

    result = result.rename(
        columns={"timestamp": "window_start_utc"}
    )

    # Calculate the actual boundaries of each fixed window.
    result["window_start_epoch"] = (
        result["window_start_utc"].astype("int64") / 1e9
    )

    result["window_end_epoch"] = (
        result["window_start_epoch"] + WINDOW_SIZE_SECONDS
    )

    # A window is complete only if it falls entirely within
    # the observed PCAP timestamp span.
    complete_mask = (
        (result["window_start_epoch"] >= capture_start)
        & (result["window_end_epoch"] <= capture_end)
    )

    result["window_coverage_status"] = "partial"

    result.loc[
        complete_mask,
        "window_coverage_status",
    ] = "complete_within_observed_span"

    result["nominal_window_duration_seconds"] = (
        WINDOW_SIZE_SECONDS
    )

    result["messages_per_second"] = pd.NA

    result.loc[
        complete_mask,
        "messages_per_second",
    ] = (
        result.loc[complete_mask, "packet_count"]
        / WINDOW_SIZE_SECONDS
    )

    result["first_packet_utc"] = pd.to_datetime(
        result["first_packet_epoch"],
        unit="s",
        utc=True,
    )

    result["last_packet_utc"] = pd.to_datetime(
        result["last_packet_epoch"],
        unit="s",
        utc=True,
    )

    result = result.drop(
        columns=[
            "first_packet_epoch",
            "last_packet_epoch",
            "window_start_epoch",
            "window_end_epoch",
        ]
    )

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("\nStarting MQTT window aggregation v3...")

    capture_start, capture_end, frame_count = (
        get_capture_timestamp_bounds()
    )

    print("\nPCAP timestamp information:")
    print(
        "Earliest frame:",
        pd.to_datetime(capture_start, unit="s", utc=True),
    )
    print(
        "Latest frame:",
        pd.to_datetime(capture_end, unit="s", utc=True),
    )
    print("Total captured frames:", frame_count)
    print(
        "Observed timestamp span:",
        round(capture_end - capture_start, 3),
        "seconds",
    )

    data = load_packet_features()

    print("\nLoaded packet-level dataset:", data.shape)

    result = aggregate_windows(
        data,
        capture_start,
        capture_end,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFeature dataset saved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("       MQTT WINDOW AGGREGATION V3 SUMMARY")
    print("=" * 70)

    print("Window size:", WINDOW_SIZE_SECONDS, "seconds")
    print("Total aggregated windows:", len(result))

    print("\nWindow coverage classification:")
    print(result["window_coverage_status"].value_counts())

    print(
        "\nTotal publisher-originated packets:",
        result["packet_count"].sum(),
    )

    print("\nAggregated dataset:")
    print(result.to_string(index=False))

    print("\nMissing values:")
    print(result.isnull().sum())

    print("\nAggregation v3 completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
