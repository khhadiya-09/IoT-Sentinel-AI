import argparse
import sys
from pathlib import Path
import pandas as pd

PACKET_REQUIRED_COLUMNS = {
    "frame_number", "timestamp_epoch", "timestamp_utc", "frame_length_bytes",
    "tcp_stream", "source_port", "destination_port", "mqtt_message_id",
    "mqtt_topic", "traffic_direction", "inter_arrival_time",
    "global_inter_arrival_time",
}
PAYLOAD_REQUIRED_COLUMNS = {
    "frame_number", "timestamp_epoch", "tcp_stream", "source_port",
    "mqtt_message_id", "mqtt_topic", "device_id", "device_type",
    "timestamp", "temperature", "humidity", "battery", "status",
    "traffic_label", "experiment_id",
}
SHARED_COLUMNS = [
    "timestamp_epoch", "tcp_stream", "source_port",
    "mqtt_message_id", "mqtt_topic",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Integrate MQTT packet and payload datasets with global timing."
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_input_columns(packet_df, payload_df):
    missing_packet = PACKET_REQUIRED_COLUMNS - set(packet_df.columns)
    missing_payload = PAYLOAD_REQUIRED_COLUMNS - set(payload_df.columns)
    if missing_packet:
        raise ValueError(f"Missing packet-level columns: {sorted(missing_packet)}")
    if missing_payload:
        raise ValueError(f"Missing payload columns: {sorted(missing_payload)}")


def validate_frame_numbers(packet_df, payload_df):
    for name, dataset in [("Packet", packet_df), ("Payload", payload_df)]:
        frames = pd.to_numeric(dataset["frame_number"], errors="coerce")
        if frames.isna().any() or not frames.mod(1).eq(0).all():
            raise ValueError(f"{name} dataset contains invalid frame numbers.")
        dataset["frame_number"] = frames.astype("int64")


def compare_metadata(packet_df, payload_df):
    left_df = packet_df.set_index("frame_number")
    right_df = payload_df.set_index("frame_number")
    common = left_df.index.intersection(right_df.index)
    report = {}
    for column in SHARED_COLUMNS:
        left = left_df.loc[common, column]
        right = right_df.loc[common, column]
        if column == "timestamp_epoch":
            matches = (
                pd.to_numeric(left, errors="coerce").reset_index(drop=True)
                .sub(pd.to_numeric(right, errors="coerce").reset_index(drop=True))
                .abs().le(0.000001)
            )
        elif column in {"tcp_stream", "source_port", "mqtt_message_id"}:
            matches = (
                pd.to_numeric(left, errors="coerce").reset_index(drop=True)
                .eq(pd.to_numeric(right, errors="coerce").reset_index(drop=True))
            )
        else:
            matches = (
                left.reset_index(drop=True).astype(str)
                .eq(right.reset_index(drop=True).astype(str))
            )
        report[column] = int((~matches).sum())
    return report, len(common)


def validate_timing_values(integrated_df, publisher_df):
    ordered = publisher_df.sort_values(["timestamp_epoch", "frame_number"])

    # Stream-level intervals: first packet of each publisher TCP stream is NaN.
    stream_first_frames = set(
        ordered.groupby("tcp_stream", sort=False).first()["frame_number"].astype("int64")
    )
    stream_missing_frames = set(
        integrated_df.loc[
            integrated_df["inter_arrival_time"].isna(), "frame_number"
        ].astype("int64")
    )
    print("\nStream-level timing validation:")
    print("Publisher TCP streams:", len(stream_first_frames))
    print("Expected missing intervals:", len(stream_first_frames))
    print("Actual missing intervals:", len(stream_missing_frames))
    if stream_missing_frames != stream_first_frames:
        raise ValueError(
            "Stream-level missing intervals must occur at each stream's first "
            f"packet. Expected {sorted(stream_first_frames)}, "
            f"found {sorted(stream_missing_frames)}."
        )

    # Global publisher intervals: only the first publisher packet is NaN.
    first_global_frame = int(ordered.iloc[0]["frame_number"])
    global_missing_frames = set(
        integrated_df.loc[
            integrated_df["global_inter_arrival_time"].isna(), "frame_number"
        ].astype("int64")
    )
    print("\nGlobal publisher timing validation:")
    print("Expected missing intervals: 1")
    print("Actual missing intervals:", len(global_missing_frames))
    if global_missing_frames != {first_global_frame}:
        raise ValueError(
            "Global timing must be missing only on the first publisher packet. "
            f"Expected {[first_global_frame]}, found {sorted(global_missing_frames)}."
        )

    for column in ["inter_arrival_time", "global_inter_arrival_time"]:
        values = pd.to_numeric(integrated_df[column], errors="coerce").dropna()
        if (values < 0).any():
            raise ValueError(f"Negative values detected in {column}.")
    print("Timing validation: PASS")


def integrate_datasets(packet_df, payload_df):
    publisher_df = packet_df[
        packet_df["traffic_direction"] == "publisher_to_broker"
    ].copy()
    print("\nPublisher-to-broker packets:", len(publisher_df))
    print("Decoded payload records:", len(payload_df))
    if publisher_df.empty or payload_df.empty:
        raise ValueError("An input dataset is empty.")

    packet_duplicates = int(publisher_df["frame_number"].duplicated().sum())
    payload_duplicates = int(payload_df["frame_number"].duplicated().sum())
    print("\nDuplicate frame numbers:")
    print("Packet dataset:", packet_duplicates)
    print("Payload dataset:", payload_duplicates)
    if packet_duplicates or payload_duplicates:
        raise ValueError("Duplicate frame numbers detected.")

    mismatch_report, common_count = compare_metadata(publisher_df, payload_df)
    print("\nCommon frame numbers:", common_count)
    print("\nMetadata mismatch report:")
    for column, count in mismatch_report.items():
        print(f"{column}: {count}")
    if any(mismatch_report.values()):
        raise ValueError("Metadata mismatch detected.")

    integrated = pd.merge(
        publisher_df, payload_df, on="frame_number", how="outer",
        suffixes=("_packet", "_payload"), indicator=True,
        validate="one_to_one",
    )
    counts = integrated["_merge"].value_counts()
    matched = int(counts.get("both", 0))
    packet_only = int(counts.get("left_only", 0))
    payload_only = int(counts.get("right_only", 0))
    print("\nIntegration matching results:")
    print("Matched records:", matched)
    print("Packet-only records:", packet_only)
    print("Payload-only records:", payload_only)
    if packet_only or payload_only:
        raise ValueError("Unmatched records detected.")

    redundant = [
        "timestamp_epoch_payload", "tcp_stream_payload",
        "source_port_payload", "mqtt_message_id_payload", "mqtt_topic_payload",
    ]
    integrated = integrated.drop(columns=redundant, errors="ignore")
    integrated = integrated.drop(columns=["_merge"])
    integrated = integrated.sort_values("frame_number").reset_index(drop=True)

    if len(integrated) != len(payload_df):
        raise ValueError("Unexpected final row count.")
    if integrated["frame_number"].duplicated().any():
        raise ValueError("Duplicate frame numbers detected.")

    missing = integrated.isnull().sum()
    unexpected = missing.drop(
        labels=["inter_arrival_time", "global_inter_arrival_time"], errors="ignore"
    )
    if unexpected.any():
        raise ValueError(
            "Unexpected missing values detected:\n"
            f"{unexpected[unexpected > 0]}"
        )

    validate_timing_values(integrated, publisher_df)
    if len(integrated.columns) != 21:
        raise ValueError(f"Expected 21 columns, found {len(integrated.columns)}.")
    return integrated, publisher_df, matched


def save_dataset(dataset, output_file, overwrite):
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            "Use --overwrite to replace it intentionally."
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file, index=False)
    return output_file


def main():
    args = parse_arguments()
    packet_file = Path(args.packet).resolve()
    payload_file = Path(args.payload).resolve()
    output_file = Path(args.output).resolve()
    print("\nStarting MQTT dataset integration (global timing v2)...")

    for file_path in [packet_file, payload_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required dataset not found: {file_path}")
    if packet_file == payload_file:
        raise ValueError("Packet and payload input files must differ.")
    if output_file in [packet_file, payload_file]:
        raise ValueError("Output cannot overwrite an input dataset.")

    packet_df = pd.read_csv(packet_file)
    payload_df = pd.read_csv(payload_file)
    print("\nInput dataset information:")
    print("Packet-level dataset shape:", packet_df.shape)
    print("Payload dataset shape:", payload_df.shape)

    validate_input_columns(packet_df, payload_df)
    validate_frame_numbers(packet_df, payload_df)
    integrated, publisher_df, matched = integrate_datasets(packet_df, payload_df)
    saved_path = save_dataset(integrated, output_file, args.overwrite)

    print("\n" + "=" * 68)
    print("          MQTT DATASET INTEGRATION SUMMARY")
    print("=" * 68)
    print("Packet-level publisher records:", len(publisher_df))
    print("Payload records:", len(payload_df))
    print("Successfully matched:", matched)
    print("Unmatched records: 0")
    print("Final dataset shape:", integrated.shape)
    print("\nDevice distribution:")
    print(integrated["device_id"].value_counts())
    print("\nTraffic label distribution:")
    print(integrated["traffic_label"].value_counts())
    print("\nMissing values:")
    print(integrated.isnull().sum())
    print("\nSaved integrated dataset:")
    print(saved_path)
    print("\nDataset integration completed successfully.")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError, FileExistsError, ValueError, OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
