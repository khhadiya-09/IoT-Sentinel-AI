"""IoT Sentinel AI V2 experiment controller.

Supports:
- normal_baseline: normal IoT device traffic only.
- message_flooding: normal traffic before, during, and after a controlled
  MQTT message-flooding phase.

All experiment outputs are stored in a unique run directory. Existing runs
and raw data are never overwritten.
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "configs" / "experiment_config.json"
TSHARK_PATH = Path(r"C:\Program Files\Wireshark\tshark.exe")
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "raw" / "experiments"
COLLECTOR_READY_MESSAGE = "Collector is ready."

DEFAULT_PRE_ATTACK_SECONDS = 15
DEFAULT_ATTACK_SECONDS = 60
DEFAULT_POST_ATTACK_SECONDS = 15
DEFAULT_ATTACK_RATE = 20.0
MAX_ATTACK_RATE = 40.0
PUBLISHER_VERSION = "2"
EXPECTED_TELEMETRY_COLUMNS = [
    "device_id", "device_type", "timestamp", "temperature", "humidity",
    "battery", "status", "traffic_label", "experiment_id", "mqtt_topic",
    "collected_at",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config(config):
    required_sections = [
        "project", "experiment_version", "broker", "capture",
        "devices", "normal_baseline", "windowing",
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")

    broker = config["broker"]
    capture = config["capture"]
    baseline = config["normal_baseline"]
    windowing = config["windowing"]

    if not isinstance(broker.get("host"), str) or not broker["host"].strip():
        raise ValueError("Invalid broker host.")
    if (not isinstance(broker.get("port"), int)
            or not 1 <= broker["port"] <= 65535):
        raise ValueError("Invalid MQTT broker port.")
    if (not isinstance(baseline.get("duration_seconds"), int)
            or baseline["duration_seconds"] <= 0):
        raise ValueError("Baseline duration must be a positive integer.")
    if windowing.get("window_size_seconds", 0) <= 0:
        raise ValueError("Window size must be positive.")
    if not isinstance(capture.get("interface"), int):
        raise ValueError("Capture interface must be an integer.")
    if not capture.get("filter"):
        raise ValueError("Capture filter cannot be empty.")
    if not isinstance(config["devices"], list) or not config["devices"]:
        raise ValueError("No devices configured.")
    if not str(baseline.get("experiment_id", "")).strip():
        raise ValueError("Baseline experiment ID cannot be empty.")
    if not TSHARK_PATH.is_file():
        raise FileNotFoundError(f"TShark executable not found: {TSHARK_PATH}")


def check_broker(host, port):
    try:
        with socket.create_connection((host, port), timeout=5):
            print("MQTT broker is reachable.")
    except OSError as error:
        raise ConnectionError(f"MQTT broker is unavailable: {error}") from error


def save_metadata(path, metadata):
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    temporary_path.replace(path)


def create_experiment(config, scenario, duration, experiment_id):
    run_id = (
        f"{experiment_id}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    experiment_dir = EXPERIMENTS_DIR / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)

    prefix = "normal_baseline" if scenario == "normal_baseline" else "message_flooding"
    paths = {
        "pcap": experiment_dir / f"{prefix}.pcapng",
        "csv": experiment_dir / "telemetry.csv",
        "jsonl": experiment_dir / "telemetry.jsonl",
        "log": experiment_dir / "experiment.log",
        "metadata": experiment_dir / "metadata.json",
    }

    metadata = {
        "project": config["project"],
        "experiment_version": config["experiment_version"],
        "experiment_id": experiment_id,
        "run_id": run_id,
        "scenario": scenario,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "started_at_utc": None,
        "ended_at_utc": None,
        "duration_seconds": duration,
        "broker": config["broker"],
        "capture": config["capture"],
        "devices": config["devices"],
        "windowing": config["windowing"],
        "outputs": {
            key: str(value.relative_to(PROJECT_ROOT))
            for key, value in paths.items()
        },
        "phase_timing": {},
        "results": {},
        "error": None,
        "process_exit_codes": {},
    }
    save_metadata(paths["metadata"], metadata)
    return run_id, experiment_dir, paths, metadata


def start_process(command, log_file, process_name):
    log_handle = log_file.open("a", encoding="utf-8")
    log_handle.write(f"\n\n===== Starting {process_name} =====\n")
    log_handle.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
        )
    except Exception:
        log_handle.close()
        raise
    print(f"{process_name} started. PID: {process.pid}")
    return process, log_handle


def wait_for_process_condition(process, condition, description,
                               timeout=20, interval=0.25):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"Process exited before {description}. Exit code: {process.returncode}"
            )
        if condition():
            print(f"{description} confirmed.")
            return
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {description}.")


def stop_process(process, process_name, timeout=20):
    if process is None or process.poll() is not None:
        return True
    print(f"Stopping {process_name}...")
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=timeout)
        print(f"{process_name} stopped. Exit code: {process.returncode}")
        return True
    except (subprocess.TimeoutExpired, OSError) as error:
        print(f"Graceful stop failed for {process_name}: {error}. Terminating.")
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"Forcefully stopping {process_name}.")
        process.kill()
        process.wait()
    return False


def values_match(csv_value, json_value):
    if csv_value is None and json_value is None:
        return True
    try:
        if csv_value is not None and json_value is not None:
            if isinstance(csv_value, bool) or isinstance(json_value, bool):
                return csv_value == json_value
            return math.isclose(
                float(csv_value), float(json_value), rel_tol=1e-9, abs_tol=1e-9
            )
    except (TypeError, ValueError):
        pass
    return str(csv_value) == str(json_value)


def read_jsonl_records(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not a JSON object.")
            records.append(record)
    return records


def validate_outputs(paths, expected_experiment_id, scenario):
    import pandas as pd

    results = {
        "pcap_exists": paths["pcap"].is_file(),
        "csv_exists": paths["csv"].is_file(),
        "jsonl_exists": paths["jsonl"].is_file(),
        "log_exists": paths["log"].is_file(),
    }
    if not results["csv_exists"]:
        raise FileNotFoundError("Telemetry CSV was not generated.")
    if not results["jsonl_exists"]:
        raise FileNotFoundError("Telemetry JSONL was not generated.")

    dataframe = pd.read_csv(paths["csv"])
    required_columns = {
        "device_id", "device_type", "timestamp", "temperature", "humidity",
        "battery", "status", "traffic_label", "experiment_id", "mqtt_topic",
        "collected_at",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    if list(dataframe.columns) != EXPECTED_TELEMETRY_COLUMNS:
        raise ValueError(
            "CSV schema does not match the exact V2 telemetry schema. "
            f"Expected {EXPECTED_TELEMETRY_COLUMNS}; got {list(dataframe.columns)}"
        )

    expected_labels = {"normal"} if scenario == "normal_baseline" else {
        "normal", "message_flooding"
    }
    observed_labels = set(dataframe["traffic_label"].dropna().unique())
    results["traffic_labels_valid"] = observed_labels.issubset(expected_labels)
    results["observed_traffic_labels"] = sorted(observed_labels)

    results["csv_records"] = len(dataframe)
    results["csv_columns"] = len(dataframe.columns)
    results["csv_missing_values"] = int(dataframe.isnull().sum().sum())
    results["experiment_ids"] = dataframe["experiment_id"].unique().tolist()
    results["device_distribution"] = dataframe["device_id"].value_counts().to_dict()
    results["traffic_label_distribution"] = (
        dataframe["traffic_label"].value_counts().to_dict()
    )

    json_records = read_jsonl_records(paths["jsonl"])
    results["jsonl_records"] = len(json_records)
    results["csv_jsonl_counts_match"] = len(dataframe) == len(json_records)
    records_match = len(dataframe) == len(json_records)
    if records_match:
        for csv_record, json_record in zip(
            dataframe.to_dict(orient="records"), json_records
        ):
            if set(csv_record) != set(json_record):
                records_match = False
                break
            if not all(values_match(csv_record[key], json_record[key])
                       for key in csv_record):
                records_match = False
                break
    results["csv_jsonl_records_match"] = records_match
    results["experiment_id_matches"] = (
        results["experiment_ids"] == [expected_experiment_id]
    )
    results["pcap_size_bytes"] = (
        paths["pcap"].stat().st_size if results["pcap_exists"] else 0
    )
    results["pcap_nonempty"] = results["pcap_size_bytes"] > 0
    results["all_devices_present"] = len(results["device_distribution"]) > 0
    return results


def validate_results(results):
    checks = {
        "PCAP exists": results.get("pcap_exists", False),
        "PCAP is non-empty": results.get("pcap_nonempty", False),
        "CSV exists": results.get("csv_exists", False),
        "JSONL exists": results.get("jsonl_exists", False),
        "Telemetry records collected": results.get("csv_records", 0) > 0,
        "No CSV missing values": results.get("csv_missing_values", -1) == 0,
        "CSV and JSONL counts match": results.get("csv_jsonl_counts_match", False),
        "CSV and JSONL contents match": results.get("csv_jsonl_records_match", False),
        "Experiment ID matches": results.get("experiment_id_matches", False),
        "Devices present": results.get("all_devices_present", False),
        "Traffic labels valid for scenario": results.get("traffic_labels_valid", False),
    }
    print("\nOutput validation:")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} - {name}")
    return all(checks.values())


def wait_until(deadline, description):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def main():
    parser = argparse.ArgumentParser(description="IoT Sentinel AI V2 experiment controller")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare experiment without starting external processes")
    parser.add_argument("--scenario", choices=["normal_baseline", "message_flooding"],
                        default="normal_baseline",
                        help="Experiment scenario (default: normal_baseline)")
    parser.add_argument("--duration-seconds", type=int,
                        help="Override baseline duration, or flooding attack-phase duration")
    parser.add_argument("--pre-attack-seconds", type=int,
                        default=DEFAULT_PRE_ATTACK_SECONDS)
    parser.add_argument("--post-attack-seconds", type=int,
                        default=DEFAULT_POST_ATTACK_SECONDS)
    parser.add_argument("--attack-rate-per-second", type=float,
                        default=DEFAULT_ATTACK_RATE)
    parser.add_argument("--experiment-id",
                        help="Explicit experiment ID; required for message_flooding")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed used by the normal simulator; flood publisher uses seed + 1")
    args = parser.parse_args()

    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("--duration-seconds must be greater than zero.")
    if args.pre_attack_seconds < 1 or args.post_attack_seconds < 1:
        parser.error("Pre-attack and post-attack durations must be at least 1 second.")
    if not 0 < args.attack_rate_per_second <= MAX_ATTACK_RATE:
        parser.error(
            f"--attack-rate-per-second must be greater than 0 and at most {MAX_ATTACK_RATE:g}."
        )
    if args.experiment_id is not None and not args.experiment_id.strip():
        parser.error("--experiment-id cannot be empty.")
    if args.scenario == "message_flooding" and not args.experiment_id:
        parser.error("--experiment-id is required for message_flooding in V2.")

    processes = {}
    log_handles = {}
    metadata = paths = None
    experiment_dir = run_id = None
    failure = None
    cleanup_errors = []

    print("\n" + "=" * 65)
    print("       IOT SENTINEL AI - EXPERIMENT CONTROLLER V2")
    print("=" * 65)

    try:
        config = load_config()
        validate_config(config)

        if args.scenario == "normal_baseline":
            experiment_id = (
                args.experiment_id or config["normal_baseline"]["experiment_id"]
            )
            duration = (args.duration_seconds if args.duration_seconds is not None
                        else config["normal_baseline"]["duration_seconds"])
            pre_seconds = attack_seconds = post_seconds = 0
        else:
            experiment_id = args.experiment_id
            attack_seconds = (args.duration_seconds if args.duration_seconds is not None
                              else DEFAULT_ATTACK_SECONDS)
            pre_seconds = args.pre_attack_seconds
            post_seconds = args.post_attack_seconds
            duration = pre_seconds + attack_seconds + post_seconds

        print(f"Scenario: {args.scenario}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Total duration: {duration} seconds")
        if args.scenario == "message_flooding":
            print(f"Pre-attack: {pre_seconds}s | Attack: {attack_seconds}s "
                  f"at {args.attack_rate_per_second:g} msg/s | Post-attack: {post_seconds}s")

        check_broker(config["broker"]["host"], config["broker"]["port"])
        run_id, experiment_dir, paths, metadata = create_experiment(
            config, args.scenario, duration, experiment_id
        )
        print("\nExperiment directory created:")
        print(experiment_dir)
        print("Run ID:", run_id)

        metadata["reproducibility"] = {
            "runner_version": "2",
            "flood_publisher_module": "src.devices.mqtt_flood_publisher_v2",
            "flood_publisher_version": PUBLISHER_VERSION,
            "simulator_random_seed": args.random_seed,
            "flood_publisher_random_seed": args.random_seed + 1,
            "telemetry_csv_schema": EXPECTED_TELEMETRY_COLUMNS,
        }
        metadata["phase_timing"] = {
            "pre_attack_seconds": pre_seconds,
            "attack_seconds": attack_seconds,
            "post_attack_seconds": post_seconds,
            "attack_rate_per_second": (
                args.attack_rate_per_second if args.scenario == "message_flooding" else None
            ),
        }
        save_metadata(paths["metadata"], metadata)

        if args.dry_run:
            print("\nDry-run mode selected. No external processes were started.")
            return 0

        tshark_command = [
            str(TSHARK_PATH), "-i", str(config["capture"]["interface"]),
            "-f", config["capture"]["filter"], "-w", str(paths["pcap"]),
        ]
        processes["tshark"], log_handles["tshark"] = start_process(
            tshark_command, paths["log"], "TShark"
        )
        wait_for_process_condition(
            processes["tshark"],
            lambda: paths["pcap"].exists() and paths["pcap"].stat().st_size > 0,
            "TShark capture initialization", timeout=20,
        )

        collector_command = [
            sys.executable, "-u", "-m", "src.mqtt.collector",
            "--experiment-id", experiment_id,
            "--csv-output", str(paths["csv"]),
            "--jsonl-output", str(paths["jsonl"]),
            "--broker-host", config["broker"]["host"],
            "--broker-port", str(config["broker"]["port"]),
            "--client-id", f"iot_collector_{uuid.uuid4().hex[:10]}",
        ]
        processes["collector"], log_handles["collector"] = start_process(
            collector_command, paths["log"], "MQTT collector"
        )

        def collector_is_ready():
            if not paths["log"].exists():
                return False
            return COLLECTOR_READY_MESSAGE in paths["log"].read_text(
                encoding="utf-8", errors="replace"
            )

        wait_for_process_condition(
            processes["collector"], collector_is_ready,
            "MQTT collector subscription", timeout=30,
        )

        metadata["status"] = "running"
        metadata["started_at_utc"] = utc_now()
        save_metadata(paths["metadata"], metadata)

        simulator_command = [
            sys.executable, "-u", "-m", "src.devices.device_simulator",
            "--duration-seconds", str(duration),
            "--experiment-id", experiment_id,
            "--broker-host", config["broker"]["host"],
            "--broker-port", str(config["broker"]["port"]),
            "--random-seed", str(args.random_seed),
        ]
        processes["simulator"], log_handles["simulator"] = start_process(
            simulator_command, paths["log"], "Device simulator"
        )

        experiment_start = time.monotonic()

        if args.scenario == "message_flooding":
            attack_start_deadline = experiment_start + pre_seconds
            wait_until(attack_start_deadline, "pre-attack phase")
            if processes["simulator"].poll() is not None:
                raise RuntimeError("Normal device simulator exited before the attack phase.")

            metadata["phase_timing"]["attack_started_at_utc"] = utc_now()
            save_metadata(paths["metadata"], metadata)

            flood_command = [
                sys.executable, "-u", "-m", "src.devices.mqtt_flood_publisher_v2",
                "--broker-host", config["broker"]["host"],
                "--broker-port", str(config["broker"]["port"]),
                "--topic", "iot-sentinel/devices/telemetry",
                "--experiment-id", experiment_id,
                "--rate-per-second", str(args.attack_rate_per_second),
                "--duration-seconds", str(attack_seconds),
                "--random-seed", str(args.random_seed + 1),
            ]
            processes["flood_publisher"], log_handles["flood_publisher"] = start_process(
                flood_command, paths["log"], "MQTT flooding publisher"
            )
            print("\nFlooding phase started.")
            flood_return_code = processes["flood_publisher"].wait(
                timeout=attack_seconds + 30
            )
            metadata["process_exit_codes"]["flood_publisher"] = flood_return_code
            metadata["phase_timing"]["attack_ended_at_utc"] = utc_now()
            save_metadata(paths["metadata"], metadata)
            if flood_return_code != 0:
                raise RuntimeError(
                    f"Flooding publisher exited with code {flood_return_code}."
                )
            print("Flooding phase completed. Normal devices continue during post-attack phase.")

        print("\nExperiment is running. Total duration:", duration, "seconds")
        simulator_return_code = processes["simulator"].wait(timeout=duration + 30)
        metadata["process_exit_codes"]["simulator"] = simulator_return_code
        if simulator_return_code != 0:
            raise RuntimeError(f"Simulator exited with code {simulator_return_code}.")

        collector_stopped = stop_process(processes["collector"], "MQTT collector")
        metadata["process_exit_codes"]["collector"] = processes["collector"].returncode
        tshark_stopped = stop_process(processes["tshark"], "TShark")
        metadata["process_exit_codes"]["tshark"] = processes["tshark"].returncode

        if not collector_stopped or not tshark_stopped:
            raise RuntimeError("One or more processes required forced termination.")
        if processes["collector"].returncode != 0:
            raise RuntimeError("MQTT collector did not exit successfully.")
        if processes["tshark"].returncode != 0:
            raise RuntimeError("TShark did not exit successfully.")

        results = validate_outputs(paths, experiment_id, args.scenario)
        metadata["results"] = results
        metadata["ended_at_utc"] = utc_now()
        if not validate_results(results):
            metadata["status"] = "validation_failed"
            raise RuntimeError("One or more experiment output validations failed.")

        metadata["status"] = "completed"
        save_metadata(paths["metadata"], metadata)
        print("\n" + "=" * 65)
        print("       EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print("Experiment ID:", experiment_id)
        print("Run ID:", run_id)
        print("Duration:", duration, "seconds")
        print("CSV records:", results["csv_records"])
        print("JSONL records:", results["jsonl_records"])
        print("Traffic labels:", results["traffic_label_distribution"])
        print("PCAP size:", results["pcap_size_bytes"], "bytes")
        print("\nExperiment output directory:")
        print(experiment_dir)

    except Exception as error:
        failure = error
        print("\nEXPERIMENT ERROR:", error)

    finally:
        for name in ["flood_publisher", "simulator", "collector", "tshark"]:
            process = processes.get(name)
            if process is not None and process.poll() is None:
                stopped = stop_process(process, name)
                if not stopped:
                    cleanup_errors.append(f"{name} required forced termination.")
            if process is not None and metadata is not None:
                metadata["process_exit_codes"][name] = process.returncode

        for handle in log_handles.values():
            if not handle.closed:
                handle.close()

        if metadata is not None and paths is not None:
            if failure is not None:
                if metadata["status"] != "validation_failed":
                    metadata["status"] = "failed"
                metadata["error"] = str(failure)
            if cleanup_errors:
                metadata["cleanup_errors"] = cleanup_errors
                metadata["status"] = "failed"
            if failure is not None or cleanup_errors:
                metadata["ended_at_utc"] = utc_now()
            save_metadata(paths["metadata"], metadata)

    print("\nExperiment controller finished.")
    if failure is not None or cleanup_errors:
        if cleanup_errors:
            print("Cleanup problems:", cleanup_errors)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
