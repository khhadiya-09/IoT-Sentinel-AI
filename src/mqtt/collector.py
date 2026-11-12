
import argparse
import csv
import json
import math
import signal
import threading
import time

from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt


DEFAULT_BROKER = "127.0.0.1"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "iot-sentinel/devices/telemetry"

CSV_FIELDS = [
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
    "mqtt_topic",
    "collected_at",
]

REQUIRED_FIELDS = [
    "device_id",
    "device_type",
    "timestamp",
    "temperature",
    "humidity",
    "battery",
    "status",
    "traffic_label",
    "experiment_id",
]


class TelemetryCollector:

    def __init__(self, args):

        self.args = args

        self.csv_file = Path(args.csv_output)
        self.jsonl_file = Path(args.jsonl_output)

        self.received_count = 0
        self.saved_count = 0
        self.rejected_count = 0
        self.storage_error_count = 0

        self.device_counts = {}

        self.csv_handle = None
        self.jsonl_handle = None
        self.csv_writer = None

        self.running = threading.Event()
        self.running.set()

        self.connected = threading.Event()
        self.connection_failed = threading.Event()

    def prepare_storage(self):

        self.csv_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.jsonl_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if self.csv_file.resolve() == self.jsonl_file.resolve():
            raise ValueError(
                "CSV and JSONL output paths must be different."
            )

        for path in [self.csv_file, self.jsonl_file]:

            if path.exists() and path.stat().st_size > 0:
                raise FileExistsError(
                    f"Output file already contains data: {path}"
                )

        self.csv_handle = self.csv_file.open(
            "w",
            newline="",
            encoding="utf-8"
        )

        self.jsonl_handle = self.jsonl_file.open(
            "w",
            encoding="utf-8"
        )

        self.csv_writer = csv.DictWriter(
            self.csv_handle,
            fieldnames=CSV_FIELDS
        )

        self.csv_writer.writeheader()

        self.csv_handle.flush()
        self.jsonl_handle.flush()

    def validate_message(self, payload):

        try:

            data = json.loads(payload.decode("utf-8"))

            if not isinstance(data, dict):
                raise ValueError(
                    "Message must be a JSON object."
                )

            missing = [
                field
                for field in REQUIRED_FIELDS
                if field not in data
            ]

            if missing:
                raise ValueError(
                    f"Missing fields: {missing}"
                )

            string_fields = [
                "device_id",
                "device_type",
                "timestamp",
                "status",
                "traffic_label",
                "experiment_id",
            ]

            for field in string_fields:

                value = data[field]

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{field} must be a non-empty string."
                    )

            if data["experiment_id"] != self.args.experiment_id:
                raise ValueError(
                    f"Experiment ID mismatch: "
                    f"expected {self.args.experiment_id}, "
                    f"received {data['experiment_id']}."
                )

            for field in [
                "temperature",
                "humidity",
                "battery",
            ]:

                value = data[field]

                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"{field} must be numeric."
                    )

                if not math.isfinite(value):
                    raise ValueError(
                        f"{field} must be finite."
                    )

            if not 0 <= data["humidity"] <= 100:
                raise ValueError(
                    "Humidity must be between 0 and 100."
                )

            if not 0 <= data["battery"] <= 100:
                raise ValueError(
                    "Battery must be between 0 and 100."
                )

            parsed_timestamp = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )

            if parsed_timestamp.tzinfo is None:
                raise ValueError(
                    "Timestamp must include timezone information."
                )

            return {
                field: data[field]
                for field in REQUIRED_FIELDS
            }

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            OverflowError,
        ) as error:

            print(f"Rejected invalid message: {error}")
            return None

    def save_message(self, data, topic):

        record = {
            **data,
            "mqtt_topic": topic,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        try:

            self.csv_writer.writerow(record)
            self.csv_handle.flush()

            self.jsonl_handle.write(
                json.dumps(record, allow_nan=False) + "\n"
            )

            self.jsonl_handle.flush()

            self.saved_count += 1

            device_id = record["device_id"]

            self.device_counts[device_id] = (
                self.device_counts.get(device_id, 0) + 1
            )

            print(
                f"Stored telemetry from {device_id} "
                f"| Experiment: {record['experiment_id']} "
                f"| Label: {record['traffic_label']}"
            )

        except (OSError, ValueError, TypeError) as error:

            self.storage_error_count += 1
            print(f"Storage error: {error}")

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties
    ):

        if reason_code == 0:

            print("\nConnected to MQTT broker.")

            result, _ = client.subscribe(
                DEFAULT_TOPIC,
                qos=1
            )

            if result != mqtt.MQTT_ERR_SUCCESS:
                print("Subscription request failed.")
                self.connection_failed.set()
                self.running.clear()
                return

            print(f"Subscribed to: {DEFAULT_TOPIC}")
            self.connected.set()

        else:

            print(f"MQTT connection failed: {reason_code}")
            self.connection_failed.set()
            self.running.clear()

    def on_message(self, client, userdata, message):

        self.received_count += 1

        data = self.validate_message(message.payload)

        if data is None:
            self.rejected_count += 1
            return

        self.save_message(data, message.topic)

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties
    ):

        print(f"MQTT disconnected. Reason: {reason_code}")

    def stop(self, signum=None, frame=None):

        if self.running.is_set():
            print("\nStopping telemetry collector...")

        self.running.clear()

    def close_storage(self):

        if self.csv_handle and not self.csv_handle.closed:
            self.csv_handle.close()

        if self.jsonl_handle and not self.jsonl_handle.closed:
            self.jsonl_handle.close()

    def print_summary(self):

        print("\n" + "=" * 65)
        print("       MQTT TELEMETRY COLLECTOR SUMMARY")
        print("=" * 65)

        print("Experiment ID:", self.args.experiment_id)
        print("Received messages:", self.received_count)
        print("Successfully saved:", self.saved_count)
        print("Rejected messages:", self.rejected_count)
        print("Storage errors:", self.storage_error_count)

        print("\nDevice distribution:")

        for device_id, count in self.device_counts.items():
            print(f"{device_id}: {count}")

        print("\nCSV output:", self.csv_file)
        print("JSONL output:", self.jsonl_file)

        print("\nCollector stopped.")
        print("=" * 65)

    def run(self):

        client = None

        try:

            self.prepare_storage()

            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.args.client_id
            )

            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.on_disconnect = self.on_disconnect

            # Windows: CTRL_BREAK_EVENT generates SIGBREAK.
            signal.signal(signal.SIGINT, self.stop)

            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, self.stop)

            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self.stop)

            print("=" * 65)
            print("       IOT SENTINEL AI - TELEMETRY COLLECTOR")
            print("=" * 65)

            print("Experiment ID:", self.args.experiment_id)
            print("Broker:", self.args.broker_host, self.args.broker_port)
            print("Topic:", DEFAULT_TOPIC)
            print("CSV output:", self.csv_file)
            print("JSONL output:", self.jsonl_file)

            client.connect(
                self.args.broker_host,
                self.args.broker_port,
                60
            )

            client.loop_start()

            # Wait for a successful connection or a failure.
            connection_deadline = time.monotonic() + 15

            while self.running.is_set() and not self.connected.is_set():

                if self.connection_failed.is_set():
                    raise ConnectionError(
                        "MQTT connection or subscription failed."
                    )

                if time.monotonic() >= connection_deadline:
                    raise TimeoutError(
                        "MQTT connection timed out."
                    )

                time.sleep(0.1)

            if self.connected.is_set():
                print("\nCollector is ready.")

            while self.running.is_set():
                time.sleep(0.2)

        except KeyboardInterrupt:

            self.stop()

        except (OSError, RuntimeError, ValueError, ConnectionError, TimeoutError) as error:

            print(f"Collector error: {error}")

        finally:

            if client is not None:

                if client.is_connected():
                    client.disconnect()

                client.loop_stop()

            self.close_storage()
            self.print_summary()


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="IoT Sentinel AI MQTT telemetry collector"
    )

    parser.add_argument(
        "--experiment-id",
        required=True
    )

    parser.add_argument(
        "--csv-output",
        required=True
    )

    parser.add_argument(
        "--jsonl-output",
        required=True
    )

    parser.add_argument(
        "--broker-host",
        default=DEFAULT_BROKER
    )

    parser.add_argument(
        "--broker-port",
        type=int,
        default=DEFAULT_PORT
    )

    parser.add_argument(
        "--client-id",
        default="iot_sentinel_experiment_collector"
    )

    args = parser.parse_args()

    if not args.experiment_id.strip():
        parser.error("Experiment ID cannot be empty.")

    if not 1 <= args.broker_port <= 65535:
        parser.error("Invalid MQTT broker port.")

    return args


def main():

    args = parse_arguments()

    collector = TelemetryCollector(args)
    collector.run()


if __name__ == "__main__":
    main()
