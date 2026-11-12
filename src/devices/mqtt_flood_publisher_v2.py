
"""Controlled MQTT flooding publisher V2 for IoT Sentinel AI."""

import argparse
import json
import random
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


DEFAULT_BROKER_HOST = "127.0.0.1"
DEFAULT_BROKER_PORT = 1883
DEFAULT_TOPIC = "iot-sentinel/devices/telemetry"

DEVICE_ID = "flood_client_001"
DEVICE_TYPE = "flood_test_client"


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Controlled MQTT flooding publisher V2."
    )

    parser.add_argument("--broker-host", default=DEFAULT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--rate-per-second", type=float, required=True)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    if not 1 <= args.broker_port <= 65535:
        parser.error("Broker port must be between 1 and 65535.")

    if not 0 < args.rate_per_second <= 40:
        parser.error("Rate must be greater than 0 and at most 40.")

    if not 1 <= args.duration_seconds <= 300:
        parser.error("Duration must be between 1 and 300 seconds.")

    if not args.experiment_id.strip():
        parser.error("Experiment ID cannot be empty.")

    return args


def create_payload(experiment_id, rng, readings):

    readings["temperature"] += rng.uniform(-0.2, 0.2)
    readings["humidity"] += rng.uniform(-0.3, 0.3)
    readings["battery"] = max(0.0, readings["battery"] - 0.0005)

    readings["temperature"] = max(18.0, min(40.0, readings["temperature"]))
    readings["humidity"] = max(25.0, min(80.0, readings["humidity"]))

    return {
        "device_id": DEVICE_ID,
        "device_type": DEVICE_TYPE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(readings["temperature"], 2),
        "humidity": round(readings["humidity"], 2),
        "battery": round(readings["battery"], 2),
        "status": "normal",
        "traffic_label": "message_flooding",
        "experiment_id": experiment_id,
    }


def main():

    args = parse_arguments()

    rng = random.Random(args.random_seed)

    readings = {
        "temperature": 27.0,
        "humidity": 50.0,
        "battery": 90.0,
    }

    connection_ready = threading.Event()
    connection_failed = threading.Event()
    stop_requested = threading.Event()

    submitted = 0
    acknowledged = 0
    failures = []

    pending_results = []

    def on_connect(client, userdata, flags, reason_code, properties):

        if reason_code == 0:
            connection_ready.set()
            print("V2 flooding publisher connected.")
        else:
            connection_failed.set()
            print(f"ERROR: MQTT connection failed: {reason_code}")

    def handle_signal(signum, frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="iot_sentinel_flood_client_v2",
        protocol=mqtt.MQTTv311,
    )

    client.on_connect = on_connect

    try:

        print("\n========== MQTT FLOODING V2 ==========")
        print("Experiment ID:", args.experiment_id)
        print("Configured rate:", args.rate_per_second)
        print("Duration:", args.duration_seconds)
        print("Random seed:", args.random_seed)
        print("QoS: 1")
        print("=====================================\n")

        client.connect(
            args.broker_host,
            args.broker_port,
            keepalive=30
        )

        client.loop_start()

        if not connection_ready.wait(timeout=10):

            if connection_failed.is_set():
                raise ConnectionError("MQTT connection rejected.")

            raise TimeoutError("MQTT connection timed out.")

        interval = 1.0 / args.rate_per_second

        start_time = time.monotonic()
        end_time = start_time + args.duration_seconds
        next_publish_time = start_time

        while time.monotonic() < end_time:

            if stop_requested.is_set():
                break

            payload = create_payload(
                args.experiment_id,
                rng,
                readings
            )

            result = client.publish(
                args.topic,
                json.dumps(payload, allow_nan=False),
                qos=1,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                submitted += 1
                pending_results.append(result)
            else:
                failures.append(f"Publish error: {result.rc}")

            next_publish_time += interval

            remaining = next_publish_time - time.monotonic()

            if remaining > 0:
                stop_requested.wait(remaining)

        actual_duration = time.monotonic() - start_time

        print("\nPublishing completed. Checking acknowledgements...")

        for result in pending_results:

            try:
                result.wait_for_publish(timeout=10)

                if result.is_published():
                    acknowledged += 1
                else:
                    failures.append("PUBACK timeout")

            except (RuntimeError, ValueError) as error:
                failures.append(str(error))

        measured_rate = submitted / actual_duration if actual_duration else 0

        print("\n========== V2 EXPERIMENT SUMMARY ==========")
        print("Messages submitted:", submitted)
        print("Messages acknowledged:", acknowledged)
        print("Publish failures/timeouts:", len(failures))
        print(f"Actual duration: {actual_duration:.2f} seconds")
        print(f"Configured rate: {args.rate_per_second:.2f} msg/s")
        print(f"Measured submission rate: {measured_rate:.2f} msg/s")
        print("Random seed:", args.random_seed)
        print("===========================================\n")

        return 0 if not failures and acknowledged == submitted else 1

    except (OSError, ValueError, mqtt.MQTTException, TimeoutError,
            ConnectionError) as error:

        print("ERROR:", error)
        return 1

    finally:

        stop_requested.set()

        try:
            client.disconnect()
            client.loop_stop()
        except (OSError, RuntimeError, mqtt.MQTTException):
            pass


if __name__ == "__main__":
    sys.exit(main())
