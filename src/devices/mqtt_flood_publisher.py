
"""Controlled MQTT message-flooding publisher for IoT Sentinel AI."""

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


DEFAULT_BROKER_HOST = "127.0.0.1"
DEFAULT_BROKER_PORT = 1883
DEFAULT_TOPIC = "iot-sentinel/devices/telemetry"
DEFAULT_EXPERIMENT_ID = "message_flooding_v1"
DEFAULT_RATE = 20
DEFAULT_DURATION = 60

DEVICE_ID = "flood_client_001"
DEVICE_TYPE = "flood_test_client"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a controlled MQTT message-flooding experiment."
    )

    parser.add_argument("--broker-host", default=DEFAULT_BROKER_HOST)
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--rate-per-second", type=float, default=DEFAULT_RATE)
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION)

    return parser.parse_args()


def create_payload(experiment_id, sequence_number):
    """Create a structurally valid telemetry message."""

    return {
        "device_id": DEVICE_ID,
        "device_type": DEVICE_TYPE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": 25.0,
        "humidity": 50.0,
        "battery": 100.0,
        "status": "anomalous",
        "traffic_label": "message_flooding",
        "experiment_id": experiment_id,
        "sequence_number": sequence_number,
    }


def main():
    args = parse_arguments()

    if args.broker_port < 1 or args.broker_port > 65535:
        print("ERROR: Broker port must be between 1 and 65535.")
        return 1

    if args.rate_per_second <= 0 or args.rate_per_second > 100:
        print("ERROR: Rate must be greater than 0 and no more than 100 messages/second.")
        return 1

    if args.duration_seconds < 1 or args.duration_seconds > 300:
        print("ERROR: Duration must be between 1 and 300 seconds.")
        return 1

    connection_ready = threading.Event()
    connection_failed = threading.Event()
    stop_requested = threading.Event()

    publish_confirmations = []
    publish_failures = []

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            connection_ready.set()
            print("Flooding publisher connected to MQTT broker.")
        else:
            connection_failed.set()
            print(f"ERROR: MQTT connection failed: {reason_code}")

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
        if reason_code != 0 and not stop_requested.is_set():
            print(f"WARNING: Unexpected MQTT disconnection: {reason_code}")

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="iot_sentinel_flood_client",
        protocol=mqtt.MQTTv311,
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        print("\n========== MQTT FLOODING EXPERIMENT ==========")
        print(f"Broker: {args.broker_host}:{args.broker_port}")
        print(f"Topic: {args.topic}")
        print(f"Experiment ID: {args.experiment_id}")
        print(f"Publishing rate: {args.rate_per_second} messages/second")
        print(f"Duration: {args.duration_seconds} seconds")
        print("==============================================\n")

        client.connect(args.broker_host, args.broker_port, keepalive=30)
        client.loop_start()

        if not connection_ready.wait(timeout=10):
            if connection_failed.is_set():
                print("ERROR: Could not connect to the broker.")
            else:
                print("ERROR: Broker connection timed out.")
            return 1

        interval = 1.0 / args.rate_per_second
        start_time = time.monotonic()
        end_time = start_time + args.duration_seconds
        next_publish_time = start_time

        sequence_number = 0

        print("Starting controlled message publishing...\n")

        while time.monotonic() < end_time:
            if stop_requested.is_set():
                break

            payload = create_payload(
                args.experiment_id,
                sequence_number + 1,
            )

            result = client.publish(
                args.topic,
                json.dumps(payload),
                qos=1,
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                sequence_number += 1
                publish_confirmations.append(result)
            else:
                publish_failures.append(result.rc)

            next_publish_time += interval
            remaining = next_publish_time - time.monotonic()

            if remaining > 0:
                stop_requested.wait(remaining)

        actual_duration = time.monotonic() - start_time

        print("\nPublishing finished. Waiting for broker acknowledgements...")

        for publish_result in publish_confirmations:
            try:
                publish_result.wait_for_publish(timeout=10)

                if not publish_result.is_published():
                    publish_failures.append("PUBACK timeout")
            except (RuntimeError, ValueError) as error:
                publish_failures.append(str(error))

        acknowledged = (
            len(publish_confirmations) - len(publish_failures)
        )

        print("\n========== EXPERIMENT SUMMARY ==========")
        print(f"Messages submitted: {sequence_number}")
        print(f"Messages acknowledged: {acknowledged}")
        print(f"Publish failures/timeouts: {len(publish_failures)}")
        print(f"Actual duration: {actual_duration:.2f} seconds")
        print(
            f"Measured submission rate: "
            f"{sequence_number / actual_duration:.2f} messages/second"
        )
        print("========================================")

        return 0 if not publish_failures else 1

    except KeyboardInterrupt:
        print("\nPublishing interrupted by user.")
        return 130

    except (OSError, ValueError, mqtt.MQTTException) as error:
        print(f"ERROR: {error}")
        return 1

    finally:
        stop_requested.set()

        try:
            client.disconnect()
        except (OSError, mqtt.MQTTException):
            pass

        try:
            client.loop_stop()
        except (OSError, RuntimeError):
            pass


if __name__ == "__main__":
    sys.exit(main())
