
import argparse
import json
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from src.devices.device_profiles import DEVICE_PROFILES


TOPIC = "iot-sentinel/devices/telemetry"
DEFAULT_BROKER = "127.0.0.1"
DEFAULT_PORT = 1883
DEFAULT_EXPERIMENT_ID = "normal_baseline_v2"
DEFAULT_DURATION = 300
DEFAULT_RANDOM_SEED = 42


class IoTDevice:

    def __init__(self, device_id, profile, rng):
        self.device_id = device_id
        self.profile = profile
        self.rng = rng

        self.temperature = profile["temperature"]["initial"]
        self.humidity = profile["humidity"]["initial"]
        self.battery = profile["battery"]["initial"]

    def update_readings(self):

        temperature_profile = self.profile["temperature"]
        humidity_profile = self.profile["humidity"]
        battery_profile = self.profile["battery"]

        # Gradual temperature changes
        self.temperature += self.rng.uniform(
            -temperature_profile["variation"],
            temperature_profile["variation"]
        )

        self.temperature = max(
            temperature_profile["minimum"],
            min(temperature_profile["maximum"], self.temperature)
        )

        # Gradual humidity changes
        self.humidity += self.rng.uniform(
            -humidity_profile["variation"],
            humidity_profile["variation"]
        )

        self.humidity = max(
            humidity_profile["minimum"],
            min(humidity_profile["maximum"], self.humidity)
        )

        # Gradual battery depletion
        self.battery = max(
            0.0,
            self.battery - battery_profile["depletion"]
        )

    def generate_telemetry(self, experiment_id):

        self.update_readings()

        return {
            "device_id": self.device_id,
            "device_type": self.profile["device_type"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": round(self.temperature, 2),
            "humidity": round(self.humidity, 2),
            "battery": round(self.battery, 2),
            "status": "normal",
            "traffic_label": "normal",
            "experiment_id": experiment_id
        }


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="IoT Sentinel AI MQTT device simulator"
    )

    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_DURATION,
        help="Simulation duration in seconds"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=DEFAULT_EXPERIMENT_ID,
        help="Experiment identifier"
    )

    parser.add_argument(
        "--broker-host",
        type=str,
        default=DEFAULT_BROKER,
        help="MQTT broker hostname or IP"
    )

    parser.add_argument(
        "--broker-port",
        type=int,
        default=DEFAULT_PORT,
        help="MQTT broker port"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if args.duration_seconds <= 0:
        parser.error("Duration must be greater than zero.")

    if not args.experiment_id.strip():
        parser.error("Experiment ID cannot be empty.")

    if not 1 <= args.broker_port <= 65535:
        parser.error("Broker port must be between 1 and 65535.")

    return args


def main():

    args = parse_arguments()

    rng = random.Random(args.random_seed)

    devices = {
        device_id: IoTDevice(device_id, profile, rng)
        for device_id, profile in DEVICE_PROFILES.items()
    }

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="iot_sentinel_device_simulator"
    )

    published_count = 0
    device_counts = {device_id: 0 for device_id in devices}

    print("=" * 65)
    print("       IOT SENTINEL AI - DEVICE SIMULATOR")
    print("=" * 65)

    print("Experiment ID:", args.experiment_id)
    print("Duration:", args.duration_seconds, "seconds")
    print("MQTT broker:", args.broker_host, args.broker_port)
    print("Devices:", ", ".join(devices.keys()))
    print("Random seed:", args.random_seed)

    try:

        print("\nConnecting to MQTT broker...")

        client.connect(
            args.broker_host,
            args.broker_port,
            60
        )

        client.loop_start()

        # Wait for successful MQTT connection.
        connection_deadline = time.monotonic() + 10

        while not client.is_connected():

            if time.monotonic() >= connection_deadline:
                raise TimeoutError(
                    "Could not establish MQTT connection within 10 seconds."
                )

            time.sleep(0.1)

        print("Connected to Mosquitto MQTT broker!")

        print("\nStarting IoT device simulation...\n")

        start_time = time.monotonic()
        end_time = start_time + args.duration_seconds

        next_publish = {
            device_id: start_time
            for device_id in devices
        }

        while time.monotonic() < end_time:

            current_time = time.monotonic()

            for device_id, device in devices.items():

                if current_time >= end_time:
                    break

                if current_time >= next_publish[device_id]:

                    telemetry = device.generate_telemetry(
                        args.experiment_id
                    )

                    message = json.dumps(
                        telemetry,
                        allow_nan=False
                    )

                    result = client.publish(
                        TOPIC,
                        message,
                        qos=1
                    )

                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        raise RuntimeError(
                            f"MQTT publish failed for {device_id}: "
                            f"{result.rc}"
                        )

                    result.wait_for_publish(timeout=5)

                    if not result.is_published():
                        raise TimeoutError(
                            f"Publish acknowledgement timed out "
                            f"for {device_id}."
                        )

                    published_count += 1
                    device_counts[device_id] += 1

                    print(
                        f"[{published_count}] "
                        f"Published telemetry from {device_id}"
                    )

                    next_publish[device_id] = (
                        next_publish[device_id]
                        + device.profile["publish_interval"]
                    )

            time.sleep(0.05)

    except KeyboardInterrupt:

        print("\nSimulation interrupted by user.")

    finally:

        if client.is_connected():
            client.disconnect()

        client.loop_stop()

        print("\nSimulator stopped.")

        print("\nSimulation summary:")
        print("Total messages published:", published_count)

        for device_id, count in device_counts.items():
            print(f"{device_id}: {count} messages")

        print("\nExperiment ID:", args.experiment_id)
        print("=" * 65)


if __name__ == "__main__":
    main()
