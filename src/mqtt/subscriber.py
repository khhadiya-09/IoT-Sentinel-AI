
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "iot-sentinel/devices/telemetry"


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Successfully connected to MQTT broker!")

        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
    else:
        print(f"Connection failed. Reason code: {reason_code}")


def on_message(client, userdata, message):
    payload = message.payload.decode("utf-8")

    print("\nMessage received!")
    print(f"Topic: {message.topic}")
    print(f"Payload: {payload}")


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="iot_sentinel_subscriber"
)

client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT broker...")

client.connect(BROKER, PORT, 60)

print("Waiting for messages. Press CTRL+C to stop.")

client.loop_forever()
