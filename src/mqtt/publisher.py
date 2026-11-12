
import paho.mqtt.client as mqtt
import time

BROKER = "localhost"
PORT = 1883
TOPIC = "iot-sentinel/devices/test"


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    client_id="iot_sentinel_publisher"
)

print("Connecting to MQTT broker...")

client.connect(BROKER, PORT, 60)
client.loop_start()

time.sleep(1)

message = "Hello from IoT Sentinel AI - Python Publisher"

result = client.publish(TOPIC, message)

result.wait_for_publish()

print("Message published successfully!")
print(f"Topic: {TOPIC}")
print(f"Message: {message}")

client.disconnect()
client.loop_stop()

