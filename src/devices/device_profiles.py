
"""
IoT Sentinel AI
Device Profiles

Defines configurable characteristics for simulated IoT devices.
"""

DEVICE_PROFILES = {
    "sensor_001": {
        "device_type": "temperature_sensor",
        "temperature": {
            "initial": 25.0,
            "minimum": 18.0,
            "maximum": 35.0,
            "variation": 1.0
        },
        "humidity": {
            "initial": 50.0,
            "minimum": 30.0,
            "maximum": 75.0,
            "variation": 2.0
        },
        "battery": {
            "initial": 95.0,
            "depletion": 0.05
        },
        "publish_interval": 2
    },

    "sensor_002": {
        "device_type": "environmental_sensor",
        "temperature": {
            "initial": 27.0,
            "minimum": 20.0,
            "maximum": 38.0,
            "variation": 0.8
        },
        "humidity": {
            "initial": 55.0,
            "minimum": 35.0,
            "maximum": 80.0,
            "variation": 1.5
        },
        "battery": {
            "initial": 90.0,
            "depletion": 0.04
        },
        "publish_interval": 3
    },

    "sensor_003": {
        "device_type": "industrial_sensor",
        "temperature": {
            "initial": 30.0,
            "minimum": 22.0,
            "maximum": 40.0,
            "variation": 1.2
        },
        "humidity": {
            "initial": 45.0,
            "minimum": 25.0,
            "maximum": 70.0,
            "variation": 2.5
        },
        "battery": {
            "initial": 85.0,
            "depletion": 0.03
        },
        "publish_interval": 4
    }
}
