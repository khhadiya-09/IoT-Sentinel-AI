# IoT Sentinel AI
### Intelligent IoT/MQTT Security Monitoring and Anomaly Detection Using Machine Learning

IoT Sentinel AI is a research-oriented cybersecurity project that explores machine-learning-based anomaly detection in MQTT-enabled IoT networks. It focuses on identifying abnormal MQTT traffic patterns, particularly message flooding, through network traffic analysis and statistical feature extraction.

The project combines a controlled MQTT testbed, packet-level traffic analysis, feature engineering, and machine-learning techniques to investigate the detection of anomalous IoT communication behaviour.

## Research Question

Can a lightweight, explainable machine-learning framework detect known and previously unseen anomalous behaviours in MQTT-based IoT networks while maintaining a low false-positive rate?

## Objectives

- Develop a controlled MQTT-based IoT experimental testbed.
- Collect normal and message-flooding traffic.
- Extract network traffic and MQTT-related features from PCAP files.
- Investigate suitable features for anomaly detection.
- Train and evaluate supervised and unsupervised machine-learning models.
- Examine generalization across different message-flooding intensities.
- Develop an inference engine for offline traffic analysis.

## Project Architecture

The current workflow consists of the following stages:

1. **MQTT Testbed:** Simulated IoT devices communicate through a local Mosquitto MQTT broker.
2. **Traffic Generation:** Normal telemetry and controlled MQTT message-flooding experiments are conducted.
3. **Traffic Capture:** Network traffic is captured using TShark and stored in PCAPNG files.
4. **Feature Extraction:** A custom Python pipeline extracts traffic features from captured packets using time-based windows.
5. **Dataset Preparation:** Experimental features are audited, validated, and prepared for machine learning.
6. **Model Training and Evaluation:** Supervised and unsupervised models are evaluated using run-grouped validation.
7. **Inference:** A trained Random Forest detector processes extracted feature CSV files and generates predictions.

## Technology Stack

- **Programming:** Python
- **MQTT Broker:** Eclipse Mosquitto
- **MQTT Communication:** Paho MQTT
- **Traffic Analysis:** Wireshark / TShark
- **Packet Capture:** Npcap
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Testing:** Pytest
- **Development Environment:** Windows 11, VS Code

## Experimental Setup

The testbed uses three simulated IoT devices publishing telemetry through a local MQTT broker.

The primary experiments include:

- Normal baseline traffic.
- MQTT message flooding at 2 messages/second.
- MQTT message flooding at 5 messages/second.
- MQTT message flooding at 10 messages/second.

The primary experimental dataset contains 2,385 telemetry records across nine experimental runs:

- 975 normal telemetry records.
- 1,410 flooding-labelled telemetry records.

Traffic is captured and processed using 10-second feature windows.

## Machine-Learning Methodology

The project investigates supervised and unsupervised approaches.

Models evaluated include:

- Dummy Classifier (baseline).
- Random Forest using network features.
- Random Forest using network and telemetry features.
- Isolation Forest using network features.

The final inference detector is a Random Forest classifier trained using four interarrival-time features:

- Mean interarrival time.
- Standard deviation of interarrival time.
- Minimum interarrival time.
- Maximum interarrival time.

The final model is saved as a Joblib artifact and used by the custom inference engine.

## Evaluation Results

The initial model evaluation used three-fold StratifiedGroupKFold validation, grouping samples by experimental run.

The Random Forest models achieved perfect out-of-fold classification metrics on the current controlled dataset. However, the dataset contains only 12 attack-present primary windows, so these results require cautious interpretation.

### Held-out Flooding-Intensity Evaluation

Both experimental runs at each flooding rate were held out during evaluation.

| Held-out rate | F1-score | False-positive rate |
|---|---:|---:|
| 2 messages/second | 1.00 | 0.00 |
| 5 messages/second | 1.00 | 0.00 |
| 10 messages/second | 1.00 | 0.00 |

These results provide preliminary evidence of generalization across held-out flooding intensities within the same attack family. They do not demonstrate detection of previously unseen attack families.

### Inference Benchmark

Measured using 1,000 prediction iterations after 100 warm-up iterations:

- Mean inference latency: approximately 76.77 ms.
- Median inference latency: approximately 76.49 ms.
- 95th-percentile latency: approximately 79.40 ms.
- Throughput: approximately 13.03 predictions/second.

These measurements include feature validation and prediction overhead. They are not measurements of end-to-end MQTT detection latency or embedded-device performance.

### End-to-End Functional Validation

An existing PCAPNG experiment was processed through the feature extraction and inference pipeline.

- Six feature windows processed.
- Five correctly classified.
- One attack-present transitional window missed.
- Accuracy: 83.33%.

This is a functional pipeline check using an experiment represented in model training, not an independent evaluation.

## Limitations

The current implementation has several limitations:

- The experimental attack dataset currently focuses on MQTT message flooding.
- Held-out intensity evaluation does not establish generalization to unseen attack families.
- The experimental dataset is relatively small.
- Boundary traffic can be difficult to distinguish from normal traffic.
- Public MQTT dataset integration and cross-dataset evaluation have not yet been completed.
- The current inference engine processes extracted CSV files; continuous live MQTT detection is not implemented.
- Performance on embedded IoT hardware has not been evaluated.

## Future Work

- Integrate suitable public MQTT security datasets.
- Investigate detection of additional MQTT attack behaviours.
- Evaluate cross-dataset generalization.
- Further investigate false positives and transitional traffic.
- Explore explainability techniques for model predictions.
- Extend the inference pipeline towards continuous MQTT traffic monitoring.
- Evaluate computational performance on resource-constrained hardware.

## Repository Structure

```text
IoT-Sentinel-AI/
│
├── configs/          # Experiment configuration
├── models/           # Trained ML model artifacts
├── notebooks/        # Research and analysis scripts
├── reports/          # Evaluation and research reports
├── scripts/          # Training, extraction and evaluation
├── src/              # Core application source code
├── tests/            # Automated tests
│
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd IoT-Sentinel-AI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The experimental pipeline also requires a configured Mosquitto MQTT broker and TShark installation.

## Running the Inference Engine

The current detector accepts an extracted feature CSV file.

```bash
python scripts/inference_engine_v2.py --input <INPUT_CSV> --output <OUTPUT_CSV>
```

The inference engine validates the required features, loads the trained model, generates predictions, and saves the output CSV.

## Running Tests

```bash
python -m pytest tests/test_inference_engine_v2.py -v
```

The current inference-engine test suite contains 21 automated tests.

## Research Focus

IoT Sentinel AI is an ongoing research project exploring practical and lightweight anomaly detection for MQTT-based IoT networks.

The project emphasizes experimental reproducibility, careful evaluation, feature analysis, and transparent reporting of model limitations.

---

**Project:** IoT Sentinel AI  
**Domain:** IoT Security | Network Security | Machine Learning | Anomaly Detection  
**Status:** Research prototype — under development