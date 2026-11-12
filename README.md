# IoT Sentinel AI
### Machine Learning-Based Security Analysis for MQTT-Enabled IoT Networks

IoT Sentinel AI is a research-oriented cybersecurity project that explores machine-learning-based detection of anomalous traffic in MQTT-enabled IoT networks.

The project combines a controlled MQTT testbed, packet-level traffic analysis, feature engineering, machine learning, and an interactive Streamlit dashboard to investigate MQTT message-flooding detection and evaluate model performance on an external IoT cybersecurity dataset.

## Research Overview

The project investigates two separate evaluation tracks:

1. **Controlled MQTT Testbed:** Binary classification of attack-free and MQTT message-flooding traffic using a locally developed experimental testbed.
2. **External Dataset Benchmark:** Six-class classification using the MQTTEEB-D dataset, evaluated through Leave-One-Loop-Out cross-validation.

These tracks are evaluated separately because they use different datasets, features, and classification tasks.

## Objectives

- Develop a controlled MQTT-based IoT experimental environment.
- Generate normal IoT telemetry and controlled MQTT message-flooding traffic.
- Capture and analyse network traffic using TShark.
- Extract time-based and packet-level traffic features.
- Investigate supervised and unsupervised machine-learning approaches.
- Evaluate detection performance using run-grouped validation.
- Examine performance across different message-flooding intensities.
- Benchmark multiclass classification using an external MQTT cybersecurity dataset.
- Develop an offline inference engine and interactive dashboard.

## Project Architecture

The project follows this workflow:

1. **MQTT Testbed:** Simulated IoT devices communicate through a local Mosquitto MQTT broker.
2. **Traffic Generation:** Normal telemetry and controlled MQTT message-flooding experiments are performed.
3. **Traffic Capture:** Network traffic is captured and stored in PCAP/PCAPNG files.
4. **Feature Extraction:** A Python pipeline uses TShark to extract network traffic features.
5. **Dataset Preparation:** Extracted features are audited, validated, and prepared for machine learning.
6. **Model Training and Evaluation:** Machine-learning models are trained and evaluated using appropriate validation strategies.
7. **Inference:** The trained Random Forest detector processes extracted feature CSV files and generates predictions.
8. **Visualization:** A Streamlit dashboard presents traffic analysis, detection results, and model evaluation metrics.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| MQTT Broker | Eclipse Mosquitto |
| MQTT Communication | Paho MQTT |
| Traffic Analysis | Wireshark / TShark |
| Packet Capture | Npcap |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Streamlit |
| Model Storage | Joblib |
| Testing | Pytest |
| Development Environment | VS Code |

## Research Track 1: Controlled MQTT Testbed

A local MQTT testbed was developed using simulated IoT devices and a Mosquitto broker.

### Experimental Setup

- Three simulated IoT devices.
- Normal sensor telemetry containing temperature, humidity, battery, and device status.
- Controlled MQTT message-flooding experiments at 2, 5, and 10 messages per second.
- Nine independent experimental runs.
- 2,385 telemetry records.
- 126 PCAP-derived traffic windows, including 12 mixed-boundary windows.

The primary binary evaluation used 114 windows after excluding mixed-boundary windows:

- 102 attack-free windows.
- 12 attack-present windows.

### Machine Learning Evaluation

The primary Random Forest models were evaluated using three-fold StratifiedGroupKFold validation, with experimental runs kept separate across folds.

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Random Forest – Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest – Combined | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dummy Baseline | 0.8947 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Isolation Forest – Network | 0.6140 | 0.7843 | 0.2143 | 1.0000 | 0.3529 |

The perfect Random Forest results should be interpreted in the context of the small experimental dataset and its limited number of attack-present windows.

### Held-Out Flooding-Rate Evaluation

The detector was also evaluated across held-out message-flooding intensities:

- 2 messages/second
- 5 messages/second
- 10 messages/second

The recorded evaluation achieved 100% accuracy at each tested intensity across eight windows per rate.

This evaluates different intensities of the same message-flooding attack family. It does not establish generalization to previously unseen attack families.

## Research Track 2: MQTTEEB-D External Benchmark

The project also evaluates a multiclass Random Forest model using the external MQTTEEB-D dataset.

### Dataset Preparation

- Original records: 222,813
- Records with missing labels excluded: 246
- Usable records: 222,567
- Classification classes: 6

The six classes are:

1. Legitimate
2. DoS
3. Flood
4. Brute Force
5. Malformed
6. SlowITe

### Evaluation Methodology

The model was evaluated using four-fold Leave-One-Loop-Out cross-validation.

In each fold, three capture loops were used for training and the remaining loop was held out for testing. This ensured that training and testing data did not share capture loops.

### Overall Results

| Metric | Mean Score |
|---|---:|
| Accuracy | 49.97% |
| Balanced Accuracy | 62.36% |
| Macro Precision | 57.66% |
| Macro Recall | 62.36% |
| Macro F1-score | 54.56% |
| Weighted F1-score | 51.47% |

### Class-wise F1-scores

| Class | F1-score |
|---|---:|
| Legitimate | 26.71% |
| DoS | 38.09% |
| Flood | 96.20% |
| Brute Force | 83.66% |
| Malformed | 52.05% |
| SlowITe | 30.66% |

The results demonstrate varying classification performance across traffic categories. In particular, the model performed strongly on Flood and Brute Force traffic, while Legitimate and SlowITe classification remained challenging.

The evaluation measures generalization across capture loops, not generalization to unseen attack families.

## Offline Inference Engine

The project includes a Random Forest inference engine that processes extracted traffic-feature CSV files.

The final binary detector uses four inter-arrival-time features:

- Mean inter-arrival time
- Standard deviation of inter-arrival time
- Minimum inter-arrival time
- Maximum inter-arrival time

The inference engine validates the input features, generates predictions and class probabilities, and produces an output containing detection results.

### Inference Benchmark

The sequential inference benchmark used 1,000 single-row predictions after 100 warm-up predictions.

| Metric | Result |
|---|---:|
| Mean Latency | 76.772 ms |
| Median Latency | 76.488 ms |
| 95th Percentile Latency | 79.398 ms |
| 99th Percentile Latency | 81.304 ms |
| Throughput | 13.025 predictions/second |

These measurements cover the model prediction and probability-calculation path after feature validation. Model loading and warm-up are excluded. They are not end-to-end MQTT latency measurements or evidence of performance on embedded IoT hardware.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with five main sections.

### 1. Overview
- Project summary and key experimental metrics.
- CSV upload functionality.
- Offline detection.
- Prediction summaries and downloadable results.

### 2. Traffic Analysis
- Recorded MQTT traffic analysis.
- Packet-rate and inter-arrival-time visualization.
- Actual versus predicted detection comparison.
- Traffic distribution and detailed records.

### 3. Intrusion Detection
- CSV-based traffic-feature upload.
- Random Forest inference.
- Attack-free and attack-present probabilities.
- Detailed predictions and downloadable output.

### 4. Model Evaluation
- Binary model comparison.
- F1-score visualization.
- Confusion-matrix results.
- Held-out flooding-rate evaluation.

### 5. External Benchmark
- MQTTEEB-D benchmark results.
- Cross-validation fold metrics.
- Class-wise precision, recall, and F1-scores.

**Note:** The dashboard performs offline CSV-based inference and displays recorded evaluation results. It does not provide live MQTT traffic monitoring.

## Repository Structure

```text
IoT-Sentinel-AI/
│
├── .streamlit/
│   └── config.toml
│
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
├── models/
│   ├── baseline_v1/
│   └── [primary trained models]
│
├── notebooks/
├── reports/
│   ├── figures/
│   └── mqtteeb/
│       └── multiclass_v1/
│
├── research/
├── scripts/
│   ├── inference_engine_v2.py
│   ├── train_evaluate_models_v2.py
│   ├── train_final_detector_v2.py
│   └── train_mqtteeb_multiclass_v1.py
│
├── src/
├── tests/
│
├── dashboard.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

Some directories contain local experimental data or working files that are not distributed with the repository.

## Installation and Setup

### Prerequisites

- Python 3.13
- Git
- Windows, Linux, or macOS
- Mosquitto and TShark for reproducing the MQTT experimental pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/khhadiya-09/IoT-Sentinel-AI.git
cd IoT-Sentinel-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 4. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard uses the included binary Random Forest model and evaluation reports. The required project artifacts must be present in their expected locations.

## Running the Inference Engine

The inference engine accepts an extracted traffic-feature CSV file.

```bash
python scripts/inference_engine_v2.py --input <INPUT_CSV> --output <OUTPUT_CSV>
```

The input CSV must contain the four required inter-arrival-time features.

The engine validates the input, performs prediction, and saves the results to the specified output CSV.

## Running Tests

Run the inference-engine test suite using:

```bash
python -m pytest tests/test_inference_engine_v2.py -v
```

## Dataset and Reproducibility

The MQTTEEB-D external dataset is not included in this repository. Users must obtain the dataset separately and place its files in the directory structure expected by the training script.

The large trained MQTTEEB-D model artifact is also not distributed in this repository. The benchmark evaluation reports are provided separately.

The controlled testbed dataset and packet captures are not included in the public repository. The repository therefore provides the implementation and selected evaluation artifacts rather than a complete copy of all experimental data.

## Limitations

- The controlled testbed contains a limited number of independent experimental runs.
- The primary binary evaluation contains only 12 attack-present windows.
- The controlled experiments focus on MQTT message flooding rather than a broad range of attack families.
- Held-out flooding-rate evaluation does not establish generalization to unseen attack families.
- External benchmark performance varies considerably across the six traffic classes.
- Inference latency measurements are not end-to-end network latency measurements.
- Live MQTT monitoring and embedded-device deployment have not been implemented or validated.

## Future Work

- Expand the controlled MQTT testbed with additional attack scenarios.
- Increase the diversity and scale of experimental traffic.
- Investigate improved generalization across attack types and environments.
- Explore lightweight deployment on resource-constrained IoT devices.
- Investigate real-time MQTT traffic monitoring.
- Improve classification performance on challenging external benchmark classes.

## Author

**Hadiya Khan**

Computer Engineering Student | Cybersecurity and IoT Research

- GitHub: [@khhadiya-09](https://github.com/khhadiya-09)
- Project: IoT Sentinel AI

## Acknowledgements

This project uses the MQTTEEB-D external dataset for additional multiclass benchmarking. Dataset ownership and original research contributions belong to their respective authors.

---

**Project:** IoT Sentinel AI  
**Domain:** IoT Security | MQTT Security | Network Security | Machine Learning
**Project Type:** Academic Research and Cybersecurity Prototype