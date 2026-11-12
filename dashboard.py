
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "scripts"))

from inference_engine_v2 import IoTSentinelDetector


MODEL_PATH = ROOT / "models" / "random_forest_interarrival_v2.joblib"

EVALUATION_FILE = ROOT / "reports" / "model_evaluation_v2.csv"
SUMMARY_FILE = ROOT / "reports" / "model_evaluation_summary_v2.json"
TRAFFIC_FILE = ROOT / "reports" / "primary_inference_v2.csv"
FLOOD_RATE_FILE = ROOT / "reports" / "unseen_flood_rate_evaluation_v2.csv"

EXTERNAL_FOLD_FILE = (
    ROOT / "reports" / "mqtteeb" / "multiclass_v1" / "fold_metrics.csv"
)
EXTERNAL_CLASS_FILE = (
    ROOT / "reports" / "mqtteeb" / "multiclass_v1" / "class_metrics.csv"
)

FEATURES = [
    "mean_interarrival_seconds",
    "std_interarrival_seconds",
    "min_interarrival_seconds",
    "max_interarrival_seconds",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IoT Sentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================




st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: #102A43 !important;
        color: #EAF4FF;
    }

    [data-testid="stAppViewContainer"] {
        background: #102A43 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0B1F33 !important;
        border-right: 1px solid #245780;
    }

    [data-testid="stSidebar"] * {
        color: #EAF4FF;
    }

    /* Main headings */
    .main-title {
        font-size: 36px;
        font-weight: 750;
        color: #64B5FF;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #B8D9F5;
        font-size: 15px;
        margin-bottom: 30px;
    }

    h1, h2, h3 {
        color: #EAF4FF !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #173B5E !important;
        border: 1px solid #245780;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #B8D9F5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #64B5FF !important;
    }

    /* Information cards */
    .info-card {
        background: #173B5E !important;
        border: 1px solid #245780;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 14px;
        color: #EAF4FF;
    }

    .info-card h3 {
        color: #64B5FF !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #1769AA !important;
        color: #FFFFFF !important;
        border: 1px solid #64B5FF !important;
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #2586D1 !important;
        border-color: #A5D5FF !important;
    }

    /* Tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: #173B5E !important;
        border: 1px solid #245780;
        border-radius: 10px;
        overflow: hidden;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #173B5E !important;
        border: 1px dashed #64B5FF;
        border-radius: 12px;
        padding: 16px;
    }

    /* Input fields */
    .stSelectbox > div > div,
    .stTextInput input,
    .stNumberInput input {
        background: #173B5E !important;
        color: #EAF4FF !important;
        border-color: #3976A5 !important;
    }

    /* Dividers */
    hr {
        border-color: #245780;
    }

    /* Captions */
    .stCaption {
        color: #B8D9F5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_json(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_detector():
    return IoTSentinelDetector(MODEL_PATH)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🛡️ IoT SENTINEL AI")
    st.caption("MQTT Intrusion Detection Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Traffic Analysis",
            "Intrusion Detection",
            "Model Evaluation",
            "External Benchmark",
        ],
    )

    st.divider()

    st.markdown("### SYSTEM INFORMATION")

    st.markdown("**Model:** Random Forest")
    st.markdown("**Version:** v2")
    st.markdown("**Detection:** Binary classification")
    st.markdown("**Mode:** Offline CSV inference")

    if MODEL_PATH.exists():
        st.success("Trained model available")
    else:
        st.error("Model file not found")

    st.divider()

    st.caption("IoT Sentinel AI | Research Project")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">🛡️ IoT Sentinel AI</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'Machine Learning-Based Intrusion Detection for MQTT-Based IoT Networks'
    '</p>',
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("## System Overview")


    st.divider()

    st.markdown("### 📂 Upload MQTT Traffic Data")

    st.caption(
        "Upload a CSV file to detect potential MQTT intrusions "
        "using the trained Random Forest model."
    )

    overview_file = st.file_uploader(
        "Choose your traffic CSV file",
        type=["csv"],
        key="overview_upload",
    )

    if overview_file is not None:

        try:
            overview_df = pd.read_csv(overview_file)

            if overview_df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            if len(overview_df) > 10000:
                st.error("Maximum upload size is 10,000 rows.")
                st.stop()

            missing = [
                feature for feature in FEATURES
                if feature not in overview_df.columns
            ]

            if missing:
                st.error(
                    "Missing required features: "
                    + ", ".join(missing)
                )
                st.stop()

            st.success(
                f"CSV uploaded successfully: {len(overview_df)} rows."
            )

            with st.expander("Preview uploaded data"):
                st.dataframe(
                    overview_df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )

            if st.button(
                "🚀 Run Intrusion Detection",
                type="primary",
                use_container_width=True,
                key="overview_detect_button",
            ):

                try:
                    detector = load_detector()

                    predictions = []

                    progress = st.progress(0)

                    for index, (_, row) in enumerate(
                        overview_df.iterrows()
                    ):

                        result = detector.predict_one(
                            row.to_dict()
                        )

                        predictions.append(result)

                        progress.progress(
                            (index + 1) / len(overview_df)
                        )

                    old_columns = [
                        "prediction",
                        "predicted_class",
                        "attack_free_probability",
                        "attack_present_probability",
                        "inference_time_ms",
                    ]

                    clean_df = overview_df.drop(
                        columns=old_columns,
                        errors="ignore",
                    )

                    results_df = pd.concat(
                        [
                            clean_df.reset_index(drop=True),
                            pd.DataFrame(predictions),
                        ],
                        axis=1,
                    )

                    st.session_state[
                        "prediction_results"
                    ] = results_df

                    st.success(
                        "Intrusion detection completed successfully!"
                    )

                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")

    # Show previous results when returning to Overview
    if "prediction_results" in st.session_state:

        overview_results = st.session_state[
            "prediction_results"
        ]

        attack_count = int(
            (overview_results["predicted_class"] == 1).sum()
        )

        normal_count = len(overview_results) - attack_count

        st.markdown("### 🔍 Latest Detection Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Windows", len(overview_results))
        c2.metric("Attack-Free", normal_count)
        c3.metric("Attack-Present", attack_count)

        st.download_button(
            "⬇️ Download Detection Results",
            data=overview_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="iot_sentinel_detection_results.csv",
            mime="text/csv",
            key="overview_download",
            use_container_width=True,
        )

        st.caption(
            "These are offline model predictions, not live MQTT monitoring."
        )

    st.divider()


    summary = load_json(str(SUMMARY_FILE))

    if summary is None:
        st.error("Model evaluation summary is unavailable.")
        st.stop()

    metrics = {
        item["model"]: item
        for item in summary["metrics"]
    }

    rf = metrics.get("RandomForest_network", {})

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training Windows", summary["samples"])
    col2.metric("Independent Runs", summary["independent_runs"])
    col3.metric("Attack-Free", summary["negative_samples"])
    col4.metric("Attack-Present", summary["positive_samples"])

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### Detection Model")

        st.markdown(
            """
            <div class="info-card">
                <h3>Random Forest Classifier</h3>
                <p>Binary MQTT intrusion detection model.</p>
                <p><b>Model version:</b> v2</p>
                <p><b>Features:</b> 4 inter-arrival time features</p>
                <p><b>Training samples:</b> 114</p>
                <p><b>Independent runs:</b> 9</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown("### Cross-Validation Results")

        if rf:

            st.metric(
                "Accuracy",
                f"{rf['accuracy'] * 100:.2f}%",
            )

            st.metric(
                "F1 Score",
                f"{rf['f1'] * 100:.2f}%",
            )

            st.metric(
                "ROC-AUC",
                f"{rf['roc_auc'] * 100:.2f}%",
            )

    st.divider()

    st.markdown("### Project Components")

    components = pd.DataFrame(
        {
            "Component": [
                "MQTT Device Simulation",
                "Telemetry Collection",
                "Network Feature Extraction",
                "Binary Intrusion Detection",
                "External Multi-Class Evaluation",
                "Offline Inference Engine",
            ],
            "Technology": [
                "Python, Paho MQTT",
                "MQTT Subscriber",
                "TShark, Wireshark",
                "Scikit-learn",
                "MQTTEEB-D Dataset",
                "Python, Random Forest",
            ],
        }
    )

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Research limitation: The controlled testbed represents "
        "MQTT message flooding only. Its evaluation does not establish "
        "generalisation to unseen attack families."
    )


# ============================================================
# TRAFFIC ANALYSIS
# ============================================================


elif page == "Traffic Analysis":

    st.markdown("## 📡 MQTT Traffic Analysis")
    st.caption("Explore captured MQTT traffic and recorded model predictions.")

    traffic = load_csv(str(TRAFFIC_FILE))

    if traffic is None or traffic.empty:
        st.error("Traffic inference dataset is unavailable or empty.")
        st.stop()

    # Experiment selection
    run_ids = sorted(traffic["run_id"].dropna().unique())

    selected_run = st.selectbox(
        "Select Experiment Run",
        run_ids,
    )

    filtered = traffic[
        traffic["run_id"] == selected_run
    ].copy()

    filtered = filtered.sort_values("window_index")

    # Summary metrics
    total_windows = len(filtered)

    attack_windows = int(
        (filtered["binary_label"] == "attack_present").sum()
    )

    normal_windows = total_windows - attack_windows

    avg_packet_rate = filtered[
        "packet_rate_per_second"
    ].mean()

    st.divider()
    st.markdown("### Traffic Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Windows", total_windows)
    c2.metric("Attack-Free", normal_windows)
    c3.metric("Attack-Present", attack_windows)
    c4.metric(
        "Average Packet Rate",
        f"{avg_packet_rate:.2f} /s",
    )

    st.divider()

    # Packet rate
    st.markdown("### 📈 Packet Rate Over Time")

    packet_chart = filtered.set_index(
        "window_index"
    )[["packet_rate_per_second"]]

    st.line_chart(
        packet_chart,
        color="#64B5FF",
    )

    # Inter-arrival time
    st.markdown("### ⏱️ Mean Inter-Arrival Time")

    interarrival_chart = filtered.set_index(
        "window_index"
    )[["mean_interarrival_seconds"]]

    st.line_chart(
        interarrival_chart,
        color="#42D6C5",
    )

    st.divider()

    # Actual vs predicted
    st.markdown("### 🎯 Actual vs Predicted Detection")

    comparison = filtered[
        ["window_index", "binary_label", "predicted_class"]
    ].copy()

    comparison["Actual Attack"] = (
        comparison["binary_label"] == "attack_present"
    ).astype(int)

    comparison["Predicted Attack"] = (
        comparison["predicted_class"] == 1
    ).astype(int)

    comparison_chart = comparison.set_index(
        "window_index"
    )[["Actual Attack", "Predicted Attack"]]

    st.line_chart(
        comparison_chart,
        color=["#42D6C5", "#64B5FF"],
    )

    # Prediction counts
    st.markdown("### Detection Distribution")

    distribution = pd.DataFrame(
        {
            "Prediction": ["Attack-Free", "Attack-Present"],
            "Windows": [
                int((filtered["predicted_class"] == 0).sum()),
                int((filtered["predicted_class"] == 1).sum()),
            ],
        }
    )

    st.bar_chart(
        distribution.set_index("Prediction"),
        color="#64B5FF",
    )

    st.divider()

    # Traffic records
    st.markdown("### 📋 Experiment Records")

    display_columns = [
        "window_index",
        "scenario",
        "binary_label",
        "prediction",
        "packet_rate_per_second",
        "mean_interarrival_seconds",
        "attack_present_probability",
    ]

    available_columns = [
        col for col in display_columns
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Experiment Records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run}_traffic_analysis.csv",
        mime="text/csv",
    )



# ============================================================
# INTRUSION DETECTION
# ============================================================


elif page == "Intrusion Detection":

    st.markdown("## 🛡️ MQTT Intrusion Detection")
    st.caption(
        "Upload MQTT traffic features and analyse them "
        "using the trained Random Forest detection model."
    )

    st.divider()

    # Model status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Detection Engine")
        st.success("Random Forest model available")
        st.caption("Model: random_forest_interarrival_v2.joblib")

    with col2:
        st.markdown("### Detection Mode")
        st.info("Offline CSV-based inference")
        st.caption("Four inter-arrival time features required")

    st.divider()

    # Required features
    with st.expander("View required input features", expanded=False):
        st.code("\n".join(FEATURES), language="text")

    # Upload
    st.markdown("### 📂 Upload Traffic Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="intrusion_upload",
        help="CSV must contain all four required numerical features."
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            if len(input_df) > 10000:
                st.error("Maximum supported upload size is 10,000 rows.")
                st.stop()

            missing = [
                feature for feature in FEATURES
                if feature not in input_df.columns
            ]

            if missing:
                st.error(
                    "Missing required features: "
                    + ", ".join(missing)
                )
                st.stop()

            st.success(
                f"File loaded successfully — {len(input_df)} rows."
            )

            st.markdown("### Uploaded Data Preview")

            st.dataframe(
                input_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "🚀 Run Intrusion Detection",
                type="primary",
                use_container_width=True,
            ):

                try:
                    detector = load_detector()

                    predictions = []
                    progress = st.progress(0)

                    for index, (_, row) in enumerate(
                        input_df.iterrows()
                    ):

                        result = detector.predict_one(
                            row.to_dict()
                        )

                        predictions.append(result)

                        progress.progress(
                            (index + 1) / len(input_df)
                        )

                    # Remove old prediction columns if present.
                    old_prediction_columns = [
                        "prediction",
                        "predicted_class",
                        "attack_free_probability",
                        "attack_present_probability",
                        "inference_time_ms",
                    ]

                    clean_df = input_df.drop(
                        columns=old_prediction_columns,
                        errors="ignore",
                    )

                    result_df = pd.concat(
                        [
                            clean_df.reset_index(drop=True),
                            pd.DataFrame(predictions),
                        ],
                        axis=1,
                    )

                    st.session_state[
                        "prediction_results"
                    ] = result_df

                    st.success(
                        "Intrusion detection completed successfully!"
                    )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        except Exception as exc:
            st.error(f"Unable to process uploaded CSV: {exc}")

    # Results
    if "prediction_results" in st.session_state:

        result_df = st.session_state["prediction_results"]

        st.divider()

        st.markdown("## 🔍 Detection Results")

        total = len(result_df)

        attack_count = int(
            (result_df["predicted_class"] == 1).sum()
        )

        normal_count = total - attack_count

        avg_probability = (
            result_df["attack_present_probability"].mean() * 100
        )

        avg_latency = result_df["inference_time_ms"].mean()

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Windows", total)
        c2.metric("Attack-Free", normal_count)
        c3.metric("Attack-Present", attack_count)
        c4.metric("Mean Inference Time", f"{avg_latency:.3f} ms")

        st.divider()

        # Attack probability
        st.markdown("### Attack Probability Analysis")

        probability_chart = result_df[
            ["attack_present_probability"]
        ].copy()

        probability_chart.index.name = "Window"

        st.line_chart(
            probability_chart,
            color="#64B5FF",
        )

        st.metric(
            "Average Attack Probability",
            f"{avg_probability:.2f}%",
        )

        st.divider()

        # Distribution
        st.markdown("### Prediction Distribution")

        distribution = pd.DataFrame(
            {
                "Prediction": [
                    "Attack-Free",
                    "Attack-Present",
                ],
                "Windows": [
                    normal_count,
                    attack_count,
                ],
            }
        )

        st.bar_chart(
            distribution.set_index("Prediction"),
            color="#64B5FF",
        )

        st.divider()

        # Result table
        st.markdown("### 📋 Detailed Prediction Results")

        display_columns = [
            "prediction",
            "predicted_class",
            "attack_free_probability",
            "attack_present_probability",
            "inference_time_ms",
        ]

        st.dataframe(
            result_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Detection Results",
            data=csv_data,
            file_name="iot_sentinel_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Note: Predictions are model outputs, not independently "
            "verified attack labels. This interface performs offline "
            "CSV inference, not live MQTT monitoring."
        )



# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.markdown("## Binary Model Evaluation")

    evaluation = load_csv(str(EVALUATION_FILE))

    if evaluation is None:
        st.error("Model evaluation results not found.")
        st.stop()

    st.markdown("### Model Comparison")

    comparison_columns = [
        "model",
        "feature_group",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
    ]

    st.dataframe(
        evaluation[comparison_columns].style.format(
            {
                "accuracy": "{:.2%}",
                "balanced_accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1": "{:.2%}",
                "false_positive_rate": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### F1 Score Comparison")

    chart = evaluation[
        ["model", "f1"]
    ].set_index("model")

    st.bar_chart(chart)

    st.markdown("### Confusion Matrix Counts")

    confusion = evaluation[
        [
            "model",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    ]

    st.dataframe(
        confusion,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown("### Held-Out Flooding Rate Evaluation")

    flood_results = load_csv(str(FLOOD_RATE_FILE))

    if flood_results is not None:

        st.dataframe(
            flood_results.style.format(
                {
                    "accuracy": "{:.2%}",
                    "balanced_accuracy": "{:.2%}",
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1": "{:.2%}",
                    "false_positive_rate": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "These results evaluate the configured MQTT flooding "
        "experiments. They do not demonstrate detection of "
        "unseen attack families."
    )


# ============================================================
# EXTERNAL MQTTEEB-D BENCHMARK
# ============================================================


elif page == "External Benchmark":

    st.markdown("## 🌐 MQTTEB-D External Benchmark")

    st.caption(
        "External six-class MQTT intrusion detection evaluation "
        "using Leave-One-Loop-Out validation."
    )

    fold_metrics = load_csv(str(EXTERNAL_FOLD_FILE))
    class_metrics = load_csv(str(EXTERNAL_CLASS_FILE))

    if fold_metrics is None or class_metrics is None:
        st.error("External benchmark evaluation files are unavailable.")
        st.stop()

    st.divider()

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    st.markdown("### 📊 Overall Classification Performance")

    mean_metrics = fold_metrics[
        [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ]
    ].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean Accuracy",
        f"{mean_metrics['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Balanced Accuracy",
        f"{mean_metrics['balanced_accuracy'] * 100:.2f}%"
    )

    c3.metric(
        "Macro F1 Score",
        f"{mean_metrics['macro_f1'] * 100:.2f}%"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Macro Precision",
        f"{mean_metrics['macro_precision'] * 100:.2f}%"
    )

    c5.metric(
        "Macro Recall",
        f"{mean_metrics['macro_recall'] * 100:.2f}%"
    )

    c6.metric(
        "Weighted F1",
        f"{mean_metrics['weighted_f1'] * 100:.2f}%"
    )

    st.divider()

    # ========================================================
    # LOOP VALIDATION
    # ========================================================

    st.markdown("### 🔄 Leave-One-Loop-Out Evaluation")

    st.caption(
        "Each fold trains on three capture loops and evaluates "
        "on the remaining held-out loop."
    )

    st.markdown("#### Macro F1 Across Capture Loops")

    loop_chart = fold_metrics[
        ["held_out_loop", "macro_f1"]
    ].set_index("held_out_loop")

    st.bar_chart(
        loop_chart,
        color="#64B5FF",
    )

    st.markdown("#### Detailed Fold Results")

    fold_display = fold_metrics.style.format(
        {
            "accuracy": "{:.2%}",
            "balanced_accuracy": "{:.2%}",
            "macro_precision": "{:.2%}",
            "macro_recall": "{:.2%}",
            "macro_f1": "{:.2%}",
            "weighted_f1": "{:.2%}",
        }
    )

    st.dataframe(
        fold_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # CLASS-WISE PERFORMANCE
    # ========================================================

    st.markdown("### 🎯 Class-Wise Performance")

    class_summary = (
        class_metrics
        .groupby("class")[
            ["precision", "recall", "f1_score"]
        ]
        .mean()
        .reset_index()
    )

    st.markdown("#### Precision vs Recall vs F1")

    class_chart = class_summary.set_index("class")[
        ["precision", "recall", "f1_score"]
    ]

    st.line_chart(
        class_chart,
        color=["#64B5FF", "#42D6C5", "#A78BFA"],
    )

    st.markdown("#### Class-Wise Metric Table")

    st.dataframe(
        class_summary.style.format(
            {
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1_score": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ========================================================
    # BEST AND WEAKEST CLASSES
    # ========================================================

    st.markdown("### 📌 Performance Insights")

    best_class = class_summary.loc[
        class_summary["f1_score"].idxmax()
    ]

    weakest_class = class_summary.loc[
        class_summary["f1_score"].idxmin()
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"Strongest class: {best_class['class'].upper()}"
        )

        st.metric(
            "F1 Score",
            f"{best_class['f1_score'] * 100:.2f}%"
        )

    with col2:

        st.warning(
            f"Lowest-performing class: "
            f"{weakest_class['class'].upper()}"
        )

        st.metric(
            "F1 Score",
            f"{weakest_class['f1_score'] * 100:.2f}%"
        )

    st.info(
        "The external benchmark evaluates six MQTT traffic classes. "
        "LOLO measures generalisation across capture loops; it does "
        "not establish generalisation to unseen attack families. "
        "These results are separate from the controlled binary "
        "MQTT flooding experiment."
    )


# ============================================================
# FOOTER
# ============================================================

    st.divider()

    st.caption(
    "IoT Sentinel AI | Research and Experimental Dashboard | "
    "MQTT Intrusion Detection"
)
