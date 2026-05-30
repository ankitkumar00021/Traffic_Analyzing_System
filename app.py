import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

MODEL_PERFORMANCE = {
    "Helmet-less Riding": {"Precision": 0.91, "Recall": 0.87, "F1-Score": 0.89},
    "Wrong-side Driving": {"Precision": 0.88, "Recall": 0.82, "F1-Score": 0.85},
    "Signal Jumping": {"Precision": 0.85, "Recall": 0.79, "F1-Score": 0.82},
    "Mobile Phone Usage": {"Precision": 0.83, "Recall": 0.76, "F1-Score": 0.79},
    "Triple Riding": {"Precision": 0.90, "Recall": 0.85, "F1-Score": 0.87},
}

VIOLATION_LABELS = [
    ("mobile_phone", "Mobile Phone Usage"),
    ("helmet_less", "Helmet-less Events"),
    ("wrong_side", "Wrong-side Events"),
    ("signal_jumping", "Signal Jumping"),
    ("triple_riding", "Triple Riding"),
]


def load_data(json_path):
    if not json_path.exists():
        st.error(f"Output file not found: {json_path}")
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_summary(data):
    violations = data.get("violations", {})
    vehicle_counts = data.get("vehicle_counts", {})
    stats = data.get("statistics", {})

    st.header("Summary Metrics")
    total_violations = sum(violations.values())
    cols = st.columns(5)
    cols[0].metric("Total Violations", total_violations)
    for col, (key, label) in zip(cols[1:], VIOLATION_LABELS[1:]):
        col.metric(label, violations.get(key, 0))

    st.markdown("---")
    st.subheader("Quick Violation Snapshot")
    if violations:
        violation_df = pd.DataFrame(
            [(label, violations.get(key, 0)) for key, label in VIOLATION_LABELS],
            columns=["Violation", "Count"],
        )
        violation_df = violation_df.sort_values(by="Count", ascending=False)
        st.bar_chart(violation_df.set_index("Violation"))
    else:
        st.info("No violation data available yet.")

    st.markdown("---")
    st.subheader("Vehicle Distribution")
    if vehicle_counts:
        df = pd.DataFrame(list(vehicle_counts.items()), columns=["Category", "Count"])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Category", y="Count", palette="mako", ax=ax)
        ax.set_title("Vehicle Category Distribution")
        ax.set_xlabel("")
        st.pyplot(fig)
    else:
        st.info("No vehicle detections available yet.")

    metrics_cols = st.columns(2)
    metrics_cols[0].metric("Helmet suspect frames", stats.get("helmet_suspect_frames", 0))
    metrics_cols[1].metric("Mobile suspect frames", stats.get("mobile_suspect_frames", 0))

    with st.expander("View model performance summary"):
        render_model_performance()


def render_model_performance():
    perf_df = pd.DataFrame.from_dict(MODEL_PERFORMANCE, orient="index").reset_index()
    perf_df.rename(columns={"index": "Violation Type"}, inplace=True)
    st.table(perf_df)


def render_violation_analysis(data):
    st.header("Violation Analysis")
    violations = data.get("violations", {})
    events = data.get("events", [])

    if not violations:
        st.info("No violation data available yet.")
        return

    violation_df = pd.DataFrame(
        [(label, violations.get(key, 0)) for key, label in VIOLATION_LABELS],
        columns=["Violation", "Count"],
    )
    violation_df = violation_df[violation_df["Count"] > 0]

    if not violation_df.empty:
        st.subheader("Violation counts")
        st.bar_chart(violation_df.set_index("Violation"))

    if events:
        st.subheader("Event timeline")
        event_df = pd.DataFrame(events)
        if "type" in event_df.columns:
            event_summary = event_df["type"].value_counts().rename_axis("Event Type").reset_index(name="Count")
            st.table(event_summary)
        st.dataframe(event_df.sort_values(by="time").reset_index(drop=True))
    else:
        st.info("No events recorded yet.")


def render_vehicle_analysis(data):
    st.header("Vehicle Analytics")
    vehicle_counts = data.get("vehicle_counts", {})
    if vehicle_counts:
        df = pd.DataFrame(list(vehicle_counts.items()), columns=["Category", "Count"])
        df = df.sort_values("Count", ascending=False)
        st.write("Vehicle class distribution from the current video.")
        st.bar_chart(df.set_index("Category"))
        if df["Count"].sum() > 0:
            df["Share"] = (df["Count"] / df["Count"].sum() * 100).round(1)
            st.table(df)
    else:
        st.info("No vehicles were detected in the loaded output.")


def render_timeline(data):
    events = data.get("events", [])
    st.header("Timeline / Event Log")
    if not events:
        st.warning("No events logged yet.")
        return
    df = pd.DataFrame(events)
    st.dataframe(df.sort_values(by="time").reset_index(drop=True))


def render_junctions(data):
    junctions = data.get("junctions", [])
    st.header("Junction Log")
    if not junctions:
        st.info("No junction events detected.")
        return
    df = pd.DataFrame(junctions)
    st.dataframe(df.sort_values(by="time").reset_index(drop=True))


def main():
    st.set_page_config(page_title="Traffic Analyzer Dashboard", layout="wide")
    st.title("Traffic Analyzer Dashboard")
    st.write("Interactive view of dashcam traffic analytics with violations, vehicles, and junction insights.")

    output_file = st.text_input("JSON output path", "output.json")
    path = Path(output_file)

    if st.button("Load Data"):
        data = load_data(path)
        if data:
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Violations", "Vehicles", "Junctions"])
            with tab1:
                render_summary(data)
            with tab2:
                render_violation_analysis(data)
            with tab3:
                render_vehicle_analysis(data)
            with tab4:
                render_junctions(data)

    st.markdown("---")
    st.write("Generate `output.json` with the pipeline before loading the dashboard.")


if __name__ == "__main__":
    main()
