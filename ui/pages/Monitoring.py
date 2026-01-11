"""
HappySardines - Model Monitoring Dashboard

Displays model performance metrics from hindcast analysis:
- Accuracy trends over time
- Actual vs predicted occupancy comparison
- Per-class performance breakdown
- Alerts for model drift
"""

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="HappySardines - Monitoring",
    page_icon="📊",
    layout="wide"
)

# Constants
ALERT_THRESHOLD = 0.65  # Alert if accuracy drops below this
MODEL_NAME = "occupancy_xgboost_model_new"
CURRENT_MODEL_VERSION = 4  # Current production model version

# Occupancy class labels
OCCUPANCY_LABELS = {
    0: "Empty",
    1: "Many seats",
    2: "Few seats",
    3: "Standing",
    4: "Crowded",
    5: "Full",
    6: "Not accepting",
}

# Colors for occupancy levels
OCCUPANCY_COLORS = {
    0: "#22c55e",  # Green
    1: "#84cc16",  # Lime
    2: "#eab308",  # Yellow
    3: "#f97316",  # Orange
    4: "#ef4444",  # Red
    5: "#ef4444",  # Red
    6: "#6b7280",  # Gray
}


@st.cache_data(ttl=3600)
def fetch_monitoring_data():
    """
    Fetch monitoring data from Hopsworks monitor_fg.

    Returns DataFrame with columns:
    - window_start, trip_id
    - actual_occupancy_mode, predicted_occupancy_mode
    - accuracy, precision, recall, f1_weighted, mae
    - model_version, generated_at
    """
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()

        monitor_fg = fs.get_feature_group("monitor_fg", version=2)
        df = monitor_fg.read()

        if df is not None and not df.empty:
            # Ensure datetime columns are properly typed
            if "window_start" in df.columns:
                df["window_start"] = pd.to_datetime(df["window_start"])
            if "generated_at" in df.columns:
                df["generated_at"] = pd.to_datetime(df["generated_at"])

            print(f"Loaded {len(df)} monitoring records from Hopsworks")
            return df

        return pd.DataFrame()

    except Exception as e:
        print(f"Error loading monitoring data: {e}")
        return pd.DataFrame()


def get_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monitoring data by day."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = df["window_start"].dt.date

    # Get first record per day (metrics are already daily aggregates)
    daily = df.groupby("date").agg({
        "accuracy": "first",
        "precision": "first",
        "recall": "first",
        "f1_weighted": "first",
        "mae": "first",
        "model_version": "first",
    }).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")


def get_hourly_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate actual vs predicted by hour."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["hour"] = df["window_start"].dt.floor("H")

    hourly = df.groupby("hour").agg({
        "actual_occupancy_mode": "mean",
        "predicted_occupancy_mode": "mean",
    }).reset_index()

    return hourly.sort_values("hour")


def get_per_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-class recall, precision, and counts for all 7 occupancy classes."""
    if df.empty:
        return pd.DataFrame()

    results = []
    # Always show all 7 classes (0-6), even if some have no data
    for cls in range(7):
        # Recall: of all actual class X, how many did we predict as X?
        actual_mask = df["actual_occupancy_mode"] == cls
        actual_subset = df[actual_mask]

        if len(actual_subset) > 0:
            correct = (actual_subset["actual_occupancy_mode"] == actual_subset["predicted_occupancy_mode"]).sum()
            actual_count = len(actual_subset)
            recall = correct / actual_count
        else:
            correct = 0
            actual_count = 0
            recall = None

        # Precision: of all predictions of class X, how many were actually X?
        pred_mask = df["predicted_occupancy_mode"] == cls
        pred_subset = df[pred_mask]

        if len(pred_subset) > 0:
            true_positives = (pred_subset["actual_occupancy_mode"] == pred_subset["predicted_occupancy_mode"]).sum()
            pred_count = len(pred_subset)
            precision = true_positives / pred_count
        else:
            pred_count = 0
            precision = None

        results.append({
            "class": cls,
            "label": OCCUPANCY_LABELS.get(cls, f"Class {cls}"),
            "actual_count": actual_count,
            "pred_count": pred_count,
            "recall": recall,
            "precision": precision,
        })

    return pd.DataFrame(results)


def render_metric_card(label: str, value: float, format_str: str = "{:.1%}",
                       threshold_low: float = None, threshold_high: float = None):
    """Render a metric with conditional coloring."""
    formatted = format_str.format(value) if value is not None else "N/A"

    # Determine color
    if threshold_low is not None and value < threshold_low:
        color = "#ef4444"  # Red
    elif threshold_high is not None and value >= threshold_high:
        color = "#22c55e"  # Green
    else:
        color = "#eab308"  # Yellow

    st.markdown(f"""
    <div style="
        background: {color}11;
        border: 1px solid {color}44;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    ">
        <div style="font-size: 0.9em; color: #6b7280; margin-bottom: 4px;">
            {label}
        </div>
        <div style="font-size: 1.8em; font-weight: 600; color: {color};">
            {formatted}
        </div>
    </div>
    """, unsafe_allow_html=True)


# Main page content
st.title("📊 Model Monitoring")
st.markdown("Track model performance over time using hindcast analysis.")

# Load data
with st.spinner("Loading monitoring data..."):
    monitor_df = fetch_monitoring_data()

if monitor_df.empty:
    st.warning("""
    **No monitoring data available yet.**

    Monitoring data is generated daily by the inference pipeline, which compares
    yesterday's predictions to actual observed occupancy.

    The inference pipeline runs after the feature pipeline completes (~10:30 UTC). Check back after it has run at least once.
    """)
    st.stop()

# Model version filter
available_versions = sorted(monitor_df["model_version"].dropna().unique())
if len(available_versions) > 1:
    st.sidebar.subheader("Filter")

    # Default to current model version if available
    default_idx = available_versions.index(CURRENT_MODEL_VERSION) if CURRENT_MODEL_VERSION in available_versions else len(available_versions) - 1

    selected_version = st.sidebar.selectbox(
        "Model Version",
        options=available_versions,
        index=default_idx,
        format_func=lambda x: f"v{int(x)}" + (" (current)" if x == CURRENT_MODEL_VERSION else "")
    )

    # Filter data by selected version
    monitor_df = monitor_df[monitor_df["model_version"] == selected_version]

    if monitor_df.empty:
        st.warning(f"No monitoring data available for model v{int(selected_version)}.")
        st.stop()
else:
    selected_version = available_versions[0] if available_versions else None

# Show warning if viewing old model data
if selected_version is not None and selected_version != CURRENT_MODEL_VERSION:
    st.info(f"""
    **Viewing historical data from model v{int(selected_version)}.**

    The current production model is v{CURRENT_MODEL_VERSION}.
    Data for v{CURRENT_MODEL_VERSION} will appear after the inference pipeline runs.
    """)

# Calculate aggregates
daily_metrics = get_daily_metrics(monitor_df)
hourly_comparison = get_hourly_comparison(monitor_df)
per_class = get_per_class_metrics(monitor_df)

# Get latest metrics
latest = daily_metrics.iloc[-1] if not daily_metrics.empty else None

# Header with model info
if latest is not None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Model Version:** v{int(latest['model_version'])}")
    with col2:
        last_date = latest["date"].strftime("%Y-%m-%d")
        st.markdown(f"**Last Updated:** {last_date}")

# Alert banner
if latest is not None and latest["accuracy"] < ALERT_THRESHOLD:
    st.error(f"""
    ⚠️ **Model Performance Alert**

    Accuracy ({latest['accuracy']:.1%}) is below the threshold ({ALERT_THRESHOLD:.0%}).
    Consider investigating recent data quality or retraining the model.
    """)

st.divider()

# Key metrics cards
st.subheader("Latest Performance")

if latest is not None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "Accuracy",
            latest["accuracy"],
            threshold_low=0.60,
            threshold_high=0.70
        )

    with col2:
        render_metric_card(
            "F1 Score (Weighted)",
            latest["f1_weighted"],
            threshold_low=0.55,
            threshold_high=0.65
        )

    with col3:
        render_metric_card(
            "Precision (weighted)",
            latest["precision"],
            threshold_low=0.55,
            threshold_high=0.70
        )

    with col4:
        render_metric_card(
            "MAE",
            latest["mae"],
            format_str="{:.2f}",
            threshold_low=0.3,  # Lower is better for MAE
            threshold_high=0.6
        )

st.divider()

# Accuracy trend chart
st.subheader("Accuracy Over Time")

if not daily_metrics.empty and len(daily_metrics) > 1:
    chart_data = daily_metrics[["date", "accuracy"]].set_index("date")
    st.line_chart(chart_data, use_container_width=True)

    # Show trend
    if len(daily_metrics) >= 2:
        recent = daily_metrics["accuracy"].iloc[-1]
        previous = daily_metrics["accuracy"].iloc[-2]
        delta = recent - previous
        trend = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
        st.caption(f"{trend} Change from previous: {delta:+.1%}")
else:
    st.info("Need at least 2 days of data to show trend chart.")

st.divider()

# Actual vs Predicted comparison
st.subheader("Actual vs Predicted Occupancy")

if not hourly_comparison.empty:
    # Rename columns for display
    chart_df = hourly_comparison.rename(columns={
        "actual_occupancy_mode": "Actual",
        "predicted_occupancy_mode": "Predicted"
    }).set_index("hour")

    st.line_chart(chart_df, use_container_width=True)
    st.caption("Hourly average occupancy levels (0=Empty, 3=Standing)")
else:
    st.info("No hourly comparison data available.")

st.divider()

# Per-class performance
st.subheader("Per-Class Performance")

if not per_class.empty:
    # Header row
    header_cols = st.columns([2, 1.5, 2, 2])
    with header_cols[0]:
        st.markdown("**Class**")
    with header_cols[1]:
        st.markdown("**Samples**")
    with header_cols[2]:
        st.markdown("**Recall**", help="TP / (TP + FN) — Sensitivity to this class")
    with header_cols[3]:
        st.markdown("**Precision**", help="TP / (TP + FP) — Positive predictive value")

    # Color-coded display for all 7 classes
    for _, row in per_class.iterrows():
        cls = int(row["class"])
        label = row["label"]
        recall = row["recall"]
        precision = row["precision"]
        actual_count = int(row["actual_count"])
        color = OCCUPANCY_COLORS.get(cls, "#6b7280")

        col1, col2, col3, col4 = st.columns([2, 1.5, 2, 2])

        with col1:
            st.markdown(f"**{cls}** - {label}")

        with col2:
            if actual_count > 0:
                st.markdown(f"{actual_count:,}")
            else:
                st.markdown("*Not observed*")

        with col3:
            if recall is not None and actual_count > 0:
                st.progress(recall, text=f"{recall:.1%}")
            else:
                st.markdown("—")

        with col4:
            if precision is not None and not np.isnan(precision):
                st.progress(precision, text=f"{precision:.1%}")
            else:
                st.markdown("—")

    # Explanation
    st.caption("""
    **Recall** = TP / (TP + FN): Of all actual positives, what fraction did we correctly identify? High recall minimizes false negatives.

    **Precision** = TP / (TP + FP): Of all predicted positives, what fraction were correct? High precision minimizes false positives.

    Classes 4-6 (Crowded, Full, Not accepting) are rare in Swedish transit data.
    """)
else:
    st.info("No per-class metrics available.")

st.divider()

# Raw data expander
with st.expander("View Raw Monitoring Data"):
    if not monitor_df.empty:
        st.dataframe(
            monitor_df.sort_values("window_start", ascending=False).head(100),
            use_container_width=True
        )
        st.caption(f"Showing latest 100 of {len(monitor_df):,} total records")
    else:
        st.info("No raw data available.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; opacity: 0.6;'>Model monitoring powered by Hopsworks Feature Store</div>",
    unsafe_allow_html=True
)
