import streamlit as st
import pandas as pd
import hopsworks
from dotenv import load_dotenv
import os
from xgboost import XGBClassifier
from pathlib import Path
import datetime
import util
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="HappySardines Monitoring",
    page_icon="🐟",
    layout="wide"
)

# Linköping bus station coordinates as default monitoring point: https://traveling.com/sv/buss/station/linkoeping-busstation 
DEFAULT_LAT = 58.419274
DEFAULT_LON = 15.619256

FEATURE_ORDER = [
    "trip_id",
    "vehicle_id",
    "max_speed",
    "n_positions",
    "lat_mean",
    "lon_mean",
    "hour",
    "day_of_week",
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "snowfall", 
    "rain",  
    "is_work_free",
    "is_red_day",
    "is_day_before_holiday",
]

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

st.title("HappySardines Monitoring")

@st.cache_resource
def get_hopsworks_project(api_key, project_name):
    return hopsworks.login(project=project_name, api_key_value=api_key)

@st.cache_resource
def get_features(_project):
    fs = _project.get_feature_store()
    # Read features
    monitor_fg = fs.get_feature_group(
            name="monitor_fg", 
            version=1
        )

    forecast_fg = fs.get_feature_group(
            name="forecast_fg", 
            version=1
        )
    return monitor_fg, forecast_fg

@st.cache_resource
def load_model(model_dir):
    xgb_model = XGBClassifier()
    xgb_model.load_model(model_dir)
    return xgb_model

project = get_hopsworks_project(HOPSWORKS_API_KEY, HOPSWORKS_PROJECT)
monitor_fg, forecast_fg = get_features(project)
mr = project.get_model_registry()

retrieved_model = mr.get_model(name="occupancy_xgboost_model_new", version=2)
saved_model_dir = retrieved_model.download()

xgb_model = load_model(Path(saved_model_dir) / "model.json")

today = pd.Timestamp(datetime.datetime.now().date(), tz="UTC")
today_str = today.strftime("%Y-%m-%d")
hours_ahead = 48
raw_df = monitor_fg.read()
forecast_df = forecast_fg.read()
forecast_df["window_start"] = pd.to_datetime(forecast_df["window_start"])
forecast_df = forecast_df.set_index("window_start")

now = pd.Timestamp.utcnow()

model_path="model_plots/daily_plots"
if not os.path.exists(model_path):
    os.mkdir(model_path)

hourly_df = (
    forecast_df
    .resample("1H")
    .agg({"predicted_occupancy_mode": "mean"})
    .reset_index()
)

hourly_df.rename(columns={
    "predicted_occupancy_mode": "avg_label"
}, inplace=True)

# Plot forecast
fig = util.plot_bus_occupancy_forecast(
    df=hourly_df,
    file_path=f"{model_path}/forecast_{today_str}.png",
)

# Threshold for alerting on accuracy
alert_threshold = 0.65

forecast = forecast_df.reset_index()
hindcast = monitor_fg.read()

# Model performance metrics
st.subheader("Model Performance Metrics (Hindcast)")

if len(hindcast) > 0:
    metrics = ["accuracy", "f1_weighted", "mae"]
    perf_df = hindcast[metrics].mean().to_frame(name="mean_value").reset_index()
    st.dataframe(perf_df)

    # Alert if accuracy is below a threshold
    if perf_df.loc[perf_df["index"] == "accuracy", "mean_value"].values[0] < alert_threshold:
        st.warning(f"Accuracy is below threshold ({alert_threshold:.2f})!")
        
# Hindcast plot
st.subheader("Hindcast: Actual vs Predicted Occupancy")

if len(hindcast) > 0:
    hindcast_plot_df = (
        hindcast
        .set_index("window_start")
        .resample("1H")
        .agg({
            "actual_occupancy_mode": "mean",
            "predicted_occupancy_mode": "mean"
        })
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hindcast_plot_df["window_start"], hindcast_plot_df["actual_occupancy_mode"], label="Actual", marker="o")
    ax.plot(hindcast_plot_df["window_start"], hindcast_plot_df["predicted_occupancy_mode"], label="Predicted", marker="x")
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupancy (0-6)")
    ax.set_title("Hindcast Occupancy")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.info("No hindcast data available for selected filters")

# Forecast plot
st.subheader("Forecasted Occupancy")

# Resample hourly for smooth plotting
forecast_plot_df = (
    forecast
    .set_index("window_start")
    .resample("1H")
    .agg({"predicted_occupancy_mode": "mean"})
    .rename(columns={"predicted_occupancy_mode": "avg_label"})
    .reset_index()
)

fig_forecast = util.plot_bus_occupancy_forecast(
    df=forecast_plot_df,
    file_path=None,  # optional: don't save, just return fig
    title="Forecasted Occupancy",
)

st.pyplot(fig_forecast)

# Model confidence in predictions
st.subheader("Forecast Confidence")

if "predicted_confidence" in forecast.columns:
    conf_df = forecast.groupby("window_start")["predicted_confidence"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(conf_df["window_start"], conf_df["predicted_confidence"], marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Predicted Confidence Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)


