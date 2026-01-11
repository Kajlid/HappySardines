"""
Trip-Based Inference Pipeline for HappySardines Bus Occupancy Prediction

This pipeline:
1. Loads the trained XGBoost model from Hopsworks Model Registry (v4)
2. Generates per-trip occupancy predictions for the next 2 days
3. Stores predictions in forecast_fg for UI consumption
4. Compares yesterday's predictions to actuals for monitoring (monitor_fg)

This complements the heatmap-based predictions in ui/precompute_heatmaps.py
by providing trip-specific forecasts rather than geographic grid predictions.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from dotenv import load_dotenv

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from util import fetch_weather_forecast, get_holiday_features_for_date

load_dotenv()

# Model configuration
MODEL_NAME = "occupancy_xgboost_model_new"
MODEL_VERSION = 4  # Use v4 which has better class 2-3 recall

# Feature order - must match training pipeline exactly
FEATURE_ORDER = [
    "trip_id",
    "vehicle_id",
    "max_speed",
    "n_positions",
    "lat_min",
    "lat_max",
    "lat_mean",
    "lon_min",
    "lon_max",
    "lon_mean",
    "bearing_min",
    "bearing_max",
    "hour",
    "day_of_week",
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "rain",
    "snowfall",
    "is_work_free",
    "is_red_day",
    "is_day_before_holiday",
]

# Default vehicle features for predictions
DEFAULT_VEHICLE_FEATURES = {
    "vehicle_id": 0,
    "max_speed": 45.0,
    "n_positions": 30,
    "bearing_min": 0.0,
    "bearing_max": 360.0,
}

# Default location (Linköping center) for trips without location data
DEFAULT_LAT = 58.41
DEFAULT_LON = 15.62

# Prediction configuration
FORECAST_DAYS = 2  # Today and tomorrow
HOURS = list(range(5, 24))  # 5:00-23:00


def load_model_from_registry(project):
    """Load model from Hopsworks Model Registry."""
    print(f"Fetching model {MODEL_NAME} v{MODEL_VERSION} from registry...")
    mr = project.get_model_registry()

    model_entry = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    print(f"Downloading model version {model_entry.version}...")
    model_dir = model_entry.download()

    print("Loading XGBoost model...")
    model = XGBClassifier()
    model.load_model(Path(model_dir) / "model.json")

    return model


def get_static_trips(fs):
    """Load static trip information."""
    print("Loading static trip info...")
    try:
        fg = fs.get_feature_group("static_trip_info_fg", version=1)
        df = fg.read()
        print(f"Loaded {len(df)} trips")
        return df
    except Exception as e:
        print(f"Error loading static trips: {e}")
        return pd.DataFrame()


def get_weather_for_datetime(target_datetime: datetime) -> dict:
    """Fetch weather for a specific datetime."""
    try:
        weather_df = fetch_weather_forecast(
            DEFAULT_LAT, DEFAULT_LON,
            past_days=1,
            forecast_days=3
        )
        if weather_df is not None and not weather_df.empty:
            # Find closest hour in forecast
            target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.tz_localize(None)

            # Find the row closest to target time
            time_diff = abs(weather_df["timestamp"] - target_hour)
            closest_idx = time_diff.idxmin()
            row = weather_df.loc[closest_idx]

            return {
                "temperature_2m": float(row.get("temperature_2m", 10.0) or 10.0),
                "precipitation": float(row.get("precipitation", 0.0) or 0.0),
                "cloud_cover": float(row.get("cloud_cover", 50.0) or 50.0),
                "wind_speed_10m": float(row.get("wind_speed_10m", 5.0) or 5.0),
                "snowfall": float(row.get("snowfall", 0.0) or 0.0),
                "rain": float(row.get("rain", 0.0) or 0.0),
            }
    except Exception as e:
        print(f"Warning: Failed to fetch weather: {e}")

    # Fallback defaults
    return {
        "temperature_2m": 10.0,
        "precipitation": 0.0,
        "cloud_cover": 50.0,
        "wind_speed_10m": 5.0,
        "snowfall": 0.0,
        "rain": 0.0,
    }


def get_holidays_for_date(target_date: datetime) -> dict:
    """Get holiday features for a specific date."""
    try:
        holidays = get_holiday_features_for_date(target_date)
        return {
            "is_work_free": int(holidays.get("is_work_free", False)),
            "is_red_day": int(holidays.get("is_red_day", False)),
            "is_day_before_holiday": int(holidays.get("is_day_before_holiday", False)),
        }
    except Exception as e:
        print(f"Warning: Failed to fetch holidays: {e}")
        return {"is_work_free": 0, "is_red_day": 0, "is_day_before_holiday": 0}


def generate_trip_forecasts(model, trips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate occupancy predictions for all trips across forecast period.

    Returns DataFrame with columns:
    - window_start, trip_id, predicted_occupancy_mode, predicted_confidence
    - route_short_name, route_long_name (for display)
    """
    print(f"\nGenerating forecasts for {len(trips_df)} trips x {len(HOURS)} hours x {FORECAST_DAYS} days...")

    now = datetime.now()
    forecast_dates = [now.date() + timedelta(days=i) for i in range(FORECAST_DAYS)]

    # Pre-fetch weather for all hours we need
    weather_cache = {}
    holiday_cache = {}

    for date in forecast_dates:
        holiday_cache[date] = get_holidays_for_date(datetime.combine(date, datetime.min.time()))
        for hour in HOURS:
            dt = datetime.combine(date, datetime.min.time()).replace(hour=hour)
            weather_cache[(date, hour)] = get_weather_for_datetime(dt)

    # Build all prediction rows
    all_rows = []
    unique_trips = trips_df["trip_id"].unique()

    for trip_id in unique_trips:
        trip_row = trips_df[trips_df["trip_id"] == trip_id].iloc[0]

        for date in forecast_dates:
            holidays = holiday_cache[date]

            for hour in HOURS:
                weather = weather_cache[(date, hour)]
                window_start = datetime.combine(date, datetime.min.time()).replace(hour=hour)

                row = {
                    "trip_id": trip_id,
                    "vehicle_id": DEFAULT_VEHICLE_FEATURES["vehicle_id"],
                    "max_speed": DEFAULT_VEHICLE_FEATURES["max_speed"],
                    "n_positions": DEFAULT_VEHICLE_FEATURES["n_positions"],
                    "lat_min": DEFAULT_LAT,
                    "lat_max": DEFAULT_LAT,
                    "lat_mean": DEFAULT_LAT,
                    "lon_min": DEFAULT_LON,
                    "lon_max": DEFAULT_LON,
                    "lon_mean": DEFAULT_LON,
                    "bearing_min": DEFAULT_VEHICLE_FEATURES["bearing_min"],
                    "bearing_max": DEFAULT_VEHICLE_FEATURES["bearing_max"],
                    "hour": hour,
                    "day_of_week": date.weekday(),
                    **weather,
                    **holidays,
                    # Metadata for output
                    "_window_start": window_start,
                    "_route_short_name": trip_row.get("route_short_name"),
                    "_route_long_name": trip_row.get("route_long_name"),
                }
                all_rows.append(row)

    print(f"Built {len(all_rows)} prediction rows")

    if not all_rows:
        return pd.DataFrame()

    # Create DataFrame and predict in batches
    df = pd.DataFrame(all_rows)
    X = df[FEATURE_ORDER]

    # Convert trip_id to numeric for prediction
    X = X.copy()
    X["trip_id"] = pd.to_numeric(X["trip_id"], errors="coerce").fillna(0).astype(int)
    X["vehicle_id"] = pd.to_numeric(X["vehicle_id"], errors="coerce").fillna(0).astype(int)

    print("Running batch predictions...")
    probabilities = model.predict_proba(X)

    # Extract results
    predicted_classes = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    # Build output DataFrame
    forecast_df = pd.DataFrame({
        "window_start": df["_window_start"],
        "trip_id": df["trip_id"],
        "hour": df["hour"],
        "weekday": df["day_of_week"],
        "predicted_occupancy": predicted_classes,
        "confidence": confidences,
        "route_short_name": df["_route_short_name"],
        "route_long_name": df["_route_long_name"],
    })

    print(f"Generated {len(forecast_df)} forecasts")
    return forecast_df


def generate_hindcast_monitoring(model, fs) -> pd.DataFrame:
    """
    Compare predictions to actual data for model monitoring.

    Uses the most recent available date in the feature group, falling back
    from yesterday if data is not available (e.g., due to ingestion issues).

    Returns DataFrame with:
    - window_start, trip_id
    - actual_occupancy_mode, predicted_occupancy_mode
    - accuracy, precision, recall, f1_weighted, mae
    - model_version
    """
    print("\nGenerating hindcast for monitoring...")

    try:
        vehicle_fg = fs.get_feature_group("vehicle_trip_agg_fg", version=2)
        vehicle_df = vehicle_fg.read()
        print(f"Read {len(vehicle_df)} total records from vehicle_trip_agg_fg")

        # Debug: show available dates
        vehicle_df["window_start"] = pd.to_datetime(vehicle_df["window_start"])
        vehicle_df["_date"] = vehicle_df["window_start"].dt.date
        unique_dates = sorted(vehicle_df["_date"].unique())
        print(f"Available dates in data (last 5): {unique_dates[-5:] if len(unique_dates) > 5 else unique_dates}")

        if not unique_dates:
            print("No data available in feature group, skipping hindcast")
            return pd.DataFrame()

        # Try yesterday first, fall back to most recent available date
        yesterday = (datetime.now() - timedelta(days=1)).date()
        yesterday_str = str(yesterday)

        vehicle_df["_date_str"] = vehicle_df["_date"].astype(str)
        yesterday_df = vehicle_df[vehicle_df["_date_str"] == yesterday_str].copy()

        if yesterday_df.empty:
            # Fall back to most recent available date
            most_recent_date = unique_dates[-1]
            most_recent_str = str(most_recent_date)
            print(f"No data for {yesterday_str}, using most recent date: {most_recent_str}")
            yesterday_df = vehicle_df[vehicle_df["_date_str"] == most_recent_str].copy()
            # Update the reference date for weather/holiday lookups
            yesterday = most_recent_date
            yesterday_str = most_recent_str

        print(f"Using {len(yesterday_df)} actual records from {yesterday_str} for hindcast")

    except Exception as e:
        print(f"Error loading actual data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    # Get weather and holidays for the hindcast date
    try:
        weather_fg = fs.get_feature_group("weather_hourly_fg", version=1)
        weather_df = weather_fg.read()
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        weather_df["_date"] = weather_df["timestamp"].dt.date
        weather_df["hour"] = weather_df["timestamp"].dt.hour
        weather_yesterday = weather_df[weather_df["_date"] == yesterday]
    except Exception as e:
        print(f"Warning: Could not load weather: {e}")
        weather_yesterday = pd.DataFrame()

    try:
        holiday_fg = fs.get_feature_group("swedish_holidays_fg", version=1)
        holiday_df = holiday_fg.read()
        holiday_df["date"] = pd.to_datetime(holiday_df["date"]).dt.date
        holiday_row = holiday_df[holiday_df["date"] == yesterday]
        if not holiday_row.empty:
            holidays = {
                "is_work_free": int(holiday_row.iloc[0].get("is_work_free", False)),
                "is_red_day": int(holiday_row.iloc[0].get("is_red_day", False)),
                "is_day_before_holiday": int(holiday_row.iloc[0].get("is_day_before_holiday", False)),
            }
        else:
            holidays = {"is_work_free": 0, "is_red_day": 0, "is_day_before_holiday": 0}
    except Exception as e:
        print(f"Warning: Could not load holidays: {e}")
        holidays = {"is_work_free": 0, "is_red_day": 0, "is_day_before_holiday": 0}

    # Merge weather with vehicle data
    yesterday_df["hour"] = yesterday_df["window_start"].dt.hour

    if not weather_yesterday.empty:
        weather_cols = ["hour", "temperature_2m", "precipitation", "cloud_cover",
                       "wind_speed_10m", "rain", "snowfall"]
        weather_subset = weather_yesterday[weather_cols].drop_duplicates(subset=["hour"])
        yesterday_df = yesterday_df.merge(weather_subset, on="hour", how="left")
    else:
        for col in ["temperature_2m", "precipitation", "cloud_cover",
                   "wind_speed_10m", "rain", "snowfall"]:
            yesterday_df[col] = 10.0 if col == "temperature_2m" else 0.0

    # Add holiday features
    for k, v in holidays.items():
        yesterday_df[k] = v

    # Fill missing values
    yesterday_df = yesterday_df.fillna({
        "temperature_2m": 10.0, "precipitation": 0.0, "cloud_cover": 50.0,
        "wind_speed_10m": 5.0, "rain": 0.0, "snowfall": 0.0,
    })

    # Prepare features for prediction
    # Handle ID columns
    yesterday_df["trip_id"] = pd.to_numeric(yesterday_df["trip_id"], errors="coerce").fillna(0).astype(int)
    yesterday_df["vehicle_id"] = pd.to_numeric(yesterday_df["vehicle_id"], errors="coerce").fillna(0).astype(int)

    # Check we have all features
    missing_cols = [col for col in FEATURE_ORDER if col not in yesterday_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for prediction: {missing_cols}")
        return pd.DataFrame()

    X = yesterday_df[FEATURE_ORDER]

    print(f"Running hindcast predictions on {len(X)} records...")
    predictions = model.predict(X)

    # Calculate metrics
    y_true = yesterday_df["occupancy_mode"].values
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\nHindcast Metrics for {yesterday}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  MAE:       {mae:.4f}")

    # Build monitoring DataFrame
    monitor_df = pd.DataFrame({
        "window_start": yesterday_df["window_start"],
        "trip_id": yesterday_df["trip_id"].astype(str),
        "actual_occupancy_mode": y_true,
        "predicted_occupancy_mode": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1,
        "mae": mae,
        "model_version": MODEL_VERSION,
    })

    return monitor_df


def upload_forecast(fs, forecast_df: pd.DataFrame):
    """Upload forecast predictions to Hopsworks."""
    if forecast_df.empty:
        print("No forecasts to upload")
        return

    print(f"\nUploading {len(forecast_df)} forecasts to forecast_fg...")

    # Prepare DataFrame for Hopsworks
    forecast_df = forecast_df.copy()
    forecast_df["generated_at"] = datetime.now()

    # Ensure trip_id is string for consistency
    forecast_df["trip_id"] = forecast_df["trip_id"].astype(str)

    forecast_fg = fs.get_or_create_feature_group(
        name="forecast_fg",
        version=2,
        description="Trip-level occupancy predictions for next 2 days (v2: hour/weekday based)",
        primary_key=["trip_id", "hour", "weekday"],
        event_time="generated_at",
        online_enabled=False,
    )

    forecast_fg.insert(
        forecast_df,
        write_options={"wait_for_job": True}
    )
    print("Forecast upload complete")


def upload_monitoring(fs, monitor_df: pd.DataFrame):
    """Upload monitoring data to Hopsworks."""
    if monitor_df.empty:
        print("No monitoring data to upload")
        return

    print(f"\nUploading {len(monitor_df)} monitoring records to monitor_fg...")

    monitor_df = monitor_df.copy()
    monitor_df["generated_at"] = datetime.now()

    monitor_fg = fs.get_or_create_feature_group(
        name="monitor_fg",
        version=2,
        description="Model monitoring: actual vs predicted occupancy for hindcast analysis (v2: trip_id as string)",
        primary_key=["window_start", "trip_id"],
        event_time="generated_at",
        online_enabled=False,
    )

    monitor_fg.insert(
        monitor_df,
        write_options={"wait_for_job": True}
    )
    print("Monitoring upload complete")


def main():
    """Run the full inference pipeline."""
    print("=" * 60)
    print("HappySardines Trip-Based Inference Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Load model
    model = load_model_from_registry(project)

    # Get static trip data
    trips_df = get_static_trips(fs)

    if trips_df.empty:
        print("No trips found, cannot generate forecasts")
    else:
        # Generate and upload forecasts
        forecast_df = generate_trip_forecasts(model, trips_df)
        upload_forecast(fs, forecast_df)

    # Generate and upload monitoring data (hindcast)
    monitor_df = generate_hindcast_monitoring(model, fs)
    upload_monitoring(fs, monitor_df)

    print("\n" + "=" * 60)
    print("Inference Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
