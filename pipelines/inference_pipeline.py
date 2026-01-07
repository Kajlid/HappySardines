"""
Inference Pipeline for HappySardines Bus Occupancy Prediction

This pipeline:
1. Loads the trained XGBoost model from Hopsworks Model Registry
2. Generates predictions across a geographic grid for the next 48 hours
3. Stores predictions in Hopsworks Feature Store for UI consumption

The pre-computed predictions enable fast heat map rendering in the UI
without making on-demand predictions for every grid point.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import hopsworks
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from util import fetch_weather_forecast, get_holiday_features_for_date

load_dotenv()

# Grid configuration - bounds derived from actual GTFS stop locations (3119 stops)
# Run ui/get_boundaries.py to recalculate if needed
GRID_CONFIG = {
    "min_lat": 56.6414,
    "max_lat": 58.8654,
    "min_lon": 14.6144,
    "max_lon": 16.9578,
    "lat_steps": 15,
    "lon_steps": 20,
}

# Feature order - must match training pipeline exactly (occupancy_xgboost_model_new v4)
# Model v4 was trained on full vehicle_trip_agg_fg features including lat/lon bounds and bearing
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

# Default vehicle features (approximate averages from training data)
# For inference without real trip data, we set lat/lon bounds equal to the point
# and use neutral bearing values
DEFAULT_VEHICLE_FEATURES = {
    "trip_id": 0,          # placeholder
    "vehicle_id": 0,       # placeholder
    "max_speed": 45.0,
    "n_positions": 30,
    "bearing_min": 0.0,    # neutral bearing
    "bearing_max": 360.0,  # full range (stationary/unknown direction)
}


def load_model_from_registry():
    """Load the latest model from Hopsworks Model Registry."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    mr = project.get_model_registry()

    print("Fetching model from registry...")
    model_entry = mr.get_model("occupancy_xgboost_model_new", version=None)
    print(f"Downloading model version {model_entry.version}...")
    model_dir = model_entry.download()

    print("Loading XGBoost model...")
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, "model.json"))

    return model, project


def generate_grid_points():
    """Generate grid of lat/lon points across the region."""
    lats = np.linspace(
        GRID_CONFIG["min_lat"],
        GRID_CONFIG["max_lat"],
        GRID_CONFIG["lat_steps"]
    )
    lons = np.linspace(
        GRID_CONFIG["min_lon"],
        GRID_CONFIG["max_lon"],
        GRID_CONFIG["lon_steps"]
    )

    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append({"lat": lat, "lon": lon})

    return grid_points


def get_weather_for_grid(target_datetime: datetime):
    """
    Fetch weather forecast for the grid center point.
    Uses the same weather for all grid points (region is small enough).
    """
    center_lat = (GRID_CONFIG["min_lat"] + GRID_CONFIG["max_lat"]) / 2
    center_lon = (GRID_CONFIG["min_lon"] + GRID_CONFIG["max_lon"]) / 2

    try:
        weather_df = fetch_weather_forecast(center_lat, center_lon, forecast_days=3)
        if weather_df is not None and not weather_df.empty:
            # Find closest hour in forecast
            target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            weather_df = weather_df.set_index("timestamp")

            if target_hour in weather_df.index:
                row = weather_df.loc[target_hour]
                return {
                    "temperature_2m": float(row.get("temperature_2m", 10.0)),
                    "precipitation": float(row.get("precipitation", 0.0)),
                    "cloud_cover": float(row.get("cloud_cover", 50.0)),
                    "wind_speed_10m": float(row.get("wind_speed_10m", 5.0)),
                    "snowfall": float(row.get("snowfall", 0.0)),
                    "rain": float(row.get("rain", 0.0)),
                }
    except Exception as e:
        print(f"Warning: Failed to fetch weather, using defaults: {e}")

    # Fallback defaults
    return {
        "temperature_2m": 10.0,
        "precipitation": 0.0,
        "cloud_cover": 50.0,
        "wind_speed_10m": 5.0,
        "snowfall": 0.0,
        "rain": 0.0,
    }


def get_holidays_for_date(target_date):
    """Get holiday features for a specific date."""
    try:
        holidays = get_holiday_features_for_date(target_date)
        return {
            "is_work_free": int(holidays.get("is_work_free", False)),
            "is_red_day": int(holidays.get("is_red_day", False)),
            "is_day_before_holiday": int(holidays.get("is_day_before_holiday", False)),
        }
    except Exception as e:
        print(f"Warning: Failed to fetch holidays, using defaults: {e}")
        return {
            "is_work_free": 0,
            "is_red_day": 0,
            "is_day_before_holiday": 0,
        }


def generate_batch_predictions(
    model,
    grid_points: list,
    target_datetime: datetime,
    weather: dict,
    holidays: dict,
) -> pd.DataFrame:
    """
    Generate predictions for all grid points at a specific time.

    Returns DataFrame with columns:
    - lat, lon, hour, day_of_week, date
    - predicted_class, confidence
    - prob_0 through prob_6 (class probabilities)
    """
    records = []

    hour = target_datetime.hour
    day_of_week = target_datetime.weekday()
    date = target_datetime.date()

    # Build feature matrix for batch prediction
    features_list = []
    for point in grid_points:
        lat, lon = point["lat"], point["lon"]
        features = {
            **DEFAULT_VEHICLE_FEATURES,
            # For grid predictions, lat/lon bounds equal the point (no trip movement)
            "lat_min": lat,
            "lat_max": lat,
            "lat_mean": lat,
            "lon_min": lon,
            "lon_max": lon,
            "lon_mean": lon,
            "hour": hour,
            "day_of_week": day_of_week,
            **weather,
            **holidays,
        }
        features_list.append(features)

    # Create DataFrame and predict in batch
    X = pd.DataFrame(features_list)[FEATURE_ORDER]
    probabilities = model.predict_proba(X)

    # Build output records
    for i, point in enumerate(grid_points):
        probs = probabilities[i]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        record = {
            "lat": point["lat"],
            "lon": point["lon"],
            "hour": hour,
            "day_of_week": day_of_week,
            "date": date,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

        # Add individual class probabilities
        for j in range(len(probs)):
            record[f"prob_{j}"] = float(probs[j])

        records.append(record)

    return pd.DataFrame(records)


def run_inference_pipeline(
    hours_ahead: int = 48,
    upload_predictions: bool = True,
) -> pd.DataFrame:
    """
    Run the full inference pipeline.

    Args:
        hours_ahead: Number of hours to generate predictions for (default 48)
        upload_predictions: Whether to upload predictions to Hopsworks

    Returns:
        DataFrame with all predictions
    """
    print("=" * 60)
    print("HappySardines Inference Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load model
    model, project = load_model_from_registry()

    # Generate grid points
    grid_points = generate_grid_points()
    print(f"Generated {len(grid_points)} grid points")

    # Generate predictions for each hour
    all_predictions = []
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

    for h in range(hours_ahead):
        target_time = start_time + timedelta(hours=h)
        print(f"Generating predictions for {target_time.isoformat()}...")

        # Get weather and holidays for this time
        weather = get_weather_for_grid(target_time)
        holidays = get_holidays_for_date(target_time.date())

        # Generate predictions
        predictions = generate_batch_predictions(
            model, grid_points, target_time, weather, holidays
        )
        all_predictions.append(predictions)

    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Add timestamp for when predictions were generated
    predictions_df["generated_at"] = datetime.now()

    print(f"\nGenerated {len(predictions_df)} total predictions")
    print(f"Predictions shape: {predictions_df.shape}")

    # Upload to Hopsworks
    if upload_predictions:
        print("\nUploading predictions to Hopsworks...")
        upload_predictions_to_hopsworks(project, predictions_df)

    print("\n" + "=" * 60)
    print("Inference Pipeline Complete!")
    print("=" * 60)

    return predictions_df


def upload_predictions_to_hopsworks(project, predictions_df: pd.DataFrame):
    """Upload predictions to Hopsworks Feature Store."""
    fs = project.get_feature_store()

    # Convert date to datetime for Hopsworks compatibility
    predictions_df = predictions_df.copy()
    predictions_df["prediction_time"] = pd.to_datetime(
        predictions_df["date"].astype(str) + " " + predictions_df["hour"].astype(str) + ":00:00"
    )

    # Get or create feature group
    heatmap_fg = fs.get_or_create_feature_group(
        name="heatmap_predictions_fg",
        version=1,
        description="Pre-computed occupancy predictions for heat map visualization",
        primary_key=["lat", "lon", "prediction_time"],
        event_time="generated_at",
        online_enabled=True,  # Enable online serving for fast UI reads
    )

    # Insert predictions
    heatmap_fg.insert(
        predictions_df,
        write_options={"wait_for_job": True}
    )
    print(f"Uploaded {len(predictions_df)} predictions to heatmap_predictions_fg")


if __name__ == "__main__":
    predictions = run_inference_pipeline()
    print(f"\nSample predictions:\n{predictions.head(10)}")
