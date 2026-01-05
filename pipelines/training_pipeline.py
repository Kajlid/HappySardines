"""
Training Pipeline for HappySardines Bus Occupancy Prediction

This pipeline:
1. Connects to Hopsworks Feature Store
2. Creates a Feature View joining vehicle, weather, and holiday data
3. Splits data by date (time-series appropriate)
4. Trains an XGBoost Classifier for occupancy prediction (7 GTFS-RT classes: 0-6)
5. Evaluates with weighted metrics for class imbalance
6. Registers the model in Hopsworks Model Registry with lineage

Based on patterns from:
- mlfs-book (course textbook): Feature View + Model Registry patterns
- WheelyFunTimes: XGBoost Classifier for bus occupancy
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import hopsworks
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from haversine import haversine, Unit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

load_dotenv()

# Configuration
MODEL_NAME = "occupancy_xgboost_model"
FEATURE_VIEW_NAME = "occupancy_fv"
FEATURE_VIEW_VERSION = 1

# XGBoost hyperparameters
# Using softprob to get probabilities for threshold tuning
# num_class=7 to be robust for all GTFS-RT OccupancyStatus values (0-6)
# even though our current data only has classes 0-3
XGBOOST_PARAMS = {
    "tree_method": "hist",
    "enable_categorical": True,
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.1,
    "objective": "multi:softprob",
    "num_class": 7,  # GTFS-RT has 7 occupancy classes (0-6)
    "random_state": 42,
}

# Class weight multipliers for severe imbalance
# Classes 2-3 are very rare, boost their importance significantly
# Classes 4-6 not yet observed but included for robustness
CLASS_WEIGHT_MULTIPLIER = {
    0: 1.0,   # EMPTY (72%) - baseline
    1: 2.0,   # MANY_SEATS (26%) - slight boost
    2: 10.0,  # FEW_SEATS (1%) - significant boost
    3: 20.0,  # STANDING (0.4%) - heavy boost
    4: 25.0,  # CRUSHED_STANDING - not observed yet
    5: 30.0,  # FULL - not observed yet
    6: 1.0,   # NOT_ACCEPTING_PASSENGERS - not observed yet
}

# Features to use for training
VEHICLE_FEATURES = [
    "avg_speed",
    "max_speed",
    "speed_std",
    "n_positions",
    "lat_mean",
    "lon_mean",
    "hour",
    "day_of_week",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
]

HOLIDAY_FEATURES = [
    "is_work_free",
    "is_red_day",
    "is_day_before_holiday",
]

TRAFFIC_FEATURES = [
    "has_nearby_event",
    "num_nearby_traffic_events",
]

# Target variable
TARGET = "occupancy_mode"

# Distance threshold for "nearby" traffic events (meters)
TRAFFIC_EVENT_RADIUS_M = 500


def connect_to_hopsworks():
    """Connect to Hopsworks and return project, feature store, and model registry."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print(f"Connected to project: {project.name}")
    return project, fs, mr


def get_feature_groups(fs):
    """Retrieve feature groups from the feature store."""
    print("Retrieving feature groups...")

    vehicle_fg = fs.get_feature_group(name="vehicle_trip_agg_fg", version=2)
    weather_fg = fs.get_feature_group(name="weather_hourly_fg", version=1)
    holiday_fg = fs.get_feature_group(name="swedish_holidays_fg", version=1)
    traffic_fg = fs.get_feature_group(name="trafikverket_traffic_event_fg", version=2)

    print(f"  - vehicle_trip_agg_fg (v2)")
    print(f"  - weather_hourly_fg (v1)")
    print(f"  - swedish_holidays_fg (v1)")
    print(f"  - trafikverket_traffic_event_fg (v2)")

    return vehicle_fg, weather_fg, holiday_fg, traffic_fg


def create_feature_view(fs, vehicle_fg, weather_fg, holiday_fg):
    """
    Create or retrieve a Feature View that joins vehicle data with weather and holiday data.

    Join logic:
    - vehicle_trip_agg_fg has: date, hour, occupancy_mode (target)
    - weather_hourly_fg has: date, hour, weather features
    - swedish_holidays_fg has: date, holiday features

    We join on date+hour for weather, and date for holidays.
    """
    print("Creating/retrieving Feature View...")

    # Select features from each feature group
    vehicle_query = vehicle_fg.select([
        "trip_id", "vehicle_id", "window_start",
        "avg_speed", "max_speed", "speed_std", "n_positions",
        "lat_mean", "lon_mean",
        "hour", "day_of_week", "date",
        "occupancy_mode",  # target
    ])

    weather_query = weather_fg.select([
        "date", "hour",
        "temperature_2m", "precipitation", "cloud_cover", "wind_speed_10m",
    ])

    holiday_query = holiday_fg.select([
        "date",
        "is_work_free", "is_red_day", "is_day_before_holiday",
    ])

    # Join queries
    # First join vehicle with weather on date + hour
    joined_query = vehicle_query.join(
        weather_query,
        on=["date", "hour"],
        join_type="left"
    )

    # Then join with holidays on date
    joined_query = joined_query.join(
        holiday_query,
        on=["date"],
        join_type="left"
    )

    # Create or get Feature View
    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        description="Occupancy prediction features: vehicle metrics joined with weather and holidays",
        labels=[TARGET],
        query=joined_query,
    )

    print(f"Feature View '{FEATURE_VIEW_NAME}' ready (v{FEATURE_VIEW_VERSION})")
    return feature_view


def distance_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points."""
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)


def calculate_nearby_traffic_events(vehicle_df, traffic_df):
    """
    Calculate nearby traffic events for each vehicle trip (optimized version).

    Uses vectorized operations and date-based bucketing to avoid O(n²) complexity.
    For each trip, counts active traffic events within TRAFFIC_EVENT_RADIUS_M meters
    of the trip's mean location during the trip's window_start time.

    Returns DataFrame with has_nearby_event and num_nearby_traffic_events columns.
    """
    print("  Calculating nearby traffic events (optimized)...")

    if traffic_df is None or traffic_df.empty:
        print("    No traffic data available, setting defaults")
        vehicle_df["has_nearby_event"] = 0
        vehicle_df["num_nearby_traffic_events"] = 0
        return vehicle_df

    # Ensure timestamps are comparable
    vehicle_df = vehicle_df.copy()
    traffic_df = traffic_df.copy()

    if "window_start" in vehicle_df.columns:
        vehicle_df["window_start"] = pd.to_datetime(vehicle_df["window_start"])
    if "start_time" in traffic_df.columns:
        traffic_df["start_time"] = pd.to_datetime(traffic_df["start_time"])
    if "end_time" in traffic_df.columns:
        traffic_df["end_time"] = pd.to_datetime(traffic_df["end_time"])

    # Filter traffic events to only those with valid coordinates
    traffic_df = traffic_df.dropna(subset=["latitude", "longitude", "start_time", "end_time"])

    if traffic_df.empty:
        print("    No valid traffic events with coordinates")
        vehicle_df["has_nearby_event"] = 0
        vehicle_df["num_nearby_traffic_events"] = 0
        return vehicle_df

    print(f"    Processing {len(vehicle_df)} trips against {len(traffic_df)} traffic events...")

    # Get date range from vehicle data
    vehicle_df["_date"] = vehicle_df["window_start"].dt.date

    # Initialize result columns
    vehicle_df["has_nearby_event"] = 0
    vehicle_df["num_nearby_traffic_events"] = 0

    # Process by date to reduce comparisons
    unique_dates = vehicle_df["_date"].unique()
    total_nearby = 0

    for i, date in enumerate(unique_dates):
        if i % 10 == 0:
            print(f"    Processing date {i+1}/{len(unique_dates)}...")

        # Get trips for this date
        date_mask = vehicle_df["_date"] == date
        date_trips = vehicle_df.loc[date_mask]

        if date_trips.empty:
            continue

        # Convert date to datetime for comparison (must be tz-aware to match traffic_df)
        date_start = pd.Timestamp(date, tz='UTC')
        date_end = date_start + pd.Timedelta(days=1)

        # Get traffic events that could be active on this date
        # Event is potentially active if: start_time <= end_of_day AND end_time >= start_of_day
        active_mask = (traffic_df["start_time"] <= date_end) & (traffic_df["end_time"] >= date_start)
        day_events = traffic_df.loc[active_mask]

        if day_events.empty:
            continue

        # For each trip on this date, check against day's events
        for idx in date_trips.index:
            trip_time = vehicle_df.at[idx, "window_start"]
            trip_lat = vehicle_df.at[idx, "lat_mean"]
            trip_lon = vehicle_df.at[idx, "lon_mean"]

            if pd.isna(trip_lat) or pd.isna(trip_lon) or pd.isna(trip_time):
                continue

            # Filter to events active at this specific time
            time_mask = (day_events["start_time"] <= trip_time) & (day_events["end_time"] >= trip_time)
            active_events = day_events.loc[time_mask]

            if active_events.empty:
                continue

            # Vectorized distance calculation using haversine formula
            event_lats = active_events["latitude"].values
            event_lons = active_events["longitude"].values

            # Calculate distances using numpy (approximate haversine)
            # Convert to radians
            lat1_rad = np.radians(trip_lat)
            lon1_rad = np.radians(trip_lon)
            lat2_rad = np.radians(event_lats)
            lon2_rad = np.radians(event_lons)

            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = 6371000 * c  # Earth radius in meters

            # Count nearby events
            nearby_count = np.sum(distances <= TRAFFIC_EVENT_RADIUS_M)

            if nearby_count > 0:
                vehicle_df.at[idx, "has_nearby_event"] = 1
                vehicle_df.at[idx, "num_nearby_traffic_events"] = int(nearby_count)
                total_nearby += 1

    # Clean up temp column
    vehicle_df.drop(columns=["_date"], inplace=True)

    print(f"    {total_nearby} trips ({100*total_nearby/len(vehicle_df):.1f}%) have nearby traffic events")

    return vehicle_df


def fetch_training_data_manual(vehicle_fg, weather_fg, holiday_fg, traffic_fg):
    """
    Manually fetch and join data from feature groups.
    Use this if Feature View creation has issues.

    Returns joined DataFrame with features and target.
    """
    print("Fetching data manually from feature groups...")

    # Fetch vehicle data (main training data with target)
    print("  Reading vehicle_trip_agg_fg...")
    vehicle_df = vehicle_fg.read()
    print(f"    {len(vehicle_df)} rows")

    # Fetch weather data
    print("  Reading weather_hourly_fg...")
    weather_df = weather_fg.read()
    print(f"    {len(weather_df)} rows")

    # Fetch holiday data
    print("  Reading swedish_holidays_fg...")
    holiday_df = holiday_fg.read()
    print(f"    {len(holiday_df)} rows")

    # Fetch traffic data
    print("  Reading trafikverket_traffic_event_fg...")
    try:
        traffic_df = traffic_fg.read()
        print(f"    {len(traffic_df)} rows")
    except Exception as e:
        print(f"    Warning: Could not read traffic data: {e}")
        traffic_df = None

    # Prepare for joining
    # Ensure date columns are comparable
    vehicle_df["date"] = pd.to_datetime(vehicle_df["date"]).dt.date
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
    holiday_df["date"] = pd.to_datetime(holiday_df["date"]).dt.date

    # Select relevant weather columns and deduplicate
    weather_cols = ["date", "hour"] + WEATHER_FEATURES
    weather_subset = weather_df[weather_cols].drop_duplicates(subset=["date", "hour"])

    # Select relevant holiday columns and deduplicate
    holiday_cols = ["date"] + HOLIDAY_FEATURES
    holiday_subset = holiday_df[holiday_cols].drop_duplicates(subset=["date"])

    # Join vehicle with weather on date + hour
    print("  Joining vehicle with weather on (date, hour)...")
    joined_df = vehicle_df.merge(
        weather_subset,
        on=["date", "hour"],
        how="left"
    )

    # Join with holidays on date
    print("  Joining with holidays on (date)...")
    joined_df = joined_df.merge(
        holiday_subset,
        on=["date"],
        how="left"
    )

    # Calculate nearby traffic events
    joined_df = calculate_nearby_traffic_events(joined_df, traffic_df)

    print(f"  Final joined dataset: {len(joined_df)} rows")

    return joined_df


def prepare_data(df, test_start_date=None, test_ratio=0.2):
    """
    Prepare training and test data with time-based split.

    Args:
        df: Joined DataFrame with features and target
        test_start_date: Date string (YYYY-MM-DD) for test split. If None, uses ratio.
        test_ratio: Fraction of data for testing (used if test_start_date is None)

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Preparing train/test split...")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    print(f"  Rows with valid target: {len(df)}")

    # Feature columns
    feature_cols = VEHICLE_FEATURES + WEATHER_FEATURES + HOLIDAY_FEATURES + TRAFFIC_FEATURES

    # Check which features exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")

    print(f"  Using features: {available_features}")

    # Sort by date for proper time-based split
    if "window_start" in df.columns:
        df = df.sort_values("window_start")
    elif "date" in df.columns:
        df = df.sort_values("date")

    # Time-based split
    if test_start_date:
        test_start = pd.to_datetime(test_start_date).date()
        train_mask = df["date"] < test_start
        test_mask = df["date"] >= test_start

        train_df = df[train_mask]
        test_df = df[test_mask]
        print(f"  Split by date: train < {test_start_date}, test >= {test_start_date}")
    else:
        # Use last test_ratio of data for testing
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        print(f"  Split by ratio: {1-test_ratio:.0%} train, {test_ratio:.0%} test")

    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Extract features and target
    X_train = train_df[available_features].copy()
    X_test = test_df[available_features].copy()
    y_train = train_df[TARGET].astype(int).copy()
    y_test = test_df[TARGET].astype(int).copy()

    # Fill missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test

    # Convert boolean columns to int
    for col in HOLIDAY_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    # Class distribution
    print(f"\n  Class distribution (train):")
    for cls in sorted(y_train.unique()):
        count = (y_train == cls).sum()
        pct = count / len(y_train) * 100
        print(f"    Class {cls}: {count} ({pct:.1f}%)")

    return X_train, X_test, y_train, y_test


def compute_sample_weights(y_train):
    """Compute sample weights to handle class imbalance with custom multipliers."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    base_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    base_weight_dict = dict(zip(classes, base_weights))

    # Apply additional multipliers for severe imbalance
    weight_dict = {}
    for cls in classes:
        multiplier = CLASS_WEIGHT_MULTIPLIER.get(cls, 1.0)
        weight_dict[cls] = base_weight_dict[cls] * multiplier

    print(f"  Base class weights: {base_weight_dict}")
    print(f"  Adjusted class weights: {weight_dict}")

    sample_weights = np.array([weight_dict[y] for y in y_train])
    return sample_weights


def train_model(X_train, y_train, use_class_weights=True):
    """Train XGBoost Classifier with optional class weighting."""
    print("\nTraining XGBoost Classifier...")
    print(f"  Parameters: {XGBOOST_PARAMS}")

    model = XGBClassifier(**XGBOOST_PARAMS)

    if use_class_weights:
        sample_weights = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    print("  Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    print("\nEvaluating model...")

    # Get probabilities (since we use softprob objective)
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # Calculate metrics (weighted for class imbalance)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Also calculate per-class recall (important for rare classes)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "recall_class_0": float(per_class_recall[0]) if len(per_class_recall) > 0 else 0,
        "recall_class_1": float(per_class_recall[1]) if len(per_class_recall) > 1 else 0,
        "recall_class_2": float(per_class_recall[2]) if len(per_class_recall) > 2 else 0,
        "recall_class_3": float(per_class_recall[3]) if len(per_class_recall) > 3 else 0,
    }

    print(f"\n  Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f} (weighted)")
    print(f"    Recall:    {recall:.4f} (weighted)")
    print(f"    F1 Score:  {f1:.4f} (weighted)")
    print(f"\n  Per-class Recall (critical for rare classes):")
    class_names = ["EMPTY", "MANY_SEATS", "FEW_SEATS", "STANDING"]
    for i, name in enumerate(class_names):
        if i < len(per_class_recall):
            print(f"    Class {i} ({name}): {per_class_recall[i]:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(cm)

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics, y_pred


def plot_feature_importance(model, feature_names, save_path=None):
    """Plot and optionally save feature importance."""
    importance = model.feature_importances_

    # Sort by importance
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]

    print("\n  Feature Importance (gain):")
    for feat, imp in zip(sorted_features, sorted_importance):
        print(f"    {feat}: {imp:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importance[::-1])
    plt.yticks(range(len(sorted_features)), sorted_features[::-1])
    plt.xlabel("Feature Importance (Gain)")
    plt.title("XGBoost Feature Importance - Occupancy Prediction")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved feature importance plot to {save_path}")

    plt.close()


def save_model_local(model, model_dir):
    """Save model to local directory."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.json")
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")
    return model_path


def register_model(mr, model, metrics, feature_view, model_dir, X_test):
    """Register model in Hopsworks Model Registry."""
    print("\nRegistering model in Hopsworks Model Registry...")

    # Save model locally first
    save_model_local(model, model_dir)

    # Save feature importance plot
    feature_importance_path = os.path.join(model_dir, "feature_importance.png")
    plot_feature_importance(model, X_test.columns.tolist(), feature_importance_path)

    # Create model in registry
    hopsworks_model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        feature_view=feature_view,
        description="XGBoost Classifier for bus occupancy prediction (GTFS-RT classes 0-6)",
        input_example=X_test.iloc[:1].values,
    )

    # Upload model directory
    hopsworks_model.save(model_dir)

    print(f"  Model registered as '{MODEL_NAME}'")
    print(f"  Model version: {hopsworks_model.version}")

    return hopsworks_model


def run_training_pipeline(test_start_date=None, upload_model=True):
    """
    Main training pipeline entry point.

    Args:
        test_start_date: Date string for train/test split (e.g., "2025-12-01")
        upload_model: Whether to upload model to Hopsworks registry

    Returns:
        Trained model and metrics
    """
    print("=" * 60)
    print("HappySardines Occupancy Training Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Connect to Hopsworks
    project, fs, mr = connect_to_hopsworks()

    # Get feature groups
    vehicle_fg, weather_fg, holiday_fg, traffic_fg = get_feature_groups(fs)

    # Use manual data fetch and join (more reliable than Feature View for this use case)
    print("\nFetching and joining training data...")
    df = fetch_training_data_manual(vehicle_fg, weather_fg, holiday_fg, traffic_fg)
    X_train, X_test, y_train, y_test = prepare_data(df, test_start_date)
    feature_view = None

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Register model
    if upload_model:
        with tempfile.TemporaryDirectory() as model_dir:
            if feature_view:
                register_model(mr, model, metrics, feature_view, model_dir, X_test)
            else:
                # Save locally without feature view linkage
                save_model_local(model, model_dir)
                plot_feature_importance(model, X_test.columns.tolist(),
                                       os.path.join(model_dir, "feature_importance.png"))

                hopsworks_model = mr.python.create_model(
                    name=MODEL_NAME,
                    metrics=metrics,
                    description="XGBoost Classifier for bus occupancy prediction (GTFS-RT classes 0-6)",
                    input_example=X_test.iloc[:1].values,
                )
                hopsworks_model.save(model_dir)
                print(f"  Model registered as '{MODEL_NAME}' v{hopsworks_model.version}")
    else:
        print("\nSkipping model upload (upload_model=False)")

    print("\n" + "=" * 60)
    print(f"Training pipeline complete: {datetime.now().isoformat()}")
    print("=" * 60)

    return model, metrics


if __name__ == "__main__":
    # Default: use last 20% of data for testing
    # Can pass a specific date for time-based split
    import argparse

    parser = argparse.ArgumentParser(description="Train occupancy prediction model")
    parser.add_argument(
        "--test-start-date",
        type=str,
        default=None,
        help="Date for train/test split (YYYY-MM-DD). If not set, uses 80/20 ratio."
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading model to Hopsworks registry"
    )

    args = parser.parse_args()

    run_training_pipeline(
        test_start_date=args.test_start_date,
        upload_model=not args.no_upload
    )
