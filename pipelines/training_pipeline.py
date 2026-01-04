"""
Training Pipeline for HappySardines Bus Occupancy Prediction

This pipeline:
1. Connects to Hopsworks Feature Store
2. Creates a Feature View joining vehicle, weather, and holiday data
3. Splits data by date (time-series appropriate)
4. Trains an XGBoost Classifier for occupancy prediction (4 classes: 0-3)
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

# XGBoost hyperparameters (inspired by WheelyFunTimes, tuned for our 4-class problem)
XGBOOST_PARAMS = {
    "tree_method": "hist",
    "enable_categorical": True,
    "max_depth": 7,
    "learning_rate": 0.02,
    "n_estimators": 150,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "objective": "multi:softmax",
    "num_class": 4,
    "random_state": 42,
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

# Target variable
TARGET = "occupancy_mode"


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

    print(f"  - vehicle_trip_agg_fg (v2)")
    print(f"  - weather_hourly_fg (v1)")
    print(f"  - swedish_holidays_fg (v1)")

    return vehicle_fg, weather_fg, holiday_fg


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


def fetch_training_data_manual(vehicle_fg, weather_fg, holiday_fg):
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
    feature_cols = VEHICLE_FEATURES + WEATHER_FEATURES + HOLIDAY_FEATURES

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
    """Compute sample weights to handle class imbalance."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))

    print(f"  Class weights: {weight_dict}")

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

    y_pred = model.predict(X_test)

    # Calculate metrics (weighted for class imbalance)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    print(f"\n  Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f} (weighted)")
    print(f"    Recall:    {recall:.4f} (weighted)")
    print(f"    F1 Score:  {f1:.4f} (weighted)")

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
        description="XGBoost Classifier for bus occupancy prediction (4 classes: 0-3)",
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
    vehicle_fg, weather_fg, holiday_fg = get_feature_groups(fs)

    # Use manual data fetch and join (more reliable than Feature View for this use case)
    print("\nFetching and joining training data...")
    df = fetch_training_data_manual(vehicle_fg, weather_fg, holiday_fg)
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
                    description="XGBoost Classifier for bus occupancy prediction (4 classes: 0-3)",
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
