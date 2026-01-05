"""
Model loading and prediction logic for HappySardines.

Loads the XGBoost model from Hopsworks Model Registry and makes predictions.
"""

import os
import numpy as np
import pandas as pd

# Global model cache
_model = None
_model_loaded = False

# Occupancy class labels with display info
OCCUPANCY_LABELS = {
    0: {
        "label": "Empty",
        "message": "Plenty of room - pick any seat!",
        "color": "green",
        "icon": "🟢"
    },
    1: {
        "label": "Many seats available",
        "message": "Lots of seats to choose from.",
        "color": "green",
        "icon": "🟢"
    },
    2: {
        "label": "Few seats available",
        "message": "Some seats left - you might need to look around.",
        "color": "yellow",
        "icon": "🟡"
    },
    3: {
        "label": "Standing room only",
        "message": "Expect to stand - pack your patience!",
        "color": "orange",
        "icon": "🟠"
    },
    4: {
        "label": "Crushed standing",
        "message": "Very crowded - consider waiting for the next one.",
        "color": "red",
        "icon": "🔴"
    },
    5: {
        "label": "Full",
        "message": "Bus is full - you may not get on.",
        "color": "red",
        "icon": "🔴"
    },
    6: {
        "label": "Not accepting passengers",
        "message": "Bus is not accepting passengers.",
        "color": "gray",
        "icon": "⚫"
    }
}

# Feature order expected by the model
# Must match training pipeline exactly
FEATURE_ORDER = [
    "avg_speed",
    "max_speed",
    "speed_std",
    "n_positions",
    "lat_mean",
    "lon_mean",
    "hour",
    "day_of_week",
    "temperature_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "is_work_free",
    "is_red_day",
    "is_day_before_holiday",
]

# Default values for vehicle features (we don't have real-time vehicle data)
# These are approximate averages from the training data
DEFAULT_VEHICLE_FEATURES = {
    "avg_speed": 20.0,      # typical urban bus speed (km/h)
    "max_speed": 45.0,      # typical max speed
    "speed_std": 12.0,      # typical speed variation
    "n_positions": 30,      # typical GPS points per trip window
}


def load_model():
    """
    Load model from Hopsworks Model Registry.

    Caches the model globally for reuse.
    """
    global _model, _model_loaded

    if _model_loaded:
        return _model

    try:
        import hopsworks
        from xgboost import XGBClassifier

        print("Connecting to Hopsworks...")
        project = hopsworks.login()
        mr = project.get_model_registry()

        print("Fetching model from registry...")
        model_entry = mr.get_model("occupancy_xgboost_model", version=None)  # Latest version

        print(f"Downloading model version {model_entry.version}...")
        model_dir = model_entry.download()

        print("Loading XGBoost model...")
        model = XGBClassifier()
        model.load_model(os.path.join(model_dir, "model.json"))

        _model = model
        _model_loaded = True
        print("Model loaded successfully!")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def predict_occupancy(lat, lon, hour, day_of_week, weather, holidays):
    """
    Predict occupancy for given inputs.

    Args:
        lat: Latitude
        lon: Longitude
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        weather: Dict with temperature_2m, precipitation, cloud_cover, wind_speed_10m
        holidays: Dict with is_work_free, is_red_day, is_day_before_holiday

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    model = load_model()

    # Assemble feature vector
    features = {
        # Vehicle features - use defaults
        "avg_speed": DEFAULT_VEHICLE_FEATURES["avg_speed"],
        "max_speed": DEFAULT_VEHICLE_FEATURES["max_speed"],
        "speed_std": DEFAULT_VEHICLE_FEATURES["speed_std"],
        "n_positions": DEFAULT_VEHICLE_FEATURES["n_positions"],

        # Location
        "lat_mean": lat,
        "lon_mean": lon,

        # Time
        "hour": hour,
        "day_of_week": day_of_week,

        # Weather
        "temperature_2m": weather.get("temperature_2m", 10.0),
        "precipitation": weather.get("precipitation", 0.0),
        "cloud_cover": weather.get("cloud_cover", 50.0),
        "wind_speed_10m": weather.get("wind_speed_10m", 5.0),

        # Holidays (convert bool to int)
        "is_work_free": int(holidays.get("is_work_free", False)),
        "is_red_day": int(holidays.get("is_red_day", False)),
        "is_day_before_holiday": int(holidays.get("is_day_before_holiday", False)),
    }

    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[FEATURE_ORDER]

    # Get prediction probabilities
    probabilities = model.predict_proba(X)[0]

    # Get predicted class (highest probability)
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return predicted_class, confidence, probabilities.tolist()


# Mock prediction for testing without Hopsworks
def predict_occupancy_mock(lat, lon, hour, day_of_week, weather, holidays):
    """
    Mock prediction for testing UI without model.
    """
    # Simple heuristic based on time
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        # Rush hour
        if holidays.get("is_work_free") or holidays.get("is_red_day"):
            predicted_class = 1  # Holiday rush hour = many seats
        else:
            predicted_class = 2 if hour < 8 or hour > 17 else 3  # Peak = standing
    elif 10 <= hour <= 15:
        predicted_class = 1  # Midday = many seats
    else:
        predicted_class = 0  # Early/late = empty

    # Mock probabilities
    probabilities = [0.1] * 7
    probabilities[predicted_class] = 0.6
    confidence = 0.6

    return predicted_class, confidence, probabilities


# For testing - use mock if model not available
def get_predictor():
    """Get the appropriate predictor function."""
    try:
        load_model()
        return predict_occupancy
    except Exception as e:
        print(f"Using mock predictor: {e}")
        return predict_occupancy_mock
