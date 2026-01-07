"""
Precompute heatmap contours for all hour/weekday combinations.

This script generates GeoJSON contour polygons from occupancy predictions
and saves them to a JSON file for fast rendering in the UI.
"""

import os
import numpy as np
from datetime import datetime
from predictor import predict_occupancy, load_model
from weather import get_weather_for_prediction
from holidays import get_holiday_features
from contours import grid_to_contour_geojson, save_contours_to_file

# Use mock predictor for testing visualization with varied colors
USE_MOCK_PREDICTOR = True


def predict_occupancy_mock(lat, lon, hour, day_of_week, weather, holidays):
    """
    Mock prediction that produces varied predictions for testing visualization.
    Returns class 0-3 based on time of day and location.
    """
    import random
    random.seed(int(lat * 1000 + lon * 100 + hour))  # Deterministic based on inputs

    # Base prediction on time of day
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        # Rush hour
        if holidays.get("is_work_free") or holidays.get("is_red_day"):
            base_class = 1  # Holiday rush hour = many seats
        else:
            base_class = 2 if hour < 8 or hour > 17 else 3  # Peak = standing
    elif 10 <= hour <= 15:
        base_class = 1  # Midday = many seats
    else:
        base_class = 0  # Early/late = empty

    # Add location-based variation (urban centers are busier)
    # Linköping inner city: ~58.41, 15.62
    dist_to_center = ((lat - 58.41)**2 + (lon - 15.62)**2)**0.5
    if dist_to_center < 0.1:
        base_class = min(base_class + 1, 3)  # More crowded in city center
    elif dist_to_center > 0.5:
        base_class = max(base_class - 1, 0)  # Less crowded far from center

    # Add some random variation
    variation = random.choice([-1, 0, 0, 0, 1])  # Mostly no change
    predicted_class = max(0, min(6, base_class + variation))

    # Mock probabilities
    probabilities = [0.1] * 7
    probabilities[predicted_class] = 0.6
    confidence = 0.6

    return predicted_class, confidence, probabilities

# Map bounds derived from actual GTFS stop locations (3119 stops)
# Run ui/get_boundaries.py to recalculate if needed
BOUNDS = {"min_lat": 56.6414, "max_lat": 58.8654, "min_lon": 14.6144, "max_lon": 16.9578}

# Grid resolution for predictions (higher = more detailed contours, slower)
LAT_STEPS = 20
LON_STEPS = 25

# Hours of interest
HOURS = list(range(5, 24))  # 5:00-23:00
WEEKDAYS = list(range(7))    # 0=Monday ... 6=Sunday

# Output files
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTOURS_FILE = os.path.join(OUTPUT_DIR, "precomputed_contours.json")


def main():
    if USE_MOCK_PREDICTOR:
        print("Using MOCK predictor for visualization testing...")
        predictor_func = predict_occupancy_mock
    else:
        print("Loading model...")
        model = load_model()
        predictor_func = predict_occupancy

    lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], LAT_STEPS)
    lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], LON_STEPS)

    # Contours dictionary: {(hour, weekday): GeoJSON FeatureCollection}
    contours = {}

    total = len(HOURS) * len(WEEKDAYS)
    count = 0

    print(f"Precomputing contours for {total} time slots...")
    print(f"Grid: {LAT_STEPS}x{LON_STEPS} = {LAT_STEPS * LON_STEPS} points per slot")

    for hour in HOURS:
        for weekday in WEEKDAYS:
            count += 1
            dt = datetime.now().replace(hour=hour)

            # Weather and holiday features (use region centroid)
            center_lat = (BOUNDS["min_lat"] + BOUNDS["max_lat"]) / 2
            center_lon = (BOUNDS["min_lon"] + BOUNDS["max_lon"]) / 2
            weather = get_weather_for_prediction(center_lat, center_lon, dt)
            holidays = get_holiday_features(dt)

            # Generate predictions across the grid
            prediction_data = []
            for lat in lats:
                for lon in lons:
                    try:
                        pred_class, confidence, _ = predictor_func(
                            lat, lon, hour, weekday, weather, holidays
                        )
                        # Normalize 0-6 -> 0-1
                        intensity = pred_class / 6.0
                        prediction_data.append([lat, lon, intensity])
                    except Exception:
                        # Default to 0 (empty) on error
                        prediction_data.append([lat, lon, 0.0])

            # Convert to contour GeoJSON
            geojson = grid_to_contour_geojson(prediction_data, BOUNDS)
            contours[(hour, weekday)] = geojson

            n_features = len(geojson.get("features", []))
            print(f"[{count}/{total}] hour={hour:02d}, weekday={weekday}: {n_features} contour levels")

    # Save contours to file
    save_contours_to_file(contours, CONTOURS_FILE)
    print(f"\nDone! Contours saved to {CONTOURS_FILE}")


if __name__ == "__main__":
    main()
