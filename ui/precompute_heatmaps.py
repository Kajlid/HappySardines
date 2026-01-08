"""
Precompute heatmap grid cells for all hour/weekday combinations.

This script generates GeoJSON grid cells from occupancy predictions
and uploads them to Hopsworks Feature Store for UI consumption.

Each grid cell is a rectangle colored by the model's prediction for that location.
No interpolation - what you see is exactly what the model predicts.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

import hopsworks

from predictor import predict_occupancy_batch, load_model
from weather import get_weather_for_prediction
from holidays import get_holiday_features
from contours import grid_to_cells_geojson, save_contours_to_file, load_contours_from_file


# Map bounds derived from actual GTFS stop locations (3119 stops)
# Run ui/get_boundaries.py to recalculate if needed
BOUNDS = {"min_lat": 56.6414, "max_lat": 58.8654, "min_lon": 14.6144, "max_lon": 16.9578}

# Grid resolution for predictions
# 20x25 = 500 points - faster generation (~5 min), good for regional overview
# Each cell is ~12km x 6km
LAT_STEPS = 20
LON_STEPS = 25

# Hours of interest
HOURS = list(range(5, 24))  # 5:00-23:00

def get_weekdays_to_compute():
    """Get weekdays for today and tomorrow."""
    from datetime import timedelta
    today = datetime.now().weekday()
    tomorrow = (datetime.now() + timedelta(days=1)).weekday()
    return [today, tomorrow]

# Output files (for local fallback)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTOURS_FILE = os.path.join(OUTPUT_DIR, "precomputed_contours.json")


def upload_heatmaps_to_hopsworks(contours: dict, max_retries: int = 3):
    """
    Upload heatmap GeoJSON data to Hopsworks Feature Store.

    Each time slot (hour, weekday) gets stored as one row with the full
    GeoJSON as a string. This allows fast retrieval by the UI.
    """
    import time

    print("\nConnecting to Hopsworks for upload...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Build DataFrame with one row per time slot
    records = []
    generated_at = datetime.now()

    for (hour, weekday), geojson in contours.items():
        records.append({
            "hour": int(hour),
            "weekday": int(weekday),
            "geojson": json.dumps(geojson),  # Store as JSON string
            "generated_at": generated_at,
        })

    df = pd.DataFrame(records)
    print(f"Prepared {len(df)} rows for upload")

    # Get or create the feature group (version 2 - v1 had Kafka issues)
    heatmap_fg = fs.get_or_create_feature_group(
        name="heatmap_geojson_fg",
        version=2,
        description="Pre-computed heatmap GeoJSON for UI visualization",
        primary_key=["hour", "weekday"],
        event_time="generated_at",
        online_enabled=True,  # Enable online serving for fast reads
    )

    # Insert with retry logic
    for attempt in range(max_retries):
        try:
            heatmap_fg.insert(df, write_options={"wait_for_job": True})
            print(f"Uploaded {len(df)} heatmap time slots to Hopsworks")
            return project
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"Upload failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Upload failed after {max_retries} attempts: {e}")
                print("Data saved locally - you can retry upload with --upload-only")
                raise

    return project


def main():
    import sys
    print("Loading model...")
    load_model()  # Pre-load model once

    lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], LAT_STEPS)
    lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], LON_STEPS)

    # Pre-compute all grid locations once
    all_locations = [(lat, lon) for lat in lats for lon in lons]

    # Contours dictionary: {(hour, weekday): GeoJSON FeatureCollection}
    # Calculate cell size based on grid resolution
    lat_step = (BOUNDS["max_lat"] - BOUNDS["min_lat"]) / (LAT_STEPS - 1)
    lon_step = (BOUNDS["max_lon"] - BOUNDS["min_lon"]) / (LON_STEPS - 1)

    contours = {}

    # Only compute for today and tomorrow
    weekdays = get_weekdays_to_compute()
    total = len(HOURS) * len(weekdays)
    count = 0

    print(f"Precomputing grid cells for {total} time slots (today={weekdays[0]}, tomorrow={weekdays[1]})...")
    print(f"Grid: {LAT_STEPS}x{LON_STEPS} = {LAT_STEPS * LON_STEPS} cells per slot")
    print(f"Cell size: {lat_step:.4f}° lat x {lon_step:.4f}° lon (~{lat_step*111:.1f}km x {lon_step*111*0.55:.1f}km)")

    for hour in HOURS:
        for weekday in weekdays:
            count += 1
            dt = datetime.now().replace(hour=hour)

            # Weather and holiday features (use region centroid)
            center_lat = (BOUNDS["min_lat"] + BOUNDS["max_lat"]) / 2
            center_lon = (BOUNDS["min_lon"] + BOUNDS["max_lon"]) / 2
            weather = get_weather_for_prediction(center_lat, center_lon, dt)
            holidays = get_holiday_features(dt)

            # Batch prediction - MUCH faster than individual calls
            try:
                results = predict_occupancy_batch(
                    all_locations, hour, weekday, weather, holidays
                )
            except Exception as e:
                print(f"  Error in batch prediction: {e}")
                results = [(0, 0.0)] * len(all_locations)

            # Build prediction data and count classes
            prediction_data = []
            class_counts = {i: 0 for i in range(7)}
            for (lat, lon), (pred_class, _) in zip(all_locations, results):
                prediction_data.append([lat, lon, pred_class])
                class_counts[pred_class] += 1

            # Convert to grid cell GeoJSON (no interpolation)
            geojson = grid_to_cells_geojson(prediction_data, lat_step, lon_step)
            contours[(hour, weekday)] = geojson

            # Show class distribution
            non_zero = {k: v for k, v in class_counts.items() if v > 0 and k != 0}
            dist_str = ", ".join(f"c{k}:{v}" for k, v in sorted(non_zero.items()))
            print(f"[{count}/{total}] hour={hour:02d}, weekday={weekday}: {len(prediction_data)} cells | {dist_str or 'all class 0'}")

    # Save local copy FIRST (so we don't lose work if upload fails)
    save_contours_to_file(contours, CONTOURS_FILE)
    print(f"Local backup saved to {CONTOURS_FILE}")

    # Upload to Hopsworks Feature Store (unless skipped)
    if "--skip-upload" not in sys.argv:
        try:
            upload_heatmaps_to_hopsworks(contours)
            print("\nDone! Heatmaps uploaded to Hopsworks Feature Store")
        except Exception as e:
            print(f"\nHopsworks upload failed: {e}")
            print("Local JSON saved successfully - UI will use fallback")
    else:
        print("\nSkipping Hopsworks upload (--skip-upload flag)")
        print("Local JSON saved successfully")


def upload_only():
    """Upload existing JSON file to Hopsworks without recomputing."""
    print("Loading existing contours from JSON...")
    contours = load_contours_from_file(CONTOURS_FILE)
    if not contours:
        print(f"No contours found in {CONTOURS_FILE}")
        return
    print(f"Loaded {len(contours)} time slots from JSON")
    upload_heatmaps_to_hopsworks(contours)
    print("\nDone! Existing heatmaps uploaded to Hopsworks")


if __name__ == "__main__":
    import sys
    if "--upload-only" in sys.argv:
        upload_only()
    else:
        main()
