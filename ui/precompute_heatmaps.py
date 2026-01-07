import numpy as np
import pickle
from datetime import datetime, timedelta
from predictor import predict_occupancy, load_model
from weather import get_weather_for_prediction
from holidays import get_holiday_features

# Map bounds and grid resolution
BOUNDS = {"min_lat": 57.8, "max_lat": 58.9, "min_lon": 14.5, "max_lon": 16.8}
LAT_STEPS = 10
LON_STEPS = 15

# Hours of interest
HOURS = list(range(5, 24))  # 5:00-23:00
WEEKDAYS = list(range(7))    # 0=Monday ... 6=Sunday

# Load model once
model = load_model()

lats = np.linspace(BOUNDS["min_lat"], BOUNDS["max_lat"], LAT_STEPS)
lons = np.linspace(BOUNDS["min_lon"], BOUNDS["max_lon"], LON_STEPS)

# Heatmaps dictionary: {(hour, weekday): [[lat, lon, intensity], ...]}
heatmaps = {}

print("Precomputing heatmaps...")

for hour in HOURS:
    for weekday in WEEKDAYS:
        dt = datetime.now().replace(hour=hour)
        # Weather and holiday features (use central lat/lon)
        weather = get_weather_for_prediction((BOUNDS["min_lat"] + BOUNDS["max_lat"]) / 2,
                                             (BOUNDS["min_lon"] + BOUNDS["max_lon"]) / 2,
                                             dt)
        holidays = get_holiday_features(dt)

        heatmap_data = []
        for lat in lats:
            for lon in lons:
                try:
                    pred_class, confidence, _ = predict_occupancy(
                        lat, lon, hour, weekday, weather, holidays
                    )
                    intensity = pred_class / 5.0
                    if intensity > 0.1:
                        heatmap_data.append([lat, lon, intensity])
                except Exception:
                    pass

        heatmaps[(hour, weekday)] = heatmap_data
        print(f"Generated heatmap for hour {hour}, weekday {weekday} with {len(heatmap_data)} points")

# Save to file
with open("ui/precomputed_heatmaps.pkl", "wb") as f:
    pickle.dump(heatmaps, f)

print("Precomputation done. Saved to precomputed_heatmaps.pkl")
