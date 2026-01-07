"""
Contour generation module for heatmap visualization.

Converts prediction grid data into GeoJSON polygons that can be rendered
as vector overlays on Folium maps. This provides zoom-independent visualization
similar to weather radar overlays.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
import json


# Color scheme: green (empty) -> yellow -> orange -> red (crowded)
CONTOUR_COLORS = [
    "#22c55e",  # 0.0-0.2: Green (empty/many seats)
    "#84cc16",  # 0.2-0.4: Light green
    "#eab308",  # 0.4-0.6: Yellow (few seats)
    "#f97316",  # 0.6-0.8: Orange (standing)
    "#ef4444",  # 0.8-1.0: Red (crowded)
]

# Contour levels (intensity thresholds)
CONTOUR_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _extract_polygons_from_contour(contour_set, level_idx):
    """
    Extract polygons from a contourf result for a specific level.
    Compatible with matplotlib 3.8+ which removed .collections attribute.
    """
    polygons = []

    # Try new API first (matplotlib 3.8+)
    if hasattr(contour_set, 'get_paths'):
        # New API: iterate through all paths
        all_paths = contour_set.get_paths()
        # In new API, paths are organized differently
        # We need to use allsegs instead
        pass

    # Use allsegs which works in both old and new matplotlib
    if hasattr(contour_set, 'allsegs'):
        if level_idx < len(contour_set.allsegs):
            segments = contour_set.allsegs[level_idx]
            for seg in segments:
                if len(seg) >= 4:
                    try:
                        poly = Polygon(seg)
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        if poly.is_valid and not poly.is_empty and poly.area > 0:
                            if isinstance(poly, MultiPolygon):
                                polygons.extend(poly.geoms)
                            else:
                                polygons.append(poly)
                    except Exception:
                        continue
    return polygons


def grid_to_contour_geojson(
    prediction_data: list,
    bounds: dict,
    interpolation_resolution: int = 100,
    fill_opacity: float = 0.35,
) -> dict:
    """
    Convert prediction grid data to GeoJSON FeatureCollection with filled contour polygons.

    Args:
        prediction_data: List of [lat, lon, intensity] where intensity is 0-1
        bounds: Dict with min_lat, max_lat, min_lon, max_lon
        interpolation_resolution: Number of points per axis for interpolation grid
        fill_opacity: Opacity for the fill color (0-1)

    Returns:
        GeoJSON FeatureCollection with colored polygon features
    """
    if not prediction_data or len(prediction_data) < 4:
        return _empty_feature_collection()

    # Extract coordinates and values
    points = np.array([[p[0], p[1]] for p in prediction_data])  # lat, lon
    values = np.array([p[2] for p in prediction_data])  # intensity

    # Create fine interpolation grid
    lat_fine = np.linspace(bounds["min_lat"], bounds["max_lat"], interpolation_resolution)
    lon_fine = np.linspace(bounds["min_lon"], bounds["max_lon"], interpolation_resolution)
    lon_grid, lat_grid = np.meshgrid(lon_fine, lat_fine)

    # Interpolate to fine grid (using lat,lon order for points)
    try:
        values_fine = griddata(
            points,  # (lat, lon) pairs
            values,
            (lat_grid, lon_grid),  # grid in (lat, lon) format
            method='cubic',
            fill_value=0.0
        )
    except Exception:
        # Fall back to linear if cubic fails
        values_fine = griddata(
            points,
            values,
            (lat_grid, lon_grid),
            method='linear',
            fill_value=0.0
        )

    # Clip values to valid range
    values_fine = np.clip(values_fine, 0.0, 1.0)

    # Generate contours using matplotlib (but don't display)
    fig, ax = plt.subplots(figsize=(10, 10))

    # contourf returns a QuadContourSet
    contour_set = ax.contourf(
        lon_grid, lat_grid, values_fine,
        levels=CONTOUR_LEVELS,
        extend='neither'
    )

    plt.close(fig)  # Don't display, just extract the polygons

    # Convert matplotlib contours to GeoJSON features
    features = []

    for level_idx in range(len(CONTOUR_LEVELS) - 1):
        if level_idx >= len(CONTOUR_COLORS):
            break

        color = CONTOUR_COLORS[level_idx]
        level_min = CONTOUR_LEVELS[level_idx]
        level_max = CONTOUR_LEVELS[level_idx + 1]

        # Extract polygons for this level
        polygons = _extract_polygons_from_contour(contour_set, level_idx)

        if not polygons:
            continue

        # Merge overlapping polygons at this level
        try:
            merged = unary_union(polygons)
            if merged.is_empty:
                continue
        except Exception:
            continue

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {
                "color": color,
                "fillOpacity": fill_opacity,
                "level_min": level_min,
                "level_max": level_max,
                "level_idx": level_idx,
            },
            "geometry": mapping(merged)
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def _empty_feature_collection() -> dict:
    """Return an empty GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": []
    }


def precompute_contours_for_all_times(
    prediction_func,
    bounds: dict,
    hours: list = None,
    weekdays: list = None,
    lat_steps: int = 20,
    lon_steps: int = 25,
) -> dict:
    """
    Precompute contour GeoJSON for all hour/weekday combinations.

    Args:
        prediction_func: Function(lat, lon, hour, weekday) -> intensity (0-1)
        bounds: Geographic bounds dict
        hours: List of hours to compute (default: 5-23)
        weekdays: List of weekdays to compute (default: 0-6)
        lat_steps: Number of latitude grid points
        lon_steps: Number of longitude grid points

    Returns:
        Dict mapping (hour, weekday) -> GeoJSON FeatureCollection
    """
    if hours is None:
        hours = list(range(5, 24))  # 5:00 to 23:00
    if weekdays is None:
        weekdays = list(range(7))  # Monday to Sunday

    # Generate grid points
    lats = np.linspace(bounds["min_lat"], bounds["max_lat"], lat_steps)
    lons = np.linspace(bounds["min_lon"], bounds["max_lon"], lon_steps)

    results = {}
    total = len(hours) * len(weekdays)
    count = 0

    for hour in hours:
        for weekday in weekdays:
            count += 1
            print(f"Generating contours {count}/{total}: hour={hour}, weekday={weekday}")

            # Generate predictions for this time slot
            prediction_data = []
            for lat in lats:
                for lon in lons:
                    try:
                        intensity = prediction_func(lat, lon, hour, weekday)
                        prediction_data.append([lat, lon, intensity])
                    except Exception:
                        prediction_data.append([lat, lon, 0.0])

            # Convert to contour GeoJSON
            geojson = grid_to_contour_geojson(prediction_data, bounds)
            results[(hour, weekday)] = geojson

    return results


def save_contours_to_file(contours: dict, filepath: str):
    """
    Save precomputed contours to a JSON file.

    The dict keys (hour, weekday) are converted to strings for JSON serialization.
    """
    # Convert tuple keys to string keys for JSON
    json_compatible = {
        f"{hour},{weekday}": geojson
        for (hour, weekday), geojson in contours.items()
    }

    with open(filepath, 'w') as f:
        json.dump(json_compatible, f)

    print(f"Saved contours to {filepath}")


def load_contours_from_file(filepath: str) -> dict:
    """
    Load precomputed contours from a JSON file.

    Returns dict mapping (hour, weekday) tuple -> GeoJSON FeatureCollection.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Convert string keys back to tuple keys
    return {
        tuple(map(int, key.split(','))): geojson
        for key, geojson in data.items()
    }
