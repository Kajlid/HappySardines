"""
Script to query actual bus stop boundaries from Hopsworks.
This derives the real geographic coverage from actual GTFS stop data.
"""

import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def get_stop_boundaries():
    """
    Query Hopsworks for all bus stops and compute the bounding box.
    Returns dict with min/max lat/lon.
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not set")

    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()

    print("Reading static_trip_and_stops_info_fg...")
    fg = fs.get_feature_group("static_trip_and_stops_info_fg", version=1)
    df = fg.read()

    print(f"Loaded {len(df)} records")

    # Extract unique stop coordinates
    stops = df[["stop_lat", "stop_lon"]].drop_duplicates()
    print(f"Found {len(stops)} unique stop locations")

    # Compute bounds
    min_lat = stops["stop_lat"].min()
    max_lat = stops["stop_lat"].max()
    min_lon = stops["stop_lon"].min()
    max_lon = stops["stop_lon"].max()

    # Add small buffer (0.02 degrees ~ 2km)
    buffer = 0.02
    bounds = {
        "min_lat": min_lat - buffer,
        "max_lat": max_lat + buffer,
        "min_lon": min_lon - buffer,
        "max_lon": max_lon + buffer,
    }

    # Also compute centroid
    centroid_lat = stops["stop_lat"].mean()
    centroid_lon = stops["stop_lon"].mean()

    return {
        "bounds": bounds,
        "raw_bounds": {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        },
        "centroid": {"lat": centroid_lat, "lon": centroid_lon},
        "num_stops": len(stops),
    }


if __name__ == "__main__":
    result = get_stop_boundaries()

    print("\n" + "=" * 50)
    print("ACTUAL STOP BOUNDARIES (from GTFS data)")
    print("=" * 50)

    print(f"\nNumber of unique stops: {result['num_stops']}")

    print(f"\nRaw bounds (exact):")
    raw = result["raw_bounds"]
    print(f"  Latitude:  {raw['min_lat']:.4f} to {raw['max_lat']:.4f}")
    print(f"  Longitude: {raw['min_lon']:.4f} to {raw['max_lon']:.4f}")

    print(f"\nBounds with buffer (for heatmap):")
    bounds = result["bounds"]
    print(f"  Latitude:  {bounds['min_lat']:.4f} to {bounds['max_lat']:.4f}")
    print(f"  Longitude: {bounds['min_lon']:.4f} to {bounds['max_lon']:.4f}")

    print(f"\nCentroid:")
    c = result["centroid"]
    print(f"  {c['lat']:.4f}, {c['lon']:.4f}")

    print("\n" + "=" * 50)
    print("PYTHON CODE (copy this to update BOUNDS):")
    print("=" * 50)
    print(f"""
BOUNDS = {{
    "min_lat": {bounds['min_lat']:.4f},
    "max_lat": {bounds['max_lat']:.4f},
    "min_lon": {bounds['min_lon']:.4f},
    "max_lon": {bounds['max_lon']:.4f},
}}
""")
