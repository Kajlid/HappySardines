import hopsworks
import os
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two points using haversine formula."""
    R = 6371000  # Earth's radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def load_static_trip_info():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    project = hopsworks.login(project=project_name, api_key_value=api_key)

    fs = project.get_feature_store()
    fg = fs.get_feature_group("static_trip_info_fg", version=1)

    df = fg.read()
    return df


def load_static_stops_info():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    project = hopsworks.login(project=project_name, api_key_value=api_key)

    fs = project.get_feature_store()
    fg = fs.get_feature_group("static_trip_and_stops_info_fg", version=1)

    df = fg.read()
    return df


def time_to_seconds(t):
    """Convert time string (HH:MM:SS) to seconds since midnight."""
    if t is None:
        return None
    try:
        h, m, s = map(int, str(t).split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def find_nearest_trip(lat, lon, datetime_obj, static_trip_and_stops_df, max_radius_m=500):
    """
    Find the nearest trip to a given location and time.

    Uses haversine distance and filters by time window.

    Args:
        lat, lon: Location to search near
        datetime_obj: Target datetime
        static_trip_and_stops_df: DataFrame with trip and stop info
        max_radius_m: Maximum search radius in meters (default 500m)

    Returns:
        Dict with trip info or None if no nearby trip found
    """
    if static_trip_and_stops_df is None or static_trip_and_stops_df.empty:
        return None

    target_s = datetime_obj.hour * 3600 + datetime_obj.minute * 60

    # Compute distance to each stop
    df = static_trip_and_stops_df.copy()

    # Check if required columns exist
    if "stop_lat" not in df.columns or "stop_lon" not in df.columns:
        return None

    df["distance_m"] = df.apply(
        lambda r: haversine_distance(lat, lon, r["stop_lat"], r["stop_lon"]),
        axis=1
    )

    # Geographic filter
    nearby = df[df["distance_m"] <= max_radius_m]
    if nearby.empty:
        # Try with larger radius
        nearby = df[df["distance_m"] <= max_radius_m * 2]
        if nearby.empty:
            return None

    # Build time window check if arrival/departure times are available
    if "arrival_time" in nearby.columns and "departure_time" in nearby.columns:
        nearby = nearby.copy()
        nearby["arr_s"] = nearby["arrival_time"].apply(time_to_seconds)
        nearby["dep_s"] = nearby["departure_time"].apply(time_to_seconds)

        # Keep trips where we're near a scheduled stop time
        time_filtered = nearby[
            (nearby["arr_s"].notna()) &
            ((nearby["arr_s"] <= target_s + 3600) & (nearby["arr_s"] >= target_s - 3600))
        ]

        if not time_filtered.empty:
            nearby = time_filtered

    # Choose the one whose stop is closest to the click
    best = nearby.sort_values("distance_m").iloc[0]

    return {
        "trip_id": best.get("trip_id"),
        "route_short_name": best.get("route_short_name"),
        "route_long_name": best.get("route_long_name"),
        "trip_headsign": best.get("trip_headsign"),
        "closest_stop": best.get("stop_name"),
        "closest_stop_headsign": best.get("stop_headsign"),
        "distance_m": round(best["distance_m"]),
    }


def find_closest_stop(lat, lon, trip_id, stops_df):
    """
    Returns the closest stop to a given lat/lon for the specified trip_id.

    Returns tuple of (stop_name, stop_headsign) or (None, None) if not found.
    """
    if stops_df is None or stops_df.empty:
        return None, None

    # Filter stops for this trip
    trip_stops = stops_df[stops_df["trip_id"] == trip_id]
    if trip_stops.empty:
        return None, None

    # Compute distances using haversine
    distances = trip_stops.apply(
        lambda r: haversine_distance(lat, lon, r["stop_lat"], r["stop_lon"]),
        axis=1
    )

    closest_stop = trip_stops.loc[distances.idxmin()]

    return closest_stop.get("stop_name"), closest_stop.get("stop_headsign")


def load_trip_forecasts(fs, hour, weekday):
    """
    Load trip forecasts from forecast_fg for a specific hour and weekday.

    Returns DataFrame with trip predictions or empty DataFrame if not available.
    """
    try:
        forecast_fg = fs.get_feature_group("forecast_fg", version=1)
        df = forecast_fg.read()

        if df.empty:
            return df

        # Filter to matching hour and weekday
        df["window_start"] = pd.to_datetime(df["window_start"])
        df["hour"] = df["window_start"].dt.hour
        df["weekday"] = df["window_start"].dt.weekday

        filtered = df[(df["hour"] == hour) & (df["weekday"] == weekday)]
        return filtered

    except Exception as e:
        print(f"Could not load trip forecasts: {e}")
        return pd.DataFrame()
