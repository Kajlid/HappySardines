import hopsworks
import os
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def load_static_trip_info():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    project = hopsworks.login(project=project_name, api_key_value=api_key)
    
    fs = project.get_feature_store()
    fg = fs.get_feature_group("static_trip_info_fg", version=1)  # adjust version
    
    df = fg.read()
    return df

def load_static_stops_info():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    project = hopsworks.login(project=project_name, api_key_value=api_key)
    
    fs = project.get_feature_store()
    fg = fs.get_feature_group("static_trip_and_stops_info_fg", version=1)  # adjust version
    
    df = fg.read()
    return df

def time_to_seconds(t):
    if t is None:
        return None
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s

# For demo purposes only:
def find_nearest_trip(lat, lon, datetime_obj, static_trip_and_stops_df, max_radius_m=100):
    """
    Return the trip closest to the requested location/time.
    Currently just filters static trips; could be enhanced with stops & routing.
    """
    # For static data, we can only match by service_id/date/time if available
    # Here we just pick a random trip as placeholder
    if static_trip_and_stops_df.empty:
        return None
    
    target_s = datetime_obj.hour * 3600 + datetime_obj.minute * 60

    # Compute distance to each stop
    df = static_trip_and_stops_df.copy()
    df["distance_m"] = df.apply(
        lambda r: haversine(lat, lon, r["stop_lat"], r["stop_lon"]),
        axis=1
    )
    
    # geographic filter
    nearby = df[df["distance_m"] <= max_radius_m]
    if nearby.empty:
        return None
    
    # build time window check
    nearby["arr_s"] = nearby["arrival_time"].apply(time_to_seconds)
    nearby["dep_s"] = nearby["departure_time"].apply(time_to_seconds)
    
    # keep trips where we're between two stop times at some point
    time_filtered = nearby[
        (nearby["arr_s"] <= target_s) | (nearby["dep_s"] >= target_s)
    ]

    if time_filtered.empty:
        return None

    # choose the one whose stop is closest to the click
    best = time_filtered.sort_values("distance_m").iloc[0]
    
    return {
        "trip_id": best["trip_id"],
        "route_short_name": best.get("route_short_name"),
        "route_long_name": best.get("route_long_name"),
        "trip_headsign": best.get("trip_headsign"),
        "closest_stop": best.get("stop_name"),
        "closest_stop_headsign": best.get("stop_headsign"),
        "distance_m": round(best["distance_m"])
    }

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def find_closest_stop(lat, lon, trip_id, stops_df):
    """
    Returns the closest stop to a given lat/lon for the specified trip_id.
    """
    # Filter stops for this trip
    trip_stops = stops_df[stops_df["trip_id"] == trip_id]
    if trip_stops.empty:
        return None

    # Compute distances
    lat_array = trip_stops["stop_lat"].to_numpy()
    lon_array = trip_stops["stop_lon"].to_numpy()

    # distances = np.sqrt((lat_array - lat)**2 + (lon_array - lon)**2)
    distances = trip_stops.apply(
        lambda r: haversine(lat, lon, r["stop_lat"], r["stop_lon"]),
        axis=1
    )
    # idx_min = distances.argmin()
    # closest_stop = trip_stops.iloc[idx_min]
    closest_stop = trip_stops.iloc[distances.idxmin()]

    return closest_stop["stop_name"], closest_stop["stop_headsign"]