import hopsworks
import os
import numpy as np

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

def find_nearest_trip(lat, lon, datetime_obj, static_trip_df):
    """
    Return the trip closest to the requested location/time.
    Currently just filters static trips; could be enhanced with stops & routing.
    """
    # For static data, we can only match by service_id/date/time if available
    # Here we just pick a random trip as placeholder
    if len(static_trip_df) == 0:
        return None
    
    trip = static_trip_df.sample(1).iloc[0]  # pick 1 random trip for demo
    return {
        "trip_id": trip["trip_id"],
        "route_short_name": trip["route_short_name"],
        "route_long_name": trip["route_long_name"],
        "trip_headsign": trip.get("trip_headsign", None)
    }


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

    distances = np.sqrt((lat_array - lat)**2 + (lon_array - lon)**2)
    idx_min = distances.argmin()
    closest_stop = trip_stops.iloc[idx_min]

    return closest_stop["stop_name"]