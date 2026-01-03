import hopsworks
import os
import pandas as pd
import requests
import tempfile
import zipfile
import py7zr
from haversine import haversine, Unit
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2
import xml.etree.ElementTree as ET
import numpy as np
import gc

sys.path.append(".")
from util import *

load_dotenv()

# Configuration
KODA_RT_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/otraf/VehiclePositions"
KODA_STATIC_API_BASE_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/otraf"
TRAFIKVERKET_API_URL = "https://api.trafikinfo.trafikverket.se/v2/data.xml"

TRAFIKVERKET_KEY = os.environ.get("TRAFIKVERKET_API_KEY")
KODA_KEY = os.environ.get("KODA_API_KEY")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

# KODA RT Data Fetching & Parsing
def fetch_koda_rt_data(date_str: str, hour: int) -> bytes:
    """Fetch GTFS-RT vehicle positions from KODA API for a specific date and hour."""
    url = f"{KODA_RT_API_BASE_URL}?date={date_str}&hour={hour}&key={KODA_KEY}"

    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200 and response.content:
            return response.content
        else:
            print(f"  Warning: HTTP {response.status_code} for {date_str} hour {hour}")
            return None
    except Exception as e:
        print(f"  Error fetching KODA RT data: {e}")
        return None

def parse_vehiclepositions(content: bytes, date: str, hour: int):
    """Parse GTFS-RT VehiclePositions protobuf into list of dict rows."""
    if not content:
        return []

    rows = []
    try:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(content)

        for entity in feed.entity:
            if not entity.HasField("vehicle"):
                continue

            v = entity.vehicle
            trip = getattr(v, 'trip', None)
            pos = getattr(v, 'position', None)

            ts = getattr(v, 'timestamp', None)
            ts_dt = pd.to_datetime(ts, unit='s') if ts else None

            vehicle_obj = getattr(v, 'vehicle', None) or v
            vehicle_id = getattr(vehicle_obj, 'id', None)

            row = {
                'feed_entity_id': getattr(entity, 'id', None),
                'date': date,
                'hour': hour,
                'timestamp': ts_dt,
                'vehicle_id': vehicle_id,
                'trip_id': getattr(trip, 'trip_id', None),
                'position_lat': getattr(pos, 'latitude', None),
                'position_lon': getattr(pos, 'longitude', None),
                'bearing': getattr(pos, 'bearing', None),
                'speed': getattr(pos, 'speed', None),
                'occupancy_status': getattr(v, 'occupancy_status', None),
            }
            rows.append(row)
    except Exception as e:
        print(f"  Error parsing protobuf: {e}")

    return rows

def fetch_koda_static_data(date_str: str, tmp_dir: str) -> str:
    """Download GTFS static ZIP file from KODA API."""
    url = f"{KODA_STATIC_API_BASE_URL}?date={date_str}&key={KODA_KEY}"
    zip_path = os.path.join(tmp_dir, f"gtfs_static_{date_str.replace('-', '_')}.zip")

    if os.path.exists(zip_path):
        return zip_path

    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200 and "application/zip" in response.headers.get("Content-Type", ""):
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return zip_path
        else:
            print(f"  Warning: Could not download static data for {date_str}")
            return None
    except Exception as e:
        print(f"  Error downloading static data: {e}")
        return None

def get_aggregated_temporal_trip_features(rt_df):
    """Aggregate real-time vehicle data into 1-minute windows."""
    if rt_df.empty:
        return pd.DataFrame()

    rt_df = rt_df.sort_values(["timestamp", "trip_id", "vehicle_id"])
    rt_df['window_start'] = rt_df['timestamp'].dt.floor('1min')

    vehicle_trip_1min_df = (
        rt_df
        .groupby(["trip_id", "window_start"], as_index=False)
        .agg(
            vehicle_id=("vehicle_id", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
            avg_speed=("speed", "mean"),
            max_speed=("speed", "max"),
            n_positions=("speed", "count"),
            speed_std=("speed", "std"),
            lat_min=("position_lat", "min"),
            lat_max=("position_lat", "max"),
            lat_mean=("position_lat", "mean"),
            lon_min=("position_lon", "min"),
            lon_max=("position_lon", "max"),
            lon_mean=("position_lon", "mean"),
            bearing_min=("bearing", "min"),
            bearing_max=("bearing", "max"),
            occupancy_mode=("occupancy_status", lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        )
    )

    vehicle_trip_1min_df["date"] = vehicle_trip_1min_df["window_start"].dt.date
    vehicle_trip_1min_df["hour"] = vehicle_trip_1min_df["window_start"].dt.hour
    vehicle_trip_1min_df["day_of_week"] = vehicle_trip_1min_df["window_start"].dt.weekday
    vehicle_trip_1min_df["window_start"] = pd.to_datetime(vehicle_trip_1min_df["window_start"])

    return vehicle_trip_1min_df

# Trafikverket Traffic Data Fetching & Parsing
def build_situation_query(day_start: str, day_end: str) -> str:
    """Build XML query for Trafikverket API."""
    return f"""
    <REQUEST>
      <LOGIN authenticationkey="{TRAFIKVERKET_KEY}" />
      <QUERY objecttype="Situation" schemaversion="1">
        <FILTER>
          <AND>
            <LT name="Deviation.StartTime" value="{day_end}" />
            <OR>
              <GT name="Deviation.EndTime" value="{day_start}" />
              <EQ name="Deviation.ValidUntilFurtherNotice" value="true" />
            </OR>
          </AND>
        </FILTER>
        <INCLUDE>Id</INCLUDE>
        <INCLUDE>Deleted</INCLUDE>
        <INCLUDE>Deviation.Id</INCLUDE>
        <INCLUDE>Deviation.StartTime</INCLUDE>
        <INCLUDE>Deviation.EndTime</INCLUDE>
        <INCLUDE>Deviation.ValidUntilFurtherNotice</INCLUDE>
        <INCLUDE>Deviation.Geometry</INCLUDE>
        <INCLUDE>Deviation.Header</INCLUDE>
        <INCLUDE>Deviation.LocationDescriptor</INCLUDE>
        <INCLUDE>Deviation.RoadNumber</INCLUDE>
        <INCLUDE>Deviation.RoadNumberNumeric</INCLUDE>
        <INCLUDE>Deviation.MessageCode</INCLUDE>
        <INCLUDE>Deviation.MessageType</INCLUDE>
        <INCLUDE>Deviation.SeverityCode</INCLUDE>
        <INCLUDE>Deviation.SeverityText</INCLUDE>
      </QUERY>
    </REQUEST>
    """

def fetch_trafikverket_situations(day_start: str, day_end: str) -> ET.Element:
    """Fetch traffic situations from Trafikverket API."""
    xml_query = build_situation_query(day_start, day_end)

    try:
        response = requests.post(
            TRAFIKVERKET_API_URL,
            data=xml_query.encode("utf-8"),
            headers={"Content-Type": "application/xml"},
            timeout=60
        )

        if response.status_code != 200:
            print(f"  API Error {response.status_code}: {response.text[:200]}")
            return None

        return ET.fromstring(response.content)
    except Exception as e:
        print(f"  Error fetching Trafikverket data: {e}")
        return None

def extract_geometry_wkt(geometry_el):
    """Extract WKT geometry from Geometry element."""
    if geometry_el is None:
        return None

    wgs84_point = geometry_el.find("Point/WGS84")
    if wgs84_point is not None:
        return wgs84_point.text

    wgs84_line = geometry_el.find("Line/WGS84")
    if wgs84_line is not None:
        return wgs84_line.text

    wgs84_simple = geometry_el.find("WGS84")
    return wgs84_simple.text if wgs84_simple is not None else None

def parse_wkt_centroid(wkt: str):
    """Extract lat/lon centroid from WKT geometry."""
    if not wkt:
        return None, None

    if wkt.startswith("POINT"):
        lon, lat = map(float, wkt.replace("POINT (", "").replace(")", "").split())
        return lat, lon

    if wkt.startswith("LINESTRING"):
        coords = [tuple(map(float, p.split())) for p in wkt.replace("LINESTRING (", "").replace(")", "").split(",")]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return np.mean(lats), np.mean(lons)

    return None, None

def extract_deviation_info(dev):
    """Extract deviation information from a Deviation element."""
    if dev is None:
        return {
            "road_number": None, "road_number_numeric": None, "start_time": None, "end_time": None,
            "header": None, "message_code": None, "message_type": None, "severity_code": None,
            "severity_text": None, "location_description": None, "valid_until_further_notice": False,
            "geometry_wkt": None, "lat": None, "lon": None, "deviation_id": None,
        }

    road_number = dev.findtext("RoadNumber")
    road_number_numeric = dev.findtext("RoadNumberNumeric")
    start_time = dev.findtext("StartTime")
    end_time = dev.findtext("EndTime")
    header = dev.findtext("Header")
    location_description = dev.findtext("LocationDescriptor")
    valid_until_further_notice = dev.findtext("ValidUntilFurtherNotice") == "true"
    deviation_id = dev.findtext("Id")
    message_code = dev.findtext("MessageCode")
    message_type = dev.findtext("MessageType")
    severity_code = dev.findtext("SeverityCode")
    severity_text = dev.findtext("SeverityText")

    geometry_el = dev.find("Geometry")
    geometry_wkt = extract_geometry_wkt(geometry_el)
    lat, lon = parse_wkt_centroid(geometry_wkt)

    return {
        "road_number": road_number, "road_number_numeric": road_number_numeric, "start_time": start_time,
        "end_time": end_time, "header": header, "message_code": message_code, "message_type": message_type,
        "severity_code": severity_code, "severity_text": severity_text, "location_description": location_description,
        "valid_until_further_notice": valid_until_further_notice, "geometry_wkt": geometry_wkt,
        "lat": lat, "lon": lon, "deviation_id": deviation_id,
    }

def parse_situations(root: ET.Element) -> pd.DataFrame:
    """Parse XML situations into DataFrame."""
    if root is None:
        return pd.DataFrame()

    rows = []
    for situation in root.findall(".//Situation"):
        situation_id = situation.findtext("Id")
        deleted = situation.findtext("Deleted") == "true"
        deviations = situation.findall("Deviation")

        if not deviations:
            deviations = [None]

        for dev in deviations:
            dev_info = extract_deviation_info(dev)
            rows.append({
                "event_id": situation_id, "deviation_id": dev_info["deviation_id"],
                "start_time": dev_info["start_time"], "end_time": dev_info["end_time"],
                "latitude": dev_info["lat"], "longitude": dev_info["lon"],
                "geometry_wkt": dev_info["geometry_wkt"], "affected_road": dev_info["road_number"],
                "road_number_numeric": dev_info["road_number_numeric"],
                "location_description": dev_info["location_description"],
                "message_code": dev_info["message_code"], "message_type": dev_info["message_type"],
                "severity_code": dev_info["severity_code"], "severity_text": dev_info["severity_text"],
                "header": dev_info["header"], "valid_until_further_notice": dev_info["valid_until_further_notice"],
                "deleted": deleted, "source": "trafikverket",
            })

    return pd.DataFrame(rows)


# Daily feature ingestion
def main():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    # Calculate yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    yesterday_ts = pd.Timestamp(yesterday.date(), tz='UTC')

    print(f"\n{'='*70}")
    print(f"Running daily feature pipeline for {yesterday.date()}")
    print(f"{'='*70}\n")

    tmp_dir = tempfile.mkdtemp()
    rt_rows_all = []

    try:
        # Fetch KODA RT vehicle data for all 24 hours
        print(f"Fetching KODA real-time vehicle positions...")
        for hour in range(24):
            print(f"  Fetching hour {hour}...")
            content = fetch_koda_rt_data(yesterday_str, hour)
            if content:
                rows = parse_vehiclepositions(content, yesterday_str, hour)
                rt_rows_all.extend(rows)

        if not rt_rows_all:
            print("  No RT data fetched. Exiting.")
            return

        rt_df = pd.DataFrame(rt_rows_all)
        rt_df["timestamp"] = pd.to_datetime(rt_df["timestamp"], utc=True)
        rt_df = rt_df[rt_df['trip_id'].notna() & rt_df['vehicle_id'].notna()]

        # Aggregate to 1-minute windows
        trip_agg_df = get_aggregated_temporal_trip_features(rt_df)
        if trip_agg_df.empty:
            print("   No aggregated data after processing. Exiting.")
            return

        print(f"    Fetched and aggregated {len(trip_agg_df)} trip windows")

        # Ingest to vehicle_trip_agg_fg
        print(f"  Ingesting to vehicle_trip_agg_fg...")
        vehicle_fg = fs.get_feature_group(name="vehicle_trip_agg_fg", version=2)
        vehicle_fg.insert(clean_column_names(trip_agg_df), write_options={"wait_for_job": False})
        print(f"  Ingested {len(trip_agg_df)} records to vehicle_trip_agg_fg")

        # Fetch Trafikverket traffic events
        print(f"\nFetching Trafikverket traffic events...")
        day_start = yesterday_str + "T00:00:00"
        day_end = yesterday_str + "T23:59:59"

        root = fetch_trafikverket_situations(day_start, day_end)
        traffic_df = parse_situations(root)

        if not traffic_df.empty:
            # Convert timestamps to UTC-aware datetime
            traffic_df["start_time"] = pd.to_datetime(traffic_df["start_time"], utc=True)
            traffic_df["end_time"] = pd.to_datetime(traffic_df["end_time"], errors="coerce", utc=True)

            # Remove rows with null start_time
            traffic_df = traffic_df.dropna(subset=["start_time"])

            # Fill missing end_time with start_time + 1 day
            missing_end = traffic_df["end_time"].isna()
            traffic_df.loc[missing_end, "end_time"] = traffic_df.loc[missing_end, "start_time"] + timedelta(days=1)

            # Type conversions
            traffic_df["latitude"] = pd.to_numeric(traffic_df["latitude"], errors="coerce")
            traffic_df["longitude"] = pd.to_numeric(traffic_df["longitude"], errors="coerce")
            traffic_df["road_number_numeric"] = pd.to_numeric(traffic_df["road_number_numeric"], errors="coerce")
            traffic_df["valid_until_further_notice"] = traffic_df["valid_until_further_notice"].astype(bool)
            traffic_df["deleted"] = traffic_df["deleted"].astype(bool)

            print(f"  Fetched {len(traffic_df)} traffic events")

            # Ingest to trafikverket_traffic_event_fg
            print(f"  Ingesting to trafikverket_traffic_event_fg...")
            traffic_fg = fs.get_feature_group(name="trafikverket_traffic_event_fg", version=2)
            traffic_fg.insert(clean_column_names(traffic_df), write_options={"wait_for_job": False})
            print(f"  Ingested {len(traffic_df)} records to trafikverket_traffic_event_fg")
        else:
            print(f"  No traffic events found for {yesterday.date()}")
            traffic_df = pd.DataFrame()

        # Fetch weather data for yesterday
        print(f"\nFetching weather data...")
        try:
            weather_df = fetch_weather_forecast(past_days=2, forecast_days=0)
            # Filter to just yesterday
            weather_df = weather_df[weather_df["date"] == yesterday.date()]

            if not weather_df.empty:
                print(f"  Fetched {len(weather_df)} hourly weather records")

                # Ingest to weather feature group
                weather_fg = fs.get_or_create_feature_group(
                    name="weather_hourly_fg",
                    version=1,
                    primary_key=["timestamp", "latitude", "longitude"],
                    event_time="timestamp",
                    online_enabled=False,
                    description="Hourly weather data from Open-Meteo for Östergötland region"
                )
                weather_fg.insert(clean_column_names(weather_df.copy()), write_options={"wait_for_job": False})
                print(f"  Ingested weather data to weather_hourly_fg")
            else:
                print(f"  No weather data available for {yesterday.date()}")
                weather_df = pd.DataFrame()
        except Exception as e:
            print(f"  Warning: Could not fetch weather data: {e}")
            weather_df = pd.DataFrame()

        # Fetch holiday data for yesterday
        print(f"\nFetching holiday data...")
        try:
            holiday_features = get_holiday_features_for_date(yesterday)
            print(f"  Holiday features: work_free={holiday_features['is_work_free']}, "
                  f"red_day={holiday_features['is_red_day']}, "
                  f"holiday={holiday_features['holiday_name']}")
        except Exception as e:
            print(f"  Warning: Could not fetch holiday data: {e}")
            holiday_features = {
                "is_work_free": False,
                "is_red_day": False,
                "is_day_before_holiday": False,
                "is_holiday": False,
                "holiday_name": None,
            }

        # Create enriched features combining trip + traffic + weather + holidays
        print(f"\nCreating enriched features...")

        if traffic_df.empty:
            enriched_df = trip_agg_df.copy()
            enriched_df["has_nearby_event"] = 0
            enriched_df["num_nearby_traffic_events"] = 0
            enriched_df["nearby_event_severity"] = None
            enriched_df["affected_roads"] = None
        else:
            # For each trip, find nearby traffic events
            enriched_data = []
            for idx, trip_row in trip_agg_df.iterrows():
                window_start = trip_row["window_start"]
                trip_lat, trip_lon = trip_row["lat_mean"], trip_row["lon_mean"]

                # Find active traffic events within 500m
                if pd.notna(trip_lat) and pd.notna(trip_lon):
                    nearby = traffic_df[
                        (traffic_df["start_time"] <= window_start) &
                        (traffic_df["end_time"] >= window_start)
                    ].copy()

                    nearby["distance_m"] = nearby.apply(
                        lambda r: distance_m(trip_lat, trip_lon, r["latitude"], r["longitude"]),
                        axis=1
                    )
                    nearby = nearby[nearby["distance_m"] <= 500]
                else:
                    nearby = pd.DataFrame()

                if nearby.empty:
                    enriched_data.append({
                        "has_nearby_event": 0,
                        "num_nearby_traffic_events": 0,
                        "nearby_event_severity": None,
                        "affected_roads": None,
                    })
                else:
                    severity_codes = nearby["severity_code"].dropna()
                    max_severity = int(severity_codes.max()) if len(severity_codes) > 0 else None
                    affected_roads = nearby["affected_road"].dropna().unique()
                    roads_str = ",".join(str(r) for r in affected_roads) if len(affected_roads) > 0 else None

                    enriched_data.append({
                        "has_nearby_event": int(len(nearby) > 0),
                        "num_nearby_traffic_events": len(nearby),
                        "nearby_event_severity": max_severity,
                        "affected_roads": roads_str,
                    })

            enriched_features = pd.DataFrame(enriched_data)
            enriched_df = pd.concat([trip_agg_df.reset_index(drop=True), enriched_features], axis=1)

        # Add weather features by joining on hour
        if not weather_df.empty:
            print(f"  Adding weather features...")
            weather_hourly = weather_df[["hour", "temperature_2m", "precipitation", "rain",
                                          "snowfall", "cloud_cover", "wind_speed_10m", "weather_code"]].copy()
            weather_hourly = weather_hourly.drop_duplicates(subset=["hour"])

            enriched_df = enriched_df.merge(weather_hourly, on="hour", how="left")
            print(f"  Added weather features: temperature, precipitation, cloud_cover, wind_speed, etc.")
        else:
            # Add empty weather columns if no weather data
            enriched_df["temperature_2m"] = None
            enriched_df["precipitation"] = None
            enriched_df["rain"] = None
            enriched_df["snowfall"] = None
            enriched_df["cloud_cover"] = None
            enriched_df["wind_speed_10m"] = None
            enriched_df["weather_code"] = None

        # Add holiday features (same for all rows since it's a single day)
        enriched_df["is_work_free"] = holiday_features["is_work_free"]
        enriched_df["is_red_day"] = holiday_features["is_red_day"]
        enriched_df["is_day_before_holiday"] = holiday_features["is_day_before_holiday"]
        enriched_df["is_holiday"] = holiday_features["is_holiday"]

        # Ingest to combined feature group
        print(f"  Creating/updating trip_occupancy_features_daily...")
        enriched_fg = fs.get_or_create_feature_group(
            name="trip_occupancy_features_daily",
            version=1,
            primary_key=["trip_id", "window_start"],
            event_time="window_start",
            online_enabled=False,
            description="Daily enriched trip features with vehicle metrics, traffic events, weather, and holidays"
        )

        enriched_df.columns = (
            enriched_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        enriched_fg.insert(clean_column_names(enriched_df), write_options={"wait_for_job": False})
        print(f"  Ingested {len(enriched_df)} enriched records to trip_occupancy_features_daily")

        print(f"\n{'='*70}")
        print(f"Daily feature pipeline completed successfully for {yesterday.date()}")
        print(f"{'='*70}\n")

    finally:
        # Cleanup
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        gc.collect()


if __name__ == "__main__":
    main()
